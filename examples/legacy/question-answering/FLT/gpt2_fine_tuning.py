from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
import re
from pathlib import Path

def extract_lemmas(mathlib_path):
    pattern = r'lemma\s+(\w+).*?:\s*(.+?)(?=:=|by)'
    lemmas = []
    for lean_file in Path(mathlib_path).rglob("*.lean"):
        content = lean_file.read_text()
        matches = re.findall(pattern, content, re.DOTALL)
        lemmas.extend([f"<LEMMA>{name}<SIG>{sig.strip()}" for name, sig in matches])
    return lemmas

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # attention layers only
)

model = get_peft_model(model, lora_config)
# Only ~0.3% parameters trainable (1M vs 355M)

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_dict({"text": extract_lemmas("./mathlib4")})
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512))

trainer = Trainer(
    model=model,  # LoRA model from before
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        output_dir="./lean-lemma-matcher",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=100,
    )
)
trainer.train()

def find_similar_lemmas(query_sig, top_k=5):
    prompt = f"<SIG>{query_sig}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256, num_return_sequences=top_k, 
                           temperature=0.3, do_sample=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)