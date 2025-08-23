from datasets import load_dataset # HuggingFace Datasets

ds = load_dataset("JeanKaddour/minipile") # 24(Approx. 250MB) files; Approx. 6GB

print_ds_example = '''
>>> print(ds)
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 1000000
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 500
    })
    test: Dataset({
        features: ['text'],
        num_rows: 10000
    })
})
'''