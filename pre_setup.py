import subprocess
import sys

def install_missing_packages(dependencies):
    missing = []
    for dep in dependencies:
        package_name = dep.split('>=')[0].split('<=')[0].split('==')[0].split('>')[0].split('<')[0]
        try:
            __import__(package_name)
        except ImportError:
            print(f"✗ {package_name} missing...")
            missing.append(dep)
    
    for package in missing:        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        except subprocess.CalledProcessError:
            print(f"✗ Install {package} failed...")
            continue

# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/transformers/dependency_versions_table.py
_deps = [
    "Pillow>=10.0.1,<=15.0",
    "accelerate>=0.26.0",
    "av",
    "beautifulsoup4",
    "blobfile",
    "codecarbon>=2.8.1",
    "cookiecutter==1.7.3",
    "dataclasses",
    "datasets>=2.15.0",  # We need either this pin or pyarrow<21.0.0
    "deepspeed>=0.9.3",
    "diffusers",
    "dill<0.3.5",
    "evaluate>=0.2.0",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flax>=0.4.1,<=0.7.0",
    "ftfy",
    "fugashi>=1.0",
    "GitPython<3.1.19",
    "hf-doc-builder>=0.3.0",
    "hf_xet",
    "huggingface-hub>=0.34.0,<1.0",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "jax>=0.4.1,<=0.4.13",
    "jaxlib>=0.4.1,<=0.4.13",
    "jieba",
    "jinja2>=3.1.0",
    "kenlm",
    # Keras pin - this is to make sure Keras 3 doesn't destroy us. Remove or change when we have proper support.
    "keras>2.9,<2.16",
    "keras-nlp>=0.3.1,<0.14.0",  # keras-nlp 0.14 doesn't support keras 2, see pin on keras.
    "kernels>=0.6.1,<0.7",
    "librosa",
    "natten>=0.14.6,<0.15.0",
    "nltk<=3.8.1",
    "num2words",
    "numpy>=1.17",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "openai",
    "opencv-python",
    "optimum-benchmark>=0.3.0",
    "optuna",
    "optax>=0.0.8,<=0.1.4",
    "pandas<2.3.0",  # `datasets` requires `pandas` while `pandas==2.3.0` has issues with CircleCI on 2025/06/05
    "packaging>=20.0",
    "parameterized>=0.9",  # older version of parameterized cause pytest collection to fail on .expand
    "phonemizer",
    "protobuf",
    "psutil",
    "pyyaml>=5.1",
    "pydantic>=2",
    "pytest>=7.2.0",
    "pytest-asyncio",
    "pytest-rerunfailures",
    "pytest-timeout",
    "pytest-xdist",
    "pytest-order",
    "python>=3.9.0",
    "ray[tune]>=2.7.0",
    "regex!=2019.12.17",
    "requests",
    "rhoknp>=1.1.0,<1.3.1",
    "rjieba",
    "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",
    "ruff==0.11.2",
    # `sacrebleu` not used in `transformers`. However, it is needed in several tests, when a test calls
    # `evaluate.load("sacrebleu")`. This metric is used in the examples that we use to test the `Trainer` with, in the
    # `Trainer` tests (see references to `run_translation.py`).
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "safetensors>=0.4.3",
    "sagemaker>=2.31.0",
    "schedulefree>=1.2.6",
    "scikit-learn",
    "scipy<1.13.0",  # SciPy >= 1.13.0 is not supported with the current jax pin (`jax>=0.4.1,<=0.4.13`)
    "sentencepiece>=0.1.91,!=0.1.92",
    "sigopt",
    "starlette",
    "sudachipy>=0.6.6",
    "sudachidict_core>=20220729",
    "tensorboard",
    # TensorFlow pin. When changing this value, update examples/tensorflow/_tests_requirements.txt accordingly
    "tensorflow-cpu>2.9,<2.16",
    "tensorflow>2.9,<2.16",
    "tensorflow-text<2.16",
    "tensorflow-probability<0.24",
    "tf2onnx",
    "timeout-decorator",
    "tiktoken",
    "timm<=1.0.19,!=1.0.18",
    "tokenizers>=0.21,<0.22",
    "torch>=2.1",
    "torchaudio",
    "torchvision",
    "pyctcdecode>=0.4.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "urllib3<2.0.0",
    "uvicorn",
    "pytest-rich",
    "libcst",
    "rich",
    "opentelemetry-api",
    "opentelemetry-exporter-otlp",
    "opentelemetry-sdk",
    "mistral-common[opencv]>=1.6.3",
]

install_missing_packages (_deps)

print("Done.")