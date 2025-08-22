from datasets import load_dataset # HuggingFace Datasets

ds = load_dataset("JeanKaddour/minipile") # 24 x ~250 MB files, approx. 6 GB

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