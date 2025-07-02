# DistilBERT model fine-tuning
import json
from transformers import AutoTokenizer
from datasets import DatasetDict

parent_dir = "data_collection_and_preprocessing"
datasets_filename = "train_validation_test_datasets.json"

def main():
    train_val_test_datasets = load_train_validation_test_datasets()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    # tokenized_datasets = train_val_test_datasets.map(
    #     lambda dataset: tokenizer(dataset["text"], truncation = True),
    #     batched = True
    # )

def load_train_validation_test_datasets() -> DatasetDict:
    with open(f"{parent_dir}/{datasets_filename}", 'r') as datasets_f:
        loaded_datasets = json.load(datasets_f)
    return DatasetDict(loaded_datasets)

if __name__ == '__main__':
    main()
