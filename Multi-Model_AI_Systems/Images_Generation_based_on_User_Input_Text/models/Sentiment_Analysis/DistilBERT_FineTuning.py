# DistilBERT model fine-tuning
from transformers import AutoTokenizer
from datasets import DatasetDict

datasets_parent_dir = "data_collection_and_preprocessing"

def main():
    train_val_test_datasets = load_train_val_test_datasets(datasets_parent_dir)
    print(train_val_test_datasets)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    tokenized_datasets = train_val_test_datasets.map(
        lambda dataset: tokenizer(dataset["text"], truncation = True),
        batched = True
    )

def load_train_val_test_datasets(datasets_parent_dir_path: str) -> DatasetDict:
    loaded_datasets = DatasetDict.load_from_disk(
        dataset_dict_path = datasets_parent_dir_path
    )
    return loaded_datasets

if __name__ == '__main__':
    main()
