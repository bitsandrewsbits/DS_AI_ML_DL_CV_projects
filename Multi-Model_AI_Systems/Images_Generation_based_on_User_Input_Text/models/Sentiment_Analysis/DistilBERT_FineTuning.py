# DistilBERT model fine-tuning
from transformers import AutoTokenizer

parent_dir = "data_collection_and_preprocessing"
datasets_filename = "train_validation_test_datasets.json"

def main():
    train_val_test_datasets = load_train_validation_test_datasets()
    print(train_val_test_datasets)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    # tokenized_datasets = train_val_test_datasets.map(
    #     lambda dataset: tokenizer(dataset["text"], truncation = True),
    #     batched = True
    # )

# TODO: create func with respect to changed datasets data structure
def load_train_validation_test_datasets() -> DatasetDict:
    pass

if __name__ == '__main__':
    main()
