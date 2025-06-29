# DistilBERT model fine-tuning
import json
from transformers import AutoTokenizer

parent_dir = "data_collection_and_preprocessing"
datasets_filename = "train_validation_test_datasets.json"

def main():
    train_val_test_datasets = load_train_validation_test_datasets()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def load_train_validation_test_datasets():
    with open(f"{parent_dir}/{datasets_filename}", 'r') as datasets_f:
        loaded_datasets = json.load(datasets_f)
    return loaded_datasets

def preprocess_review_text(review: dict):
    return tokenizer(review["text"], truncation = True)

if __name__ == '__main__':
    main()
