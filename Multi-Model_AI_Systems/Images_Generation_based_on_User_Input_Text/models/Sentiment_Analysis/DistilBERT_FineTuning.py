# DistilBERT model fine-tuning
import json

parent_dir = "data_collection_and_preprocessing"
datasets_filename = "train_validation_test_datasets.json"

def main():
    train_val_test_datasets = load_train_validation_test_datasets()

def load_train_validation_test_datasets():
    with open(f"{parent_dir}/{datasets_filename}", 'r') as datasets_f:
        datasets = json.load(datasets_f)
    return datasets

if __name__ == '__main__':
    main()
