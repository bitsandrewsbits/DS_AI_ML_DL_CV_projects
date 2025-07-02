# Train, Validation, Test datasets creation for DistilBERT fine-tuning
import DistilBERT_FineTuning_dataset_creation as ft_ds
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import additional_functions_for_data_preprocessing as ad_fncs

datasets_files_info = {
    "train": {
        "positive_reviews_path": 'data/train/pos',
        "negative_reviews_path": 'data/train/neg'
    },
    "test": {
        "positive_reviews_path": 'data/test/pos',
        "negative_reviews_path": 'data/test/neg'
    }
}

def main(datasets_files: dict):
    datasets = get_datasets_from_reviews_files(datasets_files)
    target_dataset_for_split = get_merged_train_test_datasets_into_one(datasets)
    train_val_test_datasets = get_train_validation_test_datasets(
        target_dataset_for_split
    )
    train_val_test_datasets_in_dict_type = get_converted_datasets_into_dict(
        train_val_test_datasets
    )
    save_train_val_test_datasets_into_JSON_file(train_val_test_datasets_in_dict_type)

def save_train_val_test_datasets_into_JSON_file(datasets: dict):
    with open("train_validation_test_datasets.json", 'w') as datasets_f:
        json.dump(datasets, datasets_f)
        print('Train, Validation, Test datasets saved into train_validation_test_datasets.json')

def get_train_validation_test_datasets(target_dataset: list):
    train_dataset_for_split, test_dataset = train_test_split(
        target_dataset, test_size = 0.1, shuffle = True
    )
    train_dataset, validation_dataset = train_test_split(
        train_dataset_for_split, train_size = 0.92, shuffle = True
    )
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    }

def get_converted_datasets_into_dict(datasets: dict[pd.DataFrame]) -> dict[dict]:
    converted_datasets = {
        "train": datasets["train"].to_dict(orient = 'list'),
        "validation": datasets["validation"].to_dict(orient = 'list'),
        "test": datasets["test"].to_dict(orient = 'list')
    }
    return converted_datasets

def get_datasets_from_reviews_files(datasets_files_info: dict) -> dict[pd.DataFrame]:
    result_datasets = {}
    for dataset_type in datasets_files_info:
        dataset_creation = ft_ds.DistilBERT_Fune_Tuning_Dataset_Creation(
            datasets_files_info[dataset_type]["positive_reviews_path"],
            datasets_files_info[dataset_type]["negative_reviews_path"]
        )
        result_datasets[dataset_type] = dataset_creation.main()

    return result_datasets

def get_merged_train_test_datasets_into_one(
train_test_datasets: dict[pd.DataFrame]) -> pd.DataFrame:
    target_dataset_for_splitting = pd.concat(
        [train_test_datasets["train"], train_test_datasets["test"]],
        axis = 0, ignore_index = True
    )
    return target_dataset_for_splitting

if __name__ == '__main__':
    main(datasets_files_info)
