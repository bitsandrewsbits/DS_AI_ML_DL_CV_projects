# Train, Validation, Test datasets creation for DistilBERT fine-tuning
import DistilBERT_FineTuning_dataset_creation as ft_ds
from sklearn.model_selection import train_test_split
import pandas as pd
import additional_functions_for_data_preprocessing as ad_fncs
from datasets import DatasetDict, Dataset

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

datasets_save_path = "."
Class_Samples_Amount = 100

def main(datasets_files: dict, save_path: str, class_samples_amount = 'all'):
    datasets = get_datasets_from_reviews_files(datasets_files, class_samples_amount)
    target_dataset_for_split = get_merged_train_test_datasets_into_one(datasets)
    dataset_with_neutral_reviews = include_neutral_reviews_to_result_dataset(
        target_dataset_for_split
    )
    train_val_test_datasets = get_train_validation_test_datasets(
        dataset_with_neutral_reviews
    )
    print(train_val_test_datasets)
    converted_train_val_test_datasets = get_converted_datasets_into_DatasetDict_Dataset(
        train_val_test_datasets
    )
    save_train_val_test_datasets_to_disk(
        converted_train_val_test_datasets, save_path
    )

def save_train_val_test_datasets_to_disk(datasets: DatasetDict, save_path: str):
    datasets.save_to_disk(
        dataset_dict_path = save_path,
        storage_options = {}
    )
    print('Train, Validation, Test datasets saved.')

def get_train_validation_test_datasets(target_dataset: pd.DataFrame):
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

def get_converted_datasets_into_DatasetDict_Dataset(datasets: dict[pd.DataFrame]):
    for dataset in datasets:
        datasets[dataset] = Dataset.from_pandas(
            datasets[dataset], preserve_index = False
        )
    return DatasetDict(datasets)

def get_datasets_from_reviews_files(
datasets_files_info: dict, class_samples_amount: int) -> dict[pd.DataFrame]:
    result_datasets = {}
    for dataset_type in datasets_files_info:
        dataset_creation = ft_ds.DistilBERT_Fune_Tuning_Dataset_Creation(
            datasets_files_info[dataset_type]["positive_reviews_path"],
            datasets_files_info[dataset_type]["negative_reviews_path"]
        )
        result_datasets[dataset_type] = dataset_creation.main(class_samples_amount)
    return result_datasets

def get_merged_train_test_datasets_into_one(
train_test_datasets: dict[pd.DataFrame]) -> pd.DataFrame:
    target_dataset_for_splitting = pd.concat(
        [train_test_datasets["train"], train_test_datasets["test"]],
        axis = 0, ignore_index = True
    )
    return target_dataset_for_splitting

def include_neutral_reviews_to_result_dataset(res_dataset: pd.DataFrame,
oversampling = False):
    neutral_reviews_df = pd.read_json(
        "neutral_reviews.json", orient = 'records', lines = True
    )
    # TODO: test without neutral reviews oversampling
    if oversampling:
        if neutral_reviews_df.shape[0] < res_dataset.shape[0]:
            oversample_coefficient = get_oversampling_coefficient(
                neutral_reviews_df.shape[0], res_dataset.shape[0]
            )
        neutral_reviews_df = get_oversampled_dataset(neutral_reviews_df)

    reviews_3_types_dataset = pd.concat(
        [res_dataset, neutral_reviews_df],
        axis = 0, ignore_index = True
    )
    return reviews_3_types_dataset

# TODO: test and tune after training without oversampling
def get_oversampling_coefficient(real_reviews_amount, target_reviews_amount):
    oversampling_coeff = target_reviews_amount // real_reviews_amount
    if oversampling_coeff > 100:
        oversampling_coeff //= 100
    return oversampling_coeff

# TODO: create function
def get_oversampled_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    pass

if __name__ == '__main__':
    main(datasets_files_info, datasets_save_path, Class_Samples_Amount)
