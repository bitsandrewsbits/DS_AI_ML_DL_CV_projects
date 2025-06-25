# Train, Validation, Test datasets creation for DistilBERT fine-tuning
import DistilBERT_FineTuning_dataset_creation as ft_ds
from sklearn.model_selection import train_test_split

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

def main(datasets_files: dict) -> dict[list]:
    datasets = get_datasets_from_reviews_files(datasets_files)
    target_dataset_for_split = get_merged_train_test_datasets_into_one(datasets)
    train_val_test_datasets = get_train_validation_test_datasets(
        target_dataset_for_split
    )
    return train_val_test_datasets

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

def get_datasets_from_reviews_files(datasets_files_info: dict) -> dict[list]:
    result_datasets = {}
    for dataset_type in datasets_files_info:
        dataset_creation = ft_ds.DistilBERT_Fune_Tuning_Dataset_Creation(
            datasets_files_info[dataset_type]["positive_reviews_path"],
            datasets_files_info[dataset_type]["negative_reviews_path"]
        )
        result_datasets[dataset_type] = dataset_creation.main()

    return result_datasets

def get_merged_train_test_datasets_into_one(train_test_datasets: dict):
    target_dataset_for_splitting = []
    target_dataset_for_splitting += train_test_datasets['train']
    target_dataset_for_splitting += train_test_datasets['test']
    return target_dataset_for_splitting

if __name__ == '__main__':
    result_datasets = main(datasets_files_info)
    print(result_datasets)
