# Train, Validation, Test datasets creation for DistilBERT fine-tuning
import DistilBERT_FineTuning_dataset_creation as ft_ds

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

def get_datasets_from_reviews_files(datasets_files_info: dict) -> dict[list]:
    result_datasets = {}
    for dataset_type in datasets_files_info:
        dataset_creation = ft_ds.DistilBERT_Fune_Tuning_Dataset_Creation(
            datasets_files_info[dataset_type]["positive_reviews_path"],
            datasets_files_info[dataset_type]["negative_reviews_path"]
        )
        result_datasets[dataset_type] = dataset_creation.main()

    return result_datasets

# TODO: create method, maybe need to rename it.
def get_merged_train_test_datasets(train_dataset: list, test_dataset: list):
    pass

if __name__ == '__main__':
    datasets = get_datasets_from_reviews_files(datasets_files_info)
    print(datasets)
