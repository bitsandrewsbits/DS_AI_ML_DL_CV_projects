# Train, Validation, Test datasets creation for DistilBERT fine-tuning
import one_reviews_dir_dataset_creation as rd_ds
from sklearn.model_selection import train_test_split
import pandas as pd
import additional_functions_for_data_preprocessing as ad_fncs
import download_datasets as load_ds
from datasets import DatasetDict, Dataset

path_to_downloaded_dataset = f"{load_ds.downloaded_datasets_root_dir}/{load_ds.dataset['dataset_name']}"

datasets_files_info = {
    "train": {
        "positive_reviews_path": 'pos',
        "negative_reviews_path": 'neg'
    },
    "test": {
        "positive_reviews_path": 'pos',
        "negative_reviews_path": 'neg'
    }
}

class_samples_amount = 8000  # by default - entire dataset
root_prepared_datasets_dir = "prepared_datasets"
datasets_save_path = f"{root_prepared_datasets_dir}/train_val_test_datasets"

def main(datasets_files: dict, save_path: str, class_samples_amount = 'all', oversample_neutral_reviews = True):
    if root_prepared_datasets_dir in ad_fncs.get_filenames_from_dir('.'):
        print("[INFO] Prepared Datasets dir already exist.")
    else:
        print("[INFO] Creating root dir for prepared datasets...")
        ad_fncs.create_directory(root_prepared_datasets_dir)

    datasets = get_datasets_from_reviews_files(datasets_files, class_samples_amount)
    target_dataset_for_split = get_merged_train_test_datasets_into_one(datasets)
    dataset_with_neutral_reviews = include_neutral_reviews_to_result_dataset(
        target_dataset_for_split
    )
    train_val_test_datasets = get_train_validation_test_datasets(
        dataset_with_neutral_reviews
    )
    if oversample_neutral_reviews:
      neutral_reviews_distribution_in_datasets = get_neutral_reviews_distribution_in_datasets(
          train_val_test_datasets
      )
      neutral_reviews_oversampling_coeffs = get_neutral_reviews_oversampling_coeffs_for_datasets(
          train_val_test_datasets, neutral_reviews_distribution_in_datasets
      )
      neutral_reviews_oversampled_datasets = get_neutral_reviews_oversampled_datasets(
          train_val_test_datasets,
          neutral_reviews_distribution_in_datasets,
          neutral_reviews_oversampling_coeffs
      )
      converted_train_val_test_datasets = get_converted_datasets_into_DatasetDict_Dataset(
        neutral_reviews_oversampled_datasets
      )
    # TODO: think, how to check and open already existed datasets from disk
    # so as to avoid recreating dataset from scratch.
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
datasets_files_info: dict, class_samples_amount) -> dict[pd.DataFrame]:
    result_datasets = {}
    for dataset_type in datasets_files_info:
        positive_reviews_dataset_creator = rd_ds.One_Reviews_Dir_Dataset_Creator(
            f"{path_to_downloaded_dataset}/{dataset_type}",
            datasets_files_info[dataset_type]["positive_reviews_path"]
        )
        positive_reviews_dataset = positive_reviews_dataset_creator.main(class_samples_amount)
        negative_reviews_dataset_creator = rd_ds.One_Reviews_Dir_Dataset_Creator(
            f"{path_to_downloaded_dataset}/{dataset_type}",
            datasets_files_info[dataset_type]["negative_reviews_path"]
        )
        negative_reviews_dataset = negative_reviews_dataset_creator.main(class_samples_amount)

        pos_neg_reviews_dataset = pd.concat(
            [positive_reviews_dataset, negative_reviews_dataset],
            ignore_index = True
        )
        result_datasets[dataset_type] = pos_neg_reviews_dataset
        print(result_datasets[dataset_type])
    return result_datasets

def get_merged_train_test_datasets_into_one(
train_test_datasets: dict[pd.DataFrame]) -> pd.DataFrame:
    target_dataset_for_splitting = pd.concat(
        [train_test_datasets["train"], train_test_datasets["test"]],
        axis = 0, ignore_index = True
    )
    return target_dataset_for_splitting

def include_neutral_reviews_to_result_dataset(res_dataset: pd.DataFrame):
    neutral_reviews_df = pd.read_json(
        "neutral_reviews.json", orient = 'records', lines = True
    )
    neutral_reviews_df['label'] = 2
    reviews_3_types_dataset = pd.concat(
        [res_dataset, neutral_reviews_df],
        axis = 0, ignore_index = True
    )
    return reviews_3_types_dataset

def get_neutral_reviews_oversampled_datasets(datasets: dict[pd.DataFrame],
neutral_reviews_distribution: dict, neutral_reviews_oversampling_coeffs: dict) -> pd.DataFrame:
  for dataset_type in datasets:
    current_dataset = datasets[dataset_type]
    for _ in range(neutral_reviews_oversampling_coeffs[dataset_type] - 1):
      current_dataset = pd.concat(
        [
          current_dataset,
          neutral_reviews_distribution[dataset_type]
        ],
        ignore_index = True
      )
    datasets[dataset_type] = current_dataset
  return datasets

def get_oversampling_coefficient(real_reviews_amount: int, target_reviews_amount: int):
    oversampling_coeff = target_reviews_amount // real_reviews_amount
    return oversampling_coeff

def get_neutral_reviews_oversampling_coeffs_for_datasets(datasets: dict[pd.DataFrame], neutral_reviews_distribution: dict):
  neutral_reviews_oversampling_coeffs = {}
  for dataset_type in datasets:
    positive_reviews = datasets[dataset_type][datasets[dataset_type]['label'] == 1]
    neutral_reviews_oversampling_coeffs[dataset_type] = get_oversampling_coefficient(
        neutral_reviews_distribution[dataset_type].shape[0], positive_reviews.shape[0]
    )
  return neutral_reviews_oversampling_coeffs

def get_neutral_reviews_distribution_in_datasets(datasets: dict[pd.DataFrame]):
  datasets_neutral_reviews_distrition = {}
  for dataset_type in datasets:
    datasets_neutral_reviews_distrition[dataset_type] = get_reviews_from_df_by_label(
        datasets[dataset_type], 2
    )
  return datasets_neutral_reviews_distrition

def get_reviews_from_df_by_label(dataset: pd.DataFrame, label: int) -> pd.DataFrame:
  return dataset[dataset['label'] == label]

if __name__ == "__main__":
    print('Creating prepared train-validation-test datasets...')
    main(datasets_files_info, datasets_save_path, class_samples_amount)
