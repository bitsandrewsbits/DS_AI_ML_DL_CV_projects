import os
import data_preprocessing_variables as dpv


def download_large_movie_reviews_dataset(dataset_info: dict):
    downloaded_datasets_root_dir = dpv.DOWNLOADED_DATASETS_ROOT_DIR
    dataset_info = dpv.USERS_SENTIMENTS_DATASET_INFO

    if downloaded_datasets_root_dir in os.listdir('.') or\
    downloaded_datasets_root_dir in os.listdir('../'):
        print('[INFO] Root datasets dir already exist.')
    else:
        print("[INFO] Root datasets dir doesn't exist on disk. Creating...")
        os.system(f"mkdir {downloaded_datasets_root_dir}")

    if dataset_info["dataset_name"] in os.listdir(f'../{downloaded_datasets_root_dir}') or\
    dataset_info["dataset_name"] in os.listdir(f'{downloaded_datasets_root_dir}'):
        print(f"[INFO] Dataset '{dataset_info["dataset_name"]}' already downloaded.")
    else:
        print(f"[INFO] Dataset dir '{dataset_info["dataset_name"]}' doesn't exist on disk.")
        print(f"[INFO] Creating dataset dir '{dataset_info["dataset_name"]}'...")
        os.system(f"mkdir {downloaded_datasets_root_dir}/{dataset_info["dataset_name"]}")
        print(f"[INFO] Downloading into {dataset_info["dataset_name"]}'...")
        # download Large Movie Review Dataset as archive:
        os.system(f"curl {dataset_info["URL"]} --output {downloaded_datasets_root_dir}/{dataset_info['dataset_name']}/large_movie_review_dataset.tar.gz")
        # unzip dataset:
        print("[INFO] Unzipping and preparing for further py-scripts...")
        os.system(f"gzip -d {downloaded_datasets_root_dir}/{dataset_info['dataset_name']}/large_movie_review_dataset.tar.gz")
        os.system(f"tar -xf {downloaded_datasets_root_dir}/{dataset_info['dataset_name']}/large_movie_review_dataset.tar")
        # move files from aclImdb dir -> {dataset["dataset_name"]} dir:
        os.system(f"mv aclImdb/* {downloaded_datasets_root_dir}/{dataset_info["dataset_name"]}")
        # remove empty aclImdb dir:
        os.system(f"rm -r aclImdb")

if __name__ == "__main__":
    download_large_movie_reviews_dataset(dpv.USERS_SENTIMENTS_DATASET_INFO)
