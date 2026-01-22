import os

downloaded_datasets_root_dir = "downloaded_datasets"
dataset = {
    "dataset_name": "Large_Movie_Review_Dataset",
    "URL": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
}

if downloaded_datasets_root_dir in os.listdir('.'):
    print('[INFO] Root datasets dir already exist.')
else:
    print("[INFO] Root datasets dir doesn't exist on disk. Creating...")
    os.system(f"mkdir {downloaded_datasets_root_dir}")

if dataset["dataset_name"] in os.listdir(f'{downloaded_datasets_root_dir}'):
    print(f"[INFO] Dataset '{dataset["dataset_name"]}' already downloaded.")
else:
    print(f"[INFO] Dataset dir '{dataset["dataset_name"]}' doesn't exist on disk.")
    print(f"[INFO] Creating dataset dir '{dataset["dataset_name"]}'...")
    os.system(f"mkdir {downloaded_datasets_root_dir}/{dataset["dataset_name"]}")
    print(f"[INFO] Downloading into {dataset["dataset_name"]}'...")
    # download Large Movie Review Dataset as archive:
    os.system(f"curl {dataset["URL"]} --output {downloaded_datasets_root_dir}/\{dataset['dataset_name']}/large_movie_review_dataset.tar.gz")
    # unzip dataset:
    print("[INFO] Unzipping and preparing for further py-scripts...")
    os.system(f"gzip -d {downloaded_datasets_root_dir}/{dataset['dataset_name']}/large_movie_review_dataset.tar.gz")
    os.system(f"tar -xf {downloaded_datasets_root_dir}/{dataset['dataset_name']}/large_movie_review_dataset.tar")
    # move files from aclImdb dir -> {dataset["dataset_name"]} dir:
    os.system(f"mv aclImdb/* {downloaded_datasets_root_dir}/{dataset["dataset_name"]}")
    # remove empty aclImdb dir:
    os.system(f"rm -r aclImdb")
