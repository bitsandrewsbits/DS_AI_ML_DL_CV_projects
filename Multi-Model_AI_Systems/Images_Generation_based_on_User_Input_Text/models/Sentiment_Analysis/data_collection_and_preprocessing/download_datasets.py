import os

datasets_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
downloaded_datasets_dir = "downloaded_datasets"

if downloaded_datasets_dir in os.listdir('.'):
  print('[INFO] Train-Validation-Test dataset already exist and prepared.')
else:
  print("[INFO] Datasets don't exist on disk. Downloading...")
  os.system(f"mkdir {downloaded_datasets_dir}")
  os.system(f"cd {downloaded_datasets_dir}")
  # download Large Movie Review Dataset as archive:
  os.system(f"curl --output large_movie_review_dataset.tar.gz {datasets_URL}")
  # unzip dataset:
  os.system("gzip -d large_movie_review_dataset.tar.gz")
  os.system("tar -xf large_movie_review_dataset.tar")
  # move files from aclImdb dir -> just data:
  os.system(f"mv aclImdb/* {downloaded_datasets_dir}/")
  # remove empty aclImdb dir:
  os.system(f"rm -r aclImdb")