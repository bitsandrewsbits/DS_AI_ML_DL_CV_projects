# additional functions
import os
import pandas as pd

def get_filenames_from_dir(dir_relative_path: str):
    return os.listdir(dir_relative_path)

def create_directory(dir_path: str):
    if dir_path in os.listdir('.'):
        print(f"[INFO] Dir {dir_path} already exist!")
    else:
        print(f"[INFO] Creating {dir_path} dir.")
        os.mkdir(dir_path)

def save_dataset_into_JSONL(df: pd.DataFrame, dataset_path: str):
    df.to_json(
        dataset_path, orient = 'records', lines = True
    )

if __name__ == '__main__':
    filenames = get_filenames_from_dir('data/train/pos')
    print(filenames)
