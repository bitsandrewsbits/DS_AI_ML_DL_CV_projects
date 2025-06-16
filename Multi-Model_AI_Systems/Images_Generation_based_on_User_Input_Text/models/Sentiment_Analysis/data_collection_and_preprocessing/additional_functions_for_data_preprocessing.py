# additional functions
import os

def get_filenames_from_dir(dir_relative_path: str):
    return os.listdir(dir_relative_path)

if __name__ == '__main__':
    filenames = get_filenames_from_dir('data/train/pos')
    print(filenames)
