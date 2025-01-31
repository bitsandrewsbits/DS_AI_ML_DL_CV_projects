# this is program for analyzing your free RAM and
# compute optimal size of minidatasets for reading, changing operations.
# Minidatasets will be writting into disk into dir where original large dataset exists.

import psutil
import os
import shutil
import pandas as pd

class Dataset_Splitting_by_free_RAM:
    def __init__(self, csv_dataset_path: str):
        self.csv_dataset_path = csv_dataset_path
        self.free_memory_for_minidataset = self.get_optimal_free_RAM()
        self.memory_of_one_row_in_bytes = self.get_approximately_row_per_memory_in_bytes()
        self.dataset_rows_amount = self.get_approximately_rows_amount_in_dataset()
        self.rows_amount_for_minidataset = self.get_approximately_rows_amount_for_minidataset()

    def get_optimal_free_RAM(self):
        virt_memory = psutil.virtual_memory()
        free_RAM_in_GB = virt_memory.available / (1024 * 1024 * 1024)
        optimal_free_RAM_for_minidataset = int(free_RAM_in_GB) - 1
        if optimal_free_RAM_for_minidataset > 0:
            print('Optimal RAM for minidataset')
            print(f'for reading, changing operations = {optimal_free_RAM_for_minidataset}GB')
            return optimal_free_RAM_for_minidataset
        else:
            print('Your free RAM very low. If you want to continue, release memory.')
            return False

    def main(self):
        if not self.dataset_copy_exist():
            self.make_copy_of_original_dataset_file()

    def dataset_copy_exist(self):
        data_dir_filenames = os.listdir('data')
        return 'copy_of_dataset_user_reviews.csv' in data_dir_filenames

    def make_copy_of_original_dataset_file(self):
        print('[INFO] Make a dataset copy for further splitting another one...')
        path_for_copy_of_dataset = 'data/copy_of_dataset_user_reviews.csv'
        os.system(f"rsync {self.csv_dataset_path} {path_for_copy_of_dataset}")

    def get_approximately_rows_amount_for_minidataset(self):
        free_memory_for_minidataset_in_bytes = self.free_memory_for_minidataset * 2 ** 30
        rows_amount_for_minidataset = int(
            free_memory_for_minidataset_in_bytes // self.memory_of_one_row_in_bytes
        )
        print(f'Rows amount for minidataset = {rows_amount_for_minidataset}')
        return rows_amount_for_minidataset

    def get_approximately_rows_amount_in_dataset(self):
        dataset_size_in_bytes = self.get_file_size_in_bytes(self.csv_dataset_path)
        approximately_rows_in_dataset = int(dataset_size_in_bytes / self.memory_of_one_row_in_bytes)
        print(f'Approximately rows amount in dataset = {approximately_rows_in_dataset}')
        return approximately_rows_in_dataset

    def get_approximately_row_per_memory_in_bytes(self):
        self.save_memory_evaluation_minidataset_to_csv()
        memory_evaluation_minidataset_size_in_bytes = self.get_file_size_in_bytes(
            'data/memory_evaluation_minidataset.csv')
        print(f'Size of evaluation dataset = {memory_evaluation_minidataset_size_in_bytes} bytes')
        approximately_memory_of_one_row_in_bytes = memory_evaluation_minidataset_size_in_bytes / 100
        print(f'Approximately memory for one row = {approximately_memory_of_one_row_in_bytes}')
        return approximately_memory_of_one_row_in_bytes

    def save_memory_evaluation_minidataset_to_csv(self):
        memory_evaluation_minidataset = pd.read_csv(self.csv_dataset_path, nrows = 100)
        memory_evaluation_minidataset.to_csv('data/memory_evaluation_minidataset.csv', index = False)

    def get_file_size_in_bytes(self, file_path: str):
        return os.path.getsize(file_path)

if __name__ == '__main__':
    dataset_splitter = Dataset_Splitting_by_free_RAM('data/user_reviews.csv')
    dataset_splitter.main()
