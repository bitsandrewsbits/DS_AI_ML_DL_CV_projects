# this is program for analyzing your free RAM and
# compute optimal size of minidatasets for reading, changing operations.
# Minidatasets will be writting into disk into dir where original large dataset exists.

import psutil
import os
import gc
import pandas as pd

class Dataset_Splitting_by_free_RAM:
    def __init__(self, csv_dataset_path: str):
        self.csv_dataset_path = csv_dataset_path
        self.free_memory_for_minidataset = self.get_optimal_free_RAM()
        self.rows_amount_for_minidataset = 10 ** 6
        self.dataset_columns_names = self.get_dataset_columns_names()

    def create_first_10_minidatasets_with_million_rows(self):
        for i in range(1, 11):
            try:
                print(f'[INFO] Making Minidataset #{i}...\n')
                minidataset = pd.read_csv(self.csv_dataset_path,
                    nrows = self.rows_amount_for_minidataset,
                    skiprows = self.rows_amount_for_minidataset * (i - 1),
                    header = 0,
                    names = self.dataset_columns_names)
                print(minidataset)
                self.save_minidataset_to_csv(minidataset, i)
                del minidataset
                gc.collect()
                print('DONE')
            except:
                print('[INFO] End of CSV dataset. exitting...')
                break

    def get_dataset_columns_names(self):
        dataset_head = pd.read_csv(self.csv_dataset_path, nrows = 1)
        return dataset_head.columns

    def get_optimal_free_RAM(self):
        virt_memory = psutil.virtual_memory()
        total_RAM_in_GB = int(virt_memory.total / (2 ** 30))
        free_RAM_in_GB = int(virt_memory.available / (2 ** 30))
        print(f'Free RAM = {free_RAM_in_GB}GB')
        if free_RAM_in_GB >= 1 and total_RAM_in_GB > 4:
            optimal_free_RAM_for_minidataset = free_RAM_in_GB / total_RAM_in_GB
            print('Optimal RAM for minidataset:')
            print(f'for reading, changing operations = {optimal_free_RAM_for_minidataset}GB')
            return optimal_free_RAM_for_minidataset
        else:
            print('Your free RAM very low. If you want to continue, release memory.')
            return False

    def split_dataset_into_minidatasets(self):
        minidataset_number = 1
        iteration_counter = 0
        while True:
            try:
                print(f'[INFO] Making Minidataset #{minidataset_number}...')
                minidataset = pd.read_csv(self.csv_dataset_path,
                    nrows = self.rows_amount_for_minidataset,
                    skiprows = self.rows_amount_for_minidataset * iteration_counter)
                self.save_minidataset_to_csv(minidataset, minidataset_number)
                minidataset_number += 1
                iteration_counter += 1
                del minidataset
                gc.collect()
                print('DONE')
            except:
                print('[INFO] End of CSV dataset. exitting...')
                break

    def save_minidataset_to_csv(self, minidataset_df, minidataset_number: int):
        minidataset_df.to_csv(f'data/minidataset_{minidataset_number}', index = False)

    def define_rows_amount_for_minidataset_optimal_RAM(self):
        free_RAM_in_bytes = self.free_memory_for_minidataset * (2 ** 30)
        print('RAM for minidataset in bytes:', free_RAM_in_bytes)
        step_rows_amount = 10 ** 4
        self.save_memory_evaluation_minidataset_to_csv()
        while True:
            test_eval_dataset_size = self.get_file_size_in_bytes(
                'data/memory_evaluation_minidataset.csv'
            )
            print(f'Test dataset size = {test_eval_dataset_size} bytes')
            self.rows_amount_for_minidataset += step_rows_amount
            if test_eval_dataset_size < free_RAM_in_bytes:
                self.save_memory_evaluation_minidataset_to_csv(self.rows_amount_for_minidataset)
            else:
                print('[INFO] Found rows amount of minidataset for your free RAM:')
                print(f'nrows = {self.rows_amount_for_minidataset}')
                return True

    def save_memory_evaluation_minidataset_to_csv(self, rows_n = 1):
        memory_evaluation_minidataset = pd.read_csv(self.csv_dataset_path, nrows = rows_n)
        memory_evaluation_minidataset.to_csv('data/memory_evaluation_minidataset.csv', index = False)
        del memory_evaluation_minidataset
        gc.collect()

    def get_file_size_in_bytes(self, file_path: str):
        return os.path.getsize(file_path)

if __name__ == '__main__':
    dataset_splitter = Dataset_Splitting_by_free_RAM('data/user_reviews.csv')
    dataset_splitter.create_first_10_minidatasets_with_million_rows()
