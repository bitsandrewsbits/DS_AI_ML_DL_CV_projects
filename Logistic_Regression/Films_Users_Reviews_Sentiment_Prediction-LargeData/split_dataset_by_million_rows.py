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
        self.rows_amount_for_minidataset = 10 ** 6
        self.dataset_columns_names = self.get_dataset_columns_names()

    def create_minidatasets_with_million_rows(self, minidatasets_amount = 10):
        for i in range(1, minidatasets_amount + 1):
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

    def save_minidataset_to_csv(self, minidataset_df, minidataset_number: int):
        minidataset_df.to_csv(f'data/minidataset_{minidataset_number}', index = False)

if __name__ == '__main__':
    dataset_splitter = Dataset_Splitting_by_free_RAM('data/user_reviews.csv')
    dataset_splitter.create_minidatasets_with_million_rows()
