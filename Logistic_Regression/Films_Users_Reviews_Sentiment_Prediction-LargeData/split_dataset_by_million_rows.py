
import psutil
import os
import gc
import pandas as pd
from additional_functions import get_minidatasets_filenames_in_data_dir

class Dataset_Splitting:
    def __init__(self, csv_dataset_path: str, minidatasets_amount = 10):
        self.minidatasets_amount = minidatasets_amount
        self.minidatasets_amount_already_exists = 0
        self.minidatasets_amount_need_to_create = 0
        self.minidatasets_filenames = get_minidatasets_filenames_in_data_dir()
        self.csv_dataset_path = csv_dataset_path
        self.rows_amount_for_minidataset = 10 ** 6
        self.dataset_columns_names = self.get_dataset_columns_names()

    def main(self):
        self.define_minidatasets_amount_that_need_to_create()
        if self.minidatasets_amount_need_to_create > 0:
            self.create_minidatasets_with_million_rows()

    def define_minidatasets_amount_that_need_to_create(self):
        if self.minidatasets_amount > len(self.minidatasets_filenames):
            self.minidatasets_amount_already_exists = len(self.minidatasets_filenames)
            self.minidatasets_amount_need_to_create = self.minidatasets_amount - \
            self.minidatasets_amount_already_exists
            print(f'[INFO] Already {self.minidatasets_amount_already_exists}', end = '')
            print(' minidatasets exist.')
            print(f'[INFO] Need to create {self.minidatasets_amount_need_to_create}')
        else:
            print('[INFO] Requested amount of minidatasets already exist - OK')

    def create_minidatasets_with_million_rows(self):
        for i in range(self.minidatasets_amount_already_exists + 1,
        self.minidatasets_amount + 1):
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
        minidataset_df.to_csv(f'data/minidataset_{minidataset_number}.csv', index = False)

if __name__ == '__main__':
    dataset_splitter = Dataset_Splitting('data/user_reviews.csv')
    dataset_splitter.main()
