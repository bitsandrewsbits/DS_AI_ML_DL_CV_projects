# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets

    def prepare_all_saved_minidatasets(self):
        pass

    def prepare_one_minidataset(self, minidataset_number = 0):
        mini_df = pd.read_csv(self.datasets[minidataset_number])
        dataset_columns_names = mini_df.columns
        self.clean_data_from_missing_values(mini_df)
        print('Columns types:')
        print(mini_df.dtypes)

    def clean_data_from_missing_values(self, df):
        df.dropna(axis = 1, how = 'any', inplace = True)
        df.dropna(axis = 0, how = 'all', inplace = True)

    def add_features_from_date_column(self):
        pass

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset()
