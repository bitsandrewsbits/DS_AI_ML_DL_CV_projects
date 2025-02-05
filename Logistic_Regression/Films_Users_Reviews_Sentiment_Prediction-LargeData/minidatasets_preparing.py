# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.current_dataset = pd.DataFrame()

    def prepare_all_saved_minidatasets(self):
        pass

    def prepare_one_minidataset(self, minidataset_number = 0):
        self.current_dataset = pd.read_csv(self.datasets[minidataset_number])
        self.clean_data_from_missing_values()
        self.add_features_from_date_column()
        self.remove_original_date_column()
        print(self.current_dataset)
        print('Columns types:')
        print(self.current_dataset.dtypes)

    def clean_data_from_missing_values(self):
        self.current_dataset.dropna(axis = 1, how = 'any', inplace = True)
        self.current_dataset.dropna(axis = 0, how = 'all', inplace = True)

    def add_features_from_date_column(self):
        self.current_dataset['creationDate'] = pd.to_datetime(self.current_dataset['creationDate'])
        self.current_dataset['creationYear'] = self.current_dataset['creationDate'].dt.year
        self.current_dataset['creationMonth'] = self.current_dataset['creationDate'].dt.month
        self.current_dataset['creationDay'] = self.current_dataset['creationDate'].dt.day

    def remove_original_date_column(self):
        self.current_dataset.drop('creationDate', axis = 1, inplace = True)

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset()
