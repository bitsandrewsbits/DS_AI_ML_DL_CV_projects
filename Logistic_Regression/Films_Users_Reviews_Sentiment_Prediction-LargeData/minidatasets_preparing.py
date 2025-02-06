# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.current_dataset = pd.DataFrame()
        self.movieIDs_label_encoder = LabelEncoder()
        self.all_unique_movieIDs = []

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
        self.add_only_new_unique_movieIDs()

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

    def encode_movieIDs_column(self):
        for minidataset in self.datasets:
            self.movieIDs_label_encoder.transform(minidataset)
            # TODO: finish

    def fit_all_classes_to_movieIDs_label_encoder(self):
        self.movieIDs_label_encoder.fit(self.all_unique_movieIDs)

    def add_only_new_unique_movieIDs(self):
        current_unique_movieIDs = self.get_unique_movieIDs()
        for unique_movieID in current_unique_movieIDs:
            if unique_movieID not in self.all_unique_movieIDs:
                self.all_unique_movieIDs.append(unique_movieID)
        print('Total unique movieIDs =', len(self.all_unique_movieIDs))

    def get_unique_movieIDs(self):
        return pd.unique(self.current_dataset['movieId'])

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset()
