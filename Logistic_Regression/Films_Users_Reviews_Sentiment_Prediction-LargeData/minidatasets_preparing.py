# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.current_dataset = pd.DataFrame()
        self.movieIDs_label_encoder = LabelEncoder()
        self.all_unique_movieIDs = []
        self.boolean_label_encoder = LabelEncoder().fit([True, False])
        self.userRealm_label_encoder = LabelEncoder()

    def prepare_all_saved_minidatasets(self):
        pass

    def prepare_one_minidataset(self, minidataset_number = 0):
        self.current_dataset = pd.read_csv(self.datasets[minidataset_number])
        self.clean_data_from_missing_values()
        self.remove_duplicated_rows()
        self.add_features_from_date_column()
        self.remove_original_date_column()
        self.add_only_new_unique_movieIDs()
        self.encode_boolean_type_columns()
        self.encode_userRealm(minidataset_number)
        self.convert_userID_to_numeric_dtype()
        print(self.current_dataset)
        print('Columns types:')
        print(self.current_dataset.dtypes)

    def clean_data_from_missing_values(self):
        self.current_dataset.dropna(axis = 1, how = 'any', inplace = True)
        self.current_dataset.dropna(axis = 0, how = 'all', inplace = True)

    def remove_duplicated_rows(self):
        self.current_dataset.drop_duplicates(inplace = True)

    def add_features_from_date_column(self):
        self.current_dataset['creationDate'] = pd.to_datetime(self.current_dataset['creationDate'])
        self.current_dataset['creationYear'] = self.current_dataset['creationDate'].dt.year
        self.current_dataset['creationMonth'] = self.current_dataset['creationDate'].dt.month
        self.current_dataset['creationDay'] = self.current_dataset['creationDate'].dt.day

    def remove_original_date_column(self):
        self.current_dataset.drop('creationDate', axis = 1, inplace = True)

    def encode_movieIDs_column(self):
        for minidataset in self.datasets:
            minidataset['movieId'] = self.movieIDs_label_encoder.transform(minidataset['movieId'])

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

    def encode_boolean_type_columns(self):
        for column in self.current_dataset.columns:
            if self.current_dataset[column].dtypes == bool:
                self.current_dataset[column] = pd.Series(
                    self.boolean_label_encoder.transform(
                        self.current_dataset[column]
                ), dtype = 'int32')

    def encode_userRealm(self, current_dataset_num: int):
        if current_dataset_num == 0:
            current_userRealm_label_encoder_classes = []
        else:
            current_userRealm_label_encoder_classes = list(self.userRealm_label_encoder.classes_)
        print('Unique values:')
        print(pd.unique(self.current_dataset['userRealm']))
        new_userRealm_label_encoder = LabelEncoder()
        new_userRealm_label_encoder.fit(self.current_dataset['userRealm'])
        for new_class in new_userRealm_label_encoder.classes_:
            if new_class not in current_userRealm_label_encoder_classes:
                current_userRealm_label_encoder_classes.append(new_class)
                print('Adding new class to userRealm label encoder...')
        self.userRealm_label_encoder.fit(current_userRealm_label_encoder_classes)
        self.current_dataset['userRealm'] = pd.Series(
            new_userRealm_label_encoder.transform(
                self.current_dataset['userRealm']
        ), dtype = 'int32')

    def convert_userID_to_numeric_dtype(self):
        self.replace_diff_userIDs_to_NaN()
        self.current_dataset.dropna(how = 'any', inplace = True)
        self.current_dataset['userId'] = pd.to_numeric(self.current_dataset['userId'])

    def replace_diff_userIDs_to_NaN(self):
        self.current_dataset['userId'] = self.current_dataset['userId'].replace(
            to_replace = '.*-.*', value = np.nan, regex = True
        )

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset()
