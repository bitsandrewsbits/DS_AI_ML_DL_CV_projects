# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.current_dataset = pd.DataFrame()
        self.boolean_label_encoder = LabelEncoder().fit([True, False])
        self.label_encoders = {'movieId': LabelEncoder(),
                               'userRealm': LabelEncoder()}
        self.quote_tag_regex = r'^:([a-zA-Z]*):'
        self.quote_words_stemmer = SnowballStemmer("english", ignore_stopwords = True)

    def prepare_all_saved_minidatasets(self):
        pass

    def prepare_one_minidataset(self, minidataset_number = 0):
        self.current_dataset = pd.read_csv(self.datasets[minidataset_number])
        self.clean_data_from_missing_values()
        self.remove_duplicated_rows()
        self.add_features_from_date_column()
        self.remove_original_date_column()
        self.encode_boolean_type_columns()
        self.encode_categorical_columns(minidataset_number)
        self.convert_userID_to_numeric_dtype()
        self.remove_quote_tag_from_quote_str()
        self.quote_text_preprocessing()
        print(self.current_dataset[['quote', 'quote_tokens']])

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

    def encode_categorical_columns(self, current_dataset_num: int):
        for column_and_label_encoder in self.label_encoders.items():
            column_for_encoding = column_and_label_encoder[0]
            label_encoder = column_and_label_encoder[1]
            if current_dataset_num == 0:
                current_label_encoder_classes = []
            else:
                current_label_encoder_classes = list(label_encoder.classes_)
            print('Unique values:')
            print(pd.unique(self.current_dataset[column_for_encoding]))
            new_label_encoder = LabelEncoder()
            new_label_encoder.fit(self.current_dataset[column_for_encoding])
            for new_class in new_label_encoder.classes_:
                if new_class not in current_label_encoder_classes:
                    current_label_encoder_classes.append(new_class)
            label_encoder.fit(current_label_encoder_classes)
            self.current_dataset[column_for_encoding] = pd.Series(
                new_label_encoder.transform(
                    self.current_dataset[column_for_encoding]
            ), dtype = 'int32')

    def convert_userID_to_numeric_dtype(self):
        self.replace_diff_userIDs_to_NaN()
        self.current_dataset.dropna(how = 'any', inplace = True)
        self.current_dataset['userId'] = pd.to_numeric(self.current_dataset['userId'])

    def replace_diff_userIDs_to_NaN(self):
        self.current_dataset['userId'] = self.current_dataset['userId'].replace(
            to_replace = '.*-.*', value = np.nan, regex = True
        )

    def remove_quote_tag_from_quote_str(self):
        self.current_dataset['quote'] = self.current_dataset['quote'].str.replace(
        self.quote_tag_regex, '', regex = True
        )

    def quote_text_preprocessing(self):
        self.add_tokenized_quote_column()
        self.quote_column_stemming()
        # TODO: finish

    def add_tokenized_quote_column(self):
        self.current_dataset['quote_tokens'] = self.current_dataset['quote'].str.findall(
            '[a-zA-Z]+'
        )

    def quote_column_stemming(self):
        self.current_dataset['quote_tokens'] = self.current_dataset['quote_tokens'].apply(
            self.get_stemmed_quote_tokens
        )

    def get_stemmed_quote_tokens(self, tokens: list):
        return [self.quote_words_stemmer.stem(token) for token in tokens]

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset()
