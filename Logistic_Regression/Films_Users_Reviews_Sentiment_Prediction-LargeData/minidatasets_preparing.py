# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.minidatasets = datasets
        self.current_dataset = pd.DataFrame()
        self.minidatasets_one_value_columns = []
        self.boolean_label_encoder = LabelEncoder().fit([True, False])
        self.categorical_columns_from_obj_columns = []
        self.numeric_columns_from_obj_columns = []
        self.label_encoders = {}
        self.label_encoders_classes = {}
        self.quote_tag_regex = r'^:([a-zA-Z]*):'
        self.quote_words_stemmer = SnowballStemmer("english", ignore_stopwords = True)
        self.stopwords = stopwords.words('english')
        self.quotes_vocabulary = []
        self.TF_IDF_vectorizer = TfidfVectorizer()
        self.prepared_minidatasets_filenames = self.get_prepared_minidatasets_filenames()

    def get_prepared_minidatasets_filenames(self):
        return [f'data/prepared_minidataset_{i}.csv' for i in range(1, len(self.minidatasets) + 1)]

    def prepare_all_minidatasets(self):
        self.define_one_value_columns_in_all_minidatasets()
        print('[INFO] Defined one-value columns in all minidatasets:')
        print(self.minidatasets_one_value_columns)
        for i in range(len(self.minidatasets)):
            print(f'[INFO] Preparing minidataset #{i + 1}...')
            self.prepare_one_minidataset(i)
            print('OK')

        self.set_TF_IDF_vectorizer_vocabulary()
        self.add_transformed_quote_tokens_via_fitted_TF_IDF_to_minidataset()

    def prepare_one_minidataset(self, minidataset_number = 0):
        minidataset = pd.read_csv(self.minidatasets[minidataset_number])
        self.remove_one_value_columns(minidataset)
        self.current_dataset = self.get_random_samples_dataframe(minidataset)
        self.clean_data_from_missing_values()
        self.remove_duplicated_rows()
        self.add_features_from_date_column()
        self.remove_original_date_column()
        self.encode_boolean_type_columns()
        if minidataset_number == 0:
            self.define_categorical_and_numeric_from_object_columns()
            self.set_init_label_encoders_classes()

        self.convert_object_columns_to_numeric_dtype()
        self.set_label_encoders_for_categorical_columns()
        self.encode_categorical_columns(minidataset_number)
        self.remove_quote_tag_from_quote_str()
        self.convert_rating_column_into_binary_column()
        self.quote_text_preprocessing()
        self.remove_original_quote_column()
        self.update_quotes_vocabulary()
        self.save_prepared_minidataset_to_csv(minidataset_number)

    def get_random_samples_dataframe(self, minidataset: pd.DataFrame):
        return minidataset.sample(2000, axis = 0)

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

    def encode_boolean_type_columns(self):
        for column in self.current_dataset.columns:
            if self.current_dataset[column].dtypes == bool:
                self.current_dataset[column] = self.boolean_label_encoder.transform(
                    self.current_dataset[column]
                )

    def define_categorical_and_numeric_from_object_columns(self):
        for column in self.current_dataset.columns:
            if self.current_dataset[column].dtypes == object and column != 'quote':
                if self.object_column_has_numeric_dtype_in_majority(column):
                    print(f'{column} can be convert to numeric dtype.')
                    self.numeric_columns_from_obj_columns.append(column)
                else:
                    # print('Categorical column:')
                    # print(self.current_dataset[column])
                    self.categorical_columns_from_obj_columns.append(column)

    def object_column_has_numeric_dtype_in_majority(self, df_column: str):
        numeric_dtype_amount = 0
        non_numeric_dtype_amount = 0
        for value in self.current_dataset[df_column].values:
            try:
                pd.to_numeric(value)
                numeric_dtype_amount += 1
            except:
                non_numeric_dtype_amount += 1
        if numeric_dtype_amount > non_numeric_dtype_amount:
            return True
        else:
            return False

    def convert_object_columns_to_numeric_dtype(self):
        for df_column in self.numeric_columns_from_obj_columns:
            self.current_dataset[df_column] = self.current_dataset[df_column].apply(
                self.object_type_to_numeric
            )
        self.current_dataset.dropna(axis = 0, inplace = True)

    def object_type_to_numeric(self, obj_value):
        try:
            return int(pd.to_numeric(obj_value))
        except:
            return np.nan

    def set_init_label_encoders_classes(self):
        for column in self.categorical_columns_from_obj_columns:
            self.label_encoders_classes[column] = []

    def set_label_encoders_for_categorical_columns(self):
        for column in self.categorical_columns_from_obj_columns:
            self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].classes_ = self.label_encoders_classes[column]

    def update_label_encoders_classes(self, column: str, classes: list):
        self.label_encoders_classes[column] = classes

    def encode_categorical_columns(self, current_dataset_num: int):
        for column_and_label_encoder in self.label_encoders.items():
            column_for_encoding = column_and_label_encoder[0]
            label_encoder = column_and_label_encoder[1]
            if current_dataset_num == 0:
                current_label_encoder_classes = []
            else:
                current_label_encoder_classes = list(label_encoder.classes_)
            new_label_encoder = LabelEncoder()
            new_label_encoder.fit(self.current_dataset[column_for_encoding])
            for new_class in new_label_encoder.classes_:
                if new_class not in current_label_encoder_classes:
                    current_label_encoder_classes.append(new_class)
            self.update_label_encoders_classes(
                column_for_encoding, current_label_encoder_classes
            )
            print('Current label encoder classes amount:', end = ' ')
            print(len(current_label_encoder_classes))
            self.current_dataset[column_for_encoding] = label_encoder.fit_transform(
                    self.current_dataset[column_for_encoding].values
            )

    def remove_quote_tag_from_quote_str(self):
        self.current_dataset['quote'] = self.current_dataset['quote'].str.replace(
        self.quote_tag_regex, '', regex = True
        )

    def convert_rating_column_into_binary_column(self):
        self.current_dataset['rating'] = self.current_dataset['rating'].apply(
            self.rating_to_binary_format
        )

    def rating_to_binary_format(self, rating_score: float):
        if rating_score >= 5.0:
            return 1
        else:
            return 0

    def define_one_value_columns_in_all_minidatasets(self):
        for minidataset in self.minidatasets:
            minidf = pd.read_csv(minidataset)
            one_value_columns = self.get_minidataset_columns_with_only_one_value(minidf)
            self.update_minidatasets_one_value_columns(one_value_columns)

    def update_minidatasets_one_value_columns(self, new_one_value_columns: list):
        for new_column in new_one_value_columns:
            if new_column not in self.minidatasets_one_value_columns:
                self.minidatasets_one_value_columns.append(new_column)

    def remove_one_value_columns(self, df: pd.DataFrame):
        df.drop(self.minidatasets_one_value_columns, axis = 1, inplace = True)

    def get_minidataset_columns_with_only_one_value(self, minidataset):
        one_value_columns = []
        for df_column in minidataset.columns:
            unique_values_amount = len(pd.unique(minidataset[df_column]))
            if unique_values_amount == 1:
                one_value_columns.append(df_column)
        return one_value_columns

    def quote_text_preprocessing(self):
        self.add_tokenized_quote_column()
        self.quote_column_stemming()
        self.remove_stopwords_from_quotes()

    def add_tokenized_quote_column(self):
        self.current_dataset['quote_tokens'] = self.current_dataset['quote'].str.findall(
            '[a-zA-Z]+'
        )

    def quote_column_stemming(self):
        self.current_dataset['quote_tokens'] = self.current_dataset['quote_tokens'].apply(
            self.get_stemmed_quote_tokens
        )

    def get_stemmed_quote_tokens(self, tokens: list):
        stemmed_tokens = [self.quote_words_stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    def remove_stopwords_from_quotes(self):
        self.current_dataset['quote_tokens'] = self.current_dataset['quote_tokens'].apply(
            self.remove_stopwords_from_tokens
        )

    def remove_stopwords_from_tokens(self, tokens: list):
        result_tokens = [token for token in tokens if token not in self.stopwords]
        return result_tokens

    def remove_original_quote_column(self):
        self.current_dataset.drop("quote", axis = 1, inplace = True)

    def add_transformed_quote_tokens_via_fitted_TF_IDF_to_minidataset(self):
        for minidataset_file in self.prepared_minidatasets_filenames:
            print(f'Reading {minidataset_file}...')
            self.current_dataset = pd.read_csv(minidataset_file)
            print(self.current_dataset)
            transform_tokens = self.get_transformed_quote_tokens_via_fitted_TF_IDF()
            transform_tokens_df = self.get_TF_IDF_transform_result_as_DataFrame(transform_tokens)
            self.current_dataset = pd.concat(
                [self.current_dataset, transform_tokens_df],
                axis = 1
            )
            print('[INFO] Result minidataset:')
            print(self.current_dataset)
            print()

    def update_quotes_vocabulary(self):
        current_unique_quotes_words = self.current_dataset['quote_tokens'].values
        for current_unique_quote_words in current_unique_quotes_words:
            self.quotes_vocabulary = np.union1d(
                current_unique_quote_words, self.quotes_vocabulary
            )

    def set_TF_IDF_vectorizer_vocabulary(self):
        print('Total Unique quotes words amount =', len(self.quotes_vocabulary))
        self.TF_IDF_vectorizer.vocabulary = self.quotes_vocabulary

    def get_transformed_quote_tokens_via_fitted_TF_IDF(self):
        print('Current TF-IDF vocabulary len:', end = ' ')
        print(len(self.TF_IDF_vectorizer.vocabulary))
        return self.TF_IDF_vectorizer.fit_transform(
                self.current_dataset['quote_tokens']
            )

    def get_TF_IDF_transform_result_as_DataFrame(self, transform_result):
        return pd.DataFrame(transform_result.toarray(),
                columns = self.TF_IDF_vectorizer.vocabulary
        )

    def save_prepared_minidataset_to_csv(self, minidataset_number: int):
        self.current_dataset.to_csv(
            f'data/prepared_minidataset_{minidataset_number + 1}.csv',
            index = False
        )

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_all_minidatasets()
