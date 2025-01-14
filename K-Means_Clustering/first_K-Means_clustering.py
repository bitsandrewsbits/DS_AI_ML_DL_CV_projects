# A first realization of K-Means Clustering classification algorithm
# for any datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pd.read_csv(dataset_name)
        self.selected_target_y = self.get_selected_target_y_column_name()
        self.features_X = self.get_features_X()
        self.date_columns_names = self.get_date_columns_names()
        self.clusters_amount = self.get_n_clusters_for_dataset()
        self.K_means_model = self.get_k_means_model()

    def main(self):
        self.prepare_data()
        self.train_model()
        self.add_cluster_labels_column_to_dataset()
        print(self.dataset)

    def get_selected_target_y_column_name(self):
        user_input = ''
        while user_input != 'e':
            print('Dataset columns:')
            print(self.dataset.columns)
            user_input = input('Enter target y column for prediction[e - for exit]: ')
            if user_input in self.dataset.columns:
                return user_input
            elif user_input == 'e':
                print('Exitting from y target selection mode...')
            else:
                print('Wrong column name!')

    def get_features_X(self):
        print('[INFO] Defining features X for model training...')
        features_names = self.get_features_X_columns_names()
        return self.dataset[features_names]

    def get_features_X_columns_names(self):
        df_columns = list(self.dataset.columns)
        df_columns.remove(self.selected_target_y)
        features_names = df_columns
        return features_names

    def prepare_data(self):
        print('[INFO] Preparing data...')
        self.dataset.dropna(inplace = True)
        self.dataset.drop_duplicates(inplace = True)
        self.add_year_columns()
        self.add_month_columns()
        self.add_day_columns()
        self.delete_original_date_columns()
        self.encode_categorical_features()
        self.update_features_X_after_encoding()
        self.scale_feature_columns()

    def scale_feature_columns(self):
        print('[INFO] Scaling feature columns...')
        scaler = StandardScaler()
        for column in self.features_X:
            reshaped_column = self.features_X[column].values.reshape(-1, 1)
            self.features_X[column] = scaler.fit_transform(reshaped_column)

    def update_features_X_after_encoding(self):
        updated_column_names = self.get_features_X_columns_names()
        self.features_X = self.dataset[updated_column_names]

    def get_date_columns_names(self):
        date_columns_names = []
        for column in self.dataset.columns:
            try:
                test_date_str = str(self.dataset[column].values[0])
                pd.Timestamp(test_date_str)
                date_columns_names.append(column)
            except:
                continue
        return date_columns_names

    def add_year_columns(self):
        target_period_of_date = 'Year'
        for column_name in self.date_columns_names:
            new_column_name = self.get_new_column_name(
                column_name, target_period_of_date)
            self.dataset[new_column_name] = pd.to_datetime(
                self.dataset[column_name]
            ).dt.year

    def add_month_columns(self):
        target_period_of_date = 'Month'
        for column_name in self.date_columns_names:
            new_column_name = self.get_new_column_name(
                column_name, target_period_of_date)
            self.dataset[new_column_name] = pd.to_datetime(
                self.dataset[column_name]
            ).dt.month

    def add_day_columns(self):
        target_period_of_date = 'Day'
        for column_name in self.date_columns_names:
            new_column_name = self.get_new_column_name(
                column_name, target_period_of_date)
            self.dataset[new_column_name] = pd.to_datetime(
                self.dataset[column_name]
            ).dt.day

    def get_new_column_name(self, column_name: str, period_of_date: str):
        if 'Date' in column_name:
            target_name_parts = column_name.split(' ')
            index_of_date_word = target_name_parts.index('Date')
            target_name_parts.remove('Date')
            target_name_parts.insert(index_of_date_word, period_of_date)
            new_column_name = ' '.join(target_name_parts)
            return new_column_name

    def delete_original_date_columns(self):
        print('[INFO] Deleting original date columns...')
        self.dataset.drop(self.date_columns_names,
                        axis = 'columns', inplace = True)

    def encode_categorical_features(self):
        print('[INFO] Encoding categorical features...')
        label_encoder = LabelEncoder()
        for column in self.dataset.columns:
            if self.dataset[column].dtypes == object:
                label_encoder.fit(self.dataset[column].values)
                self.dataset[column] = label_encoder.transform(
                    self.dataset[column].values)

    def train_model(self):
        print('[INFO] Training K-means model...')
        self.K_means_model.fit(self.dataset)

    def add_cluster_labels_column_to_dataset(self):
        print('[INFO] Adding labels column to dataset...')
        self.dataset['cluster'] = self.K_means_model.labels_

    def get_k_means_model(self):
        print('[INFO] Initialize K-means algorithm model...')
        return KMeans(n_clusters = self.clusters_amount, random_state = 42)

    def get_n_clusters_for_dataset(self):
        print('[INFO] Defining n_clusters parameter for K-means model...')
        sorted_unique_values_by_columns = sorted(self.get_unique_values_amount_by_columns())
        median_number = np.median(sorted_unique_values_by_columns)
        n_clusters_parameter = 0
        for column_unique_values_amount in sorted_unique_values_by_columns:
            if median_number > column_unique_values_amount * 2:
                n_clusters_parameter = column_unique_values_amount
            else:
                break
        print('n_clusters =', n_clusters_parameter)
        return n_clusters_parameter

    def get_unique_values_amount_by_columns(self):
        print('[INFO] Defining amount of unique values by columns...')
        unique_values_amount_by_columns = []
        for column in self.dataset.columns:
            column_unique_values_amount = len(self.dataset[column].unique())
            unique_values_amount_by_columns.append(column_unique_values_amount)
        return unique_values_amount_by_columns

if __name__ == "__main__":
    k_means_classifier = K_Means_Clustering("your_dataset.csv")
    k_means_classifier.main()
