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
        self.scale_numeric_dataset_columns()
        self.add_year_columns()
        print(self.dataset)

    def scale_numeric_dataset_columns(self):
        scaler = StandardScaler()
        for column in self.dataset.columns:
            if self.dataset[column].dtypes != object:
                reshaped_df_column = self.dataset[column].values.reshape(-1, 1)
                self.dataset[column] = scaler.fit_transform(reshaped_df_column)

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

    def get_new_column_name(self, column_name: str, period_of_date: str):
        if 'Date' in column_name:
            target_name_parts = column_name.split(' ')
            index_of_date_word = target_name_parts.index('Date')
            target_name_parts.remove('Date')
            target_name_parts.insert(index_of_date_word, period_of_date)
            new_column_name = ' '.join(target_name_parts)
            return new_column_name

    def train_model(self):
        pass

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
