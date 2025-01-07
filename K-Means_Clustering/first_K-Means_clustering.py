# A first realization of K-Means Clustering classification algorithm
# for any datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pd.read_csv(dataset_name)
        self.clusters_amount = self.get_n_clusters_for_dataset()

    def main(self):
        self.prepare_data()

    def prepare_data(self):
        print('[INFO] Cleaning data from missing values...')
        self.dataset.dropna(inplace = True)
        self.scale_numeric_dataset_columns()

    def scale_numeric_dataset_columns(self):
        scaler = StandardScaler()
        for column in self.dataset.columns:
            if self.dataset[column].dtypes != object:
                reshaped_df_column = self.dataset[column].values.reshape(-1, 1)
                self.dataset[column] = scaler.fit_transform(reshaped_df_column)
        print(self.dataset)

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
