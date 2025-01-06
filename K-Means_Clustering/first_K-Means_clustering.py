# A first realization of K-Means Clustering classification algorithm
# for any datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pd.read_csv(dataset_name)
        self.clusters_amount = 0

    def main(self):
        self.prepare_data()

    def prepare_data(self):
        print('[INFO] Cleaning data from missing values...')
        self.dataset.dropna(inplace = True)
        self.scale_numeric_dataset_columns()

    def scale_numeric_dataset_columns(self):
        scaler = StandardScaler()
        # check if column - is number type column and transform through
        # standard scaler
        for column in self.dataset.columns:
            if self.dataset[column].dtypes != object:
                reshaped_df_column = self.dataset[column].values.reshape(-1, 1)
                self.dataset[column] = scaler.fit_transform(reshaped_df_column)
        print(self.dataset)

if __name__ == "__main__":
    k_means_classifier = K_Means_Clustering("your_dataset.csv")
    k_means_classifier.main()
