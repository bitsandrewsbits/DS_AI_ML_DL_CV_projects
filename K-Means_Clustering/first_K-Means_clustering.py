# A first realization of K-Means Clustering classification algorithm
# for any datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pf.read_csv(dataset_name)
        self.clusters_amount = 0

    def main(self):
        self.prepare_data()

    def prepare_data(self):
        print('[INFO] Cleaning data from missing values...')
        self.dataset.dropna(inplace = True)
        # TODO: finish method

if __name__ == "__main__":
    k_means_classifier = K_Means_Clustering("your_dataset.csv")
