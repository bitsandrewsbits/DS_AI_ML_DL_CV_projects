# class for init k-means clustering algorithm and
# try to define 3 classes
import os
from sklearn.cluster import KMeans
import pandas as pd

embed_datasets_dirname = "review_datasets_with_embeddings"

class K_Means_Clustering_Manager:
    def __init__(self, datasets_dir_path: str, clusters_amount: int):
        self.datasets_dir_path = datasets_dir_path
        self.datasets_filenames = os.listdir(self.datasets_dir_path)
        self.reviews_embed_big_dataset = pd.DataFrame()
        self.clusters_amount = clusters_amount
        self.k_means = KMeans(n_clusters = self.clusters_amount)
    
    def create_one_big_embed_dataset(self):
        for small_set_file in self.datasets_filenames:
            current_set = pd.read_json(
                f"{self.datasets_dir_path}/{small_set_file}",
                orient = "records", lines = True
            )
            self.reviews_embed_big_dataset = pd.concat(
                [self.reviews_embed_big_dataset, current_set], ignore_index = True
            )
        print(self.reviews_embed_big_dataset)
    
if __name__ == "__main__":
    clusters_manager = K_Means_Clustering_Manager(embed_datasets_dirname, 3)
    clusters_manager.create_one_big_embed_dataset()