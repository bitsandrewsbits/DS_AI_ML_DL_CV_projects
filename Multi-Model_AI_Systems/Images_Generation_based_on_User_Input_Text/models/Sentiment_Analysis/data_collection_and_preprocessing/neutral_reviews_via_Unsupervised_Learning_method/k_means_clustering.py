# class for init k-means clustering algorithm and
# try to define 3 classes
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

embed_datasets_dirname = "review_datasets_with_embeddings"

class K_Means_Clustering_Manager:
    def __init__(self, datasets_dir_path: str, clusters_amount: int):
        self.datasets_dir_path = datasets_dir_path
        self.big_embed_dataset_filename = "big_reviews_embedding_dataset.jsonl"
        self.datasets_filenames = os.listdir(self.datasets_dir_path)
        self.reviews_embed_big_dataset = pd.DataFrame()
        self.clusters_amount = clusters_amount
        self.k_means = KMeans(n_clusters = self.clusters_amount)
        self.embeddings_ndarray = []
        self.clusters_labels = []

    def main(self):
        if self.big_embed_dataset_filename in os.listdir(self.datasets_dir_path):
            print("[INFO] Merged reviews embed dataset file already exists!")
            print("[INFO] Loading dataset to dataframe...")
            self.reviews_embed_big_dataset = self.get_dataset_from_file(
                f"{self.datasets_dir_path}/{self.big_embed_dataset_filename}"
            )
        else:
            print("[INFO] Creating merged one reviews embedding dataset...")
            self.create_one_big_embed_dataset()
            self.save_big_embed_dataset_to_JSONL()
        
        print("[INFO] Classify reviews via K-means...")
        reviews_embeddings = self.get_embeddings_ndarray(
            self.reviews_embed_big_dataset["embedding_vector"].values
        )
        print(reviews_embeddings)
        self.clusters_labels = self.k_means.fit_predict(reviews_embeddings)
        print(self.clusters_labels)
        self.reviews_embed_big_dataset["sentiment_label"] = self.clusters_labels
        print(self.reviews_embed_big_dataset[["text", "sentiment_label"]])

    def create_one_big_embed_dataset(self):
        for small_set_file in self.datasets_filenames:
            current_set = self.get_dataset_from_file(
                f"{self.datasets_dir_path}/{small_set_file}"
            )
            self.reviews_embed_big_dataset = pd.concat(
                [self.reviews_embed_big_dataset, current_set], ignore_index = True
            )
    
    def get_embeddings_ndarray(self, embed_column_values):
        embeds_ndarray = []
        for embed_vector_list in embed_column_values:
            embed_vector_arr = np.array(embed_vector_list)
            embeds_ndarray.append(embed_vector_arr)
        return np.array(embeds_ndarray)
    
    def save_big_embed_dataset_to_JSONL(self):
        print("[INFO] Saving big reviews embed dataset to JSONL...")
        self.reviews_embed_big_dataset.to_json(
            f"{self.datasets_dir_path}/{self.big_embed_dataset_filename}",
            orient = 'records',
            lines = True
        )
    
    def get_dataset_from_file(self, jsonl_path):
        return pd.read_json(
            jsonl_path, orient = 'records', lines = True
        )

if __name__ == "__main__":
    clusters_manager = K_Means_Clustering_Manager(embed_datasets_dirname, 3)
    clusters_manager.main()