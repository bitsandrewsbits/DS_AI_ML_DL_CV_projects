# class for init k-means clustering algorithm and
# try to define 3 classes
import sys
sys.path.append("..")
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

import data_preprocessing_variables as dpv
import additional_functions_for_data_preprocessing as adfdp

class K_Means_Clustering_Manager:
    def __init__(self, datasets_dir_path: str, big_embed_dataset: str, classified_reviews_big_dataset: str, clusters_amount: int):
        self.datasets_dir_path = datasets_dir_path
        self.big_embed_dataset_filename = big_embed_dataset
        self.datasets_filenames = os.listdir(self.datasets_dir_path)
        self.reviews_embed_big_dataset = pd.DataFrame()
        self.clusters_amount = clusters_amount
        self.k_means = KMeans(n_clusters = self.clusters_amount)
        self.pca_decompositor = object
        self.embeddings_ndarray = []
        self.clusters_labels = []
        self.classified_reviews_big_dataset = pd.DataFrame()
        self.classified_reviews_big_dataset_file = classified_reviews_big_dataset

    def main(self):
        if self.big_embed_dataset_filename in self.datasets_filenames:
            print("[INFO] Merged reviews embed dataset file already exists!")
            print("[INFO] Loading dataset to dataframe...")
            self.reviews_embed_big_dataset = self.get_dataset_from_file(
                f"{self.datasets_dir_path}/{self.big_embed_dataset_filename}"
            )
        else:
            print("[INFO] Creating merged one reviews embedding dataset...")
            self.create_one_big_embed_dataset()
            print("[INFO] Saving big reviews embed dataset to JSONL...")
            adfdp.save_dataset_into_JSONL(
                self.reviews_embed_big_dataset,
                f"{self.datasets_dir_path}/{self.big_embed_dataset_filename}"
            )
        self.define_sentiment_classes_labels()
        self.create_classified_reviews_big_dataset()
        print("[INFO] Saving classified reviews dataset to JSONL...")
        adfdp.save_dataset_into_JSONL(
            self.classified_reviews_big_dataset,
            f"{self.datasets_dir_path}/{self.classified_reviews_big_dataset_file}"
        )
        
    def create_classified_reviews_big_dataset(self):
        self.reviews_embed_big_dataset["sentiment_label"] = self.clusters_labels
        self.classified_reviews_big_dataset = self.reviews_embed_big_dataset[["text", "sentiment_label"]]

    def define_sentiment_classes_labels(self):
        print("[INFO] Classify reviews via K-means...")
        reviews_embeddings = self.get_embeddings_ndarray(
            self.reviews_embed_big_dataset["embedding_vector"].values
        )
        self.clusters_labels = self.k_means.fit_predict(reviews_embeddings)
        print(self.clusters_labels)

    def create_one_big_embed_dataset(self):
        for small_set_file in self.datasets_filenames:
            if small_set_file == self.classified_reviews_big_dataset_file:
                continue
            current_set = self.get_dataset_from_file(
                f"{self.datasets_dir_path}/{small_set_file}"
            )
            normalized_reduced_embeds = self.get_reduced_embed_set_via_PCA(
                current_set["embedding_vector"]
            )
            current_set["embedding_vector"] = self.get_series_from_ndarray(normalized_reduced_embeds)
            self.reviews_embed_big_dataset = pd.concat(
                [self.reviews_embed_big_dataset, current_set], ignore_index = True
            )
    
    def get_reduced_embed_set_via_PCA(self, embed_column: pd.Series):
        reviews_embeddings = self.get_embeddings_ndarray(
            embed_column.values
        )
        embedding_size = reviews_embeddings[0].shape[0]
        self.pca_decompositor = PCA(n_components = int(embedding_size * 0.6))
        reduced_reviews_embeddings = self.pca_decompositor.fit_transform(reviews_embeddings)
        normalized_reduced_embeddings = normalize(reduced_reviews_embeddings, norm = "l2")
        return normalized_reduced_embeddings
    
    def get_embeddings_ndarray(self, embed_column_values):
        embeds_ndarray = []
        for embed_vector_list in embed_column_values:
            embed_vector_arr = np.array(embed_vector_list)
            embeds_ndarray.append(embed_vector_arr)
        return np.array(embeds_ndarray)
    
    def get_series_from_ndarray(self, embed_ndarrays):
        embed_df = {"embedding_vector": list(embed_ndarrays)}
        return pd.DataFrame(embed_df)["embedding_vector"]
    
    def get_dataset_from_file(self, jsonl_path):
        return pd.read_json(
            jsonl_path, orient = 'records', lines = True
        )

if __name__ == "__main__":
    clusters_manager = K_Means_Clustering_Manager(
        dpv.REVIEWS_EMBEDDING_DATASETS_DIR, dpv.BIG_REVIEWS_EMBED_DATASET,
        dpv.CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING, 3
    )
    clusters_manager.main()