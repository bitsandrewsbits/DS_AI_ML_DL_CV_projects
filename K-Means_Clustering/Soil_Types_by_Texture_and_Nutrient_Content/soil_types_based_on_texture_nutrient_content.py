# Realization of K-Means Clustering classification algorithm
# Classification Soil types based on Soil texture and nutrient content(N, P, K).
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pd.read_csv(dataset_name)
        self.columns_for_target_y = ['Texture', 'Nitrogen_N_ppm', 'Phosphorus_P_ppm', 'Potassium_K_ppm']
        
        # TODO: think how to create target feature Y 
        # from 'Nitrogen_N_ppm', 'Phosphorus_P_ppm', 'Potassium_K_ppm' features
        # as a result target Y = ['Texture', 'N_P_K_ppm']
        
        self.target_y = []
        self.features_X = pd.DataFrame()
        self.clusters_amount = self.get_n_clusters_for_dataset()
        self.K_means_model = self.get_k_means_model()
        self.n_clusters_and_inertia_values = {}

    def main(self):
        self.prepare_data()
        # self.train_model()
        # self.add_labels_column_to_dataset('cluster')
        # self.set_optimal_n_clusters_for_model()
        # self.add_labels_column_to_dataset('Optimal cluster')
        # print(self.dataset.head())
        # self.show_scatterplot_of_target_Y_segmentation()

    def show_scatterplot_of_target_Y_segmentation(self):
        selected_feature = self.get_selected_feature_for_scatterplot()
        if selected_feature:
            sns.scatterplot(data = self.dataset,
                            x = selected_feature, y = self.selected_target_y,
                            hue = 'Optimal cluster')
            plt.title(f'{self.selected_target_y}({selected_feature})')
            plt.xlabel(selected_feature)
            plt.ylabel(self.selected_target_y)
            plt.show()
        else:
            print('Bye.')

    def get_features_X(self):
        features_names = self.get_features_X_columns_names()
        return self.dataset[features_names]

    def get_features_X_columns_names(self):
        df_columns.remove(self.columns_for_target_y)
        features_names = df_columns
        return features_names

    def prepare_data(self):
        print('[INFO] Preparing data...')
        self.dataset.dropna(inplace = True)
        self.dataset.drop_duplicates(inplace = True)
        
        # TODO: 1)create N_P_K_ppm feature(column), 2)delete N, P, K columns

        print(self.dataset.dtypes)
        # self.encode_categorical_features()
        # self.update_features_X_after_encoding()
        # self.scale_feature_columns()

    def scale_feature_columns(self):
        print('[INFO] Scaling feature columns...')
        scaler = StandardScaler()
        for column in self.features_X:
            reshaped_column = self.features_X[column].values.reshape(-1, 1)
            self.features_X[column] = scaler.fit_transform(reshaped_column)

    def update_features_X_after_encoding(self):
        updated_column_names = self.get_features_X_columns_names()
        self.features_X = self.dataset[updated_column_names]

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

    def add_labels_column_to_dataset(self, column_name: str):
        print('[INFO] Adding labels column to dataset...')
        self.dataset[column_name] = self.K_means_model.labels_

    def get_k_means_model(self):
        print('[INFO] Initialize K-means algorithm model...')
        return KMeans(n_clusters = self.clusters_amount, random_state = 42)

    def set_optimal_n_clusters_for_model(self):
        self.clusters_amount = self.get_optimal_n_clusters_for_model()

    def define_inertia_values_for_diff_n_clusters(self):
        min_n_clusters = 2
        median_of_unique_values_amount = self.get_median_from_unique_values_amounts()
        max_n_clusters = self.get_max_n_clusters()
        print('[INFO] Training K-Means model for different n_clusters...')
        for test_n_clusters in range(min_n_clusters, max_n_clusters + 1):
            print(f'[INFO] K-Means with {test_n_clusters} clusters.')
            self.clusters_amount = test_n_clusters
            self.K_means_model = self.get_k_means_model()
            self.train_model()
            self.n_clusters_and_inertia_values[test_n_clusters] = self.K_means_model.inertia_
        return True

    def show_plot_inertia_from_n_clusters(self):
        plt.plot(self.n_clusters_and_inertia_values.keys(),
                 self.n_clusters_and_inertia_values.values(), marker = 'o')
        plt.title('Inertia(clusters) for Elbow method')
        plt.xlabel('Clusters number')
        plt.ylabel('Inertia')
        plt.show()

    def get_n_clusters_for_dataset(self):
        print('[INFO] Defining n_clusters parameter for K-means model...')
        sorted_unique_values_by_columns = sorted(self.get_unique_values_amount_by_features_columns())
        median_number = self.get_median_from_unique_values_amounts()
        n_clusters_parameter = 0
        for column_unique_values_amount in sorted_unique_values_by_columns:
            if median_number > column_unique_values_amount * 2:
                n_clusters_parameter = column_unique_values_amount
            else:
                break
        print('n_clusters =', n_clusters_parameter)
        return n_clusters_parameter

    def get_median_from_unique_values_amounts(self):
        sorted_unique_values_by_columns = sorted(self.get_unique_values_amount_by_features_columns())
        median_number = np.median(sorted_unique_values_by_columns)
        return int(median_number)

    def get_unique_values_amount_by_features_columns(self):
        unique_values_amount_by_columns = []
        for column in self.features_X.columns:
            column_unique_values_amount = len(self.features_X[column].unique())
            unique_values_amount_by_columns.append(column_unique_values_amount)
        return unique_values_amount_by_columns

if __name__ == "__main__":
    k_means_classifier = K_Means_Clustering("data/soil_data.csv")
    k_means_classifier.main()
