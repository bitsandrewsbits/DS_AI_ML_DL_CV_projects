# A first realization of K-Means Clustering classification algorithm
# for any datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_Means_Clustering:
    def __init__(self, dataset_name: str):
        self.dataset = pf.read_csv(dataset_name)
        
