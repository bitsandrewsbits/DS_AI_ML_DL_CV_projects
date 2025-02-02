# Sentiment Analysis with created 10 Films Reviews minidatasets

import pandas as pd

class Minidatasets_Preparing:
    def __init__(self, datasets: list):
        self.datasets = datasets

    def prepare_all_saved_minidatasets(self):
        pass

    def prepare_one_minidataset(self, minidataset_number = 0):
        mini_df = pd.read_csv(self.datasets[minidataset_number])
        print(mini_df.columns)

if __name__ == "__main__":
    minidatasets = [f'data/minidataset_{i}' for i in range(1, 11)]
    data_preparing = Minidatasets_Preparing(minidatasets)
    data_preparing.prepare_one_minidataset(2)
