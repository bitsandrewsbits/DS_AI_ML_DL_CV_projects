# this is program that implements an Exploratory Data Analysis of
# any dataset from CSV file.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

class Exploratory_Data_Analysis:
    def __init__(self, dataset):
        self.dataframe = pd.read_csv(dataset)
        self.df_columns_names = []

    def main(self):
        self.prepare_data()
        self.define_df_columns_names()

    def prepare_data(self):
        print('[INFO] Preparing dataset...')
        self.dataframe.dropna(inplace = True)

    def define_df_columns_names(self):
        print('[INFO] Defining dataset columns names...')
        self.df_columns_names = list(self.dataframe.columns)
        print(self.df_columns_names)

if __name__ == "__main__":
    eda = Exploratory_Data_Analysis("movie_ratings.csv")
    eda.main()
