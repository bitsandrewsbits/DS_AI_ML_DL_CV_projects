# this is program that implements an Exploratory Data Analysis of
# any dataset from CSV file with user interaction.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

class Exploratory_Data_Analysis:
    def __init__(self, dataset):
        self.dataframe = pd.read_csv(dataset)
        self.df_columns_names = []
        self.df_columns_with_number_types = pd.DataFrame()
        self.df_columns_with_string_types = pd.DataFrame()

    def main(self):
        self.prepare_data()
        self.define_df_columns_names()
        self.split_df_into_number_and_string_types_dframes()

    def prepare_data(self):
        print('[INFO] Preparing dataset...')
        self.dataframe.dropna(inplace = True)

    def define_df_columns_names(self):
        print('[INFO] Defining dataset columns names...')
        self.df_columns_names = list(self.dataframe.columns)
        print(self.df_columns_names)

    def split_df_into_number_and_string_types_dframes(self):
        for df_column_name in self.df_columns_names:
            if self.get_df_column_type(df_column_name) != object:
                self.df_columns_with_number_types[df_column_name] = self.dataframe[df_column_name]
            else:
                self.df_columns_with_string_types[df_column_name] = self.dataframe[df_column_name]
        print('df only with number columns')
        print(self.df_columns_with_number_types.head())
        print('\ndf only with string columns')
        print(self.df_columns_with_string_types.head())
        return True

    def get_df_column_type(self, df_column: str):
        return self.dataframe[df_column].dtypes

if __name__ == "__main__":
    eda = Exploratory_Data_Analysis("Your_dataset.csv")
    eda.main()
