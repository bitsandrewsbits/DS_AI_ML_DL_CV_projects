# this is program that implements an Exploratory Data Analysis of
# any dataset from CSV file with user interaction.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Exploratory_Data_Analysis:
    def __init__(self, dataset):
        self.dataframe = pd.read_csv(dataset)
        self.df_columns_names = self.dataframe.columns
        self.df_columns_with_number_types = pd.DataFrame()
        self.df_columns_with_str_types = pd.DataFrame()
        self.encoded_str_types_columns_df = pd.DataFrame()
        self.result_df_with_encoded_columns = pd.DataFrame()

    def main(self):
        self.prepare_data()
        self.create_from_df_number_and_string_types_dframes()
        self.encode_categorical_variables()
        self.create_result_df_with_encoded_columns()
        self.show_resulted_df_scatterplot_matricies()
        self.show_resulted_df_heatmap()

    def prepare_data(self):
        print('[INFO] Preparing dataset...')
        for df_column in self.df_columns_names:
            if self.column_dtype_need_to_change(df_column):
                self.dataframe[df_column] = pd.to_numeric(
                self.dataframe[df_column], errors = 'coerce')
                self.dataframe.dropna(inplace = True)
                print('After removing missing values...')
        print(self.dataframe)

    def show_resulted_df_scatterplot_matricies(self):
        target_column_names = list(self.result_df_with_encoded_columns.columns)
        for df_column in target_column_names:
            sns.pairplot(self.result_df_with_encoded_columns, hue = f"{df_column}")
            plt.show()

    def show_resulted_df_heatmap(self):
        correlation_matrix = self.result_df_with_encoded_columns.corr()
        sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    def encode_categorical_variables(self):
        print('[INFO] Encoding categorical variables...')
        for df_column in self.df_columns_with_str_types.columns:
            self.encoded_str_types_columns_df[df_column], uniques = \
            self.df_columns_with_str_types[df_column].factorize()
        print(self.encoded_str_types_columns_df)

    def create_result_df_with_encoded_columns(self):
        self.set_df_ordered_indexes_from_zero(self.df_columns_with_number_types)
        self.result_df_with_encoded_columns = pd.concat([
            self.encoded_str_types_columns_df, self.df_columns_with_number_types
        ], axis = 1)
        print('[INFO] Creating df with encoded columns...')
        print(self.result_df_with_encoded_columns)

    def set_df_ordered_indexes_from_zero(self, df):
        rows_amount = df.shape[0]
        df.index = [index for index in range(0, rows_amount)]

    def create_from_df_number_and_string_types_dframes(self):
        for df_column_name in self.df_columns_names:
            if self.get_df_column_type(df_column_name) != object:
                self.df_columns_with_number_types[df_column_name] = self.dataframe[df_column_name]
            else:
                self.df_columns_with_str_types[df_column_name] = self.dataframe[df_column_name]
        print('df only with number columns:')
        print(self.df_columns_with_number_types)
        print('\ndf only with string columns:')
        print(self.df_columns_with_str_types)
        return True

    def get_df_column_type(self, df_column: str):
        return self.dataframe[df_column].dtypes

    def column_dtype_need_to_change(self, column_name: str):
        amount_of_digit_dtype = 0
        amount_of_object_dtype = 0
        statistically_defined_column_dtype = ""
        for column_value in self.dataframe[column_name]:
            if type(column_value) == str:
                if column_value.isdigit():
                    amount_of_digit_dtype += 1
                else:
                    amount_of_object_dtype += 1
        print('Digit column value amount:', amount_of_digit_dtype)
        print('Object type column value amount:', amount_of_object_dtype)
        if amount_of_digit_dtype > amount_of_object_dtype:
            statistically_defined_column_dtype = "int64"
        else:
            statistically_defined_column_dtype = object
        print('Statistically defined column data type:', statistically_defined_column_dtype)
        if self.dataframe[column_name].dtypes != statistically_defined_column_dtype:
            return True
        else:
            return False

if __name__ == "__main__":
    eda = Exploratory_Data_Analysis("your_dataset.csv")
    eda.main()
