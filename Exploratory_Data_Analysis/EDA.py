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
        self.user_cmds = {'pairplot': self.show_resulted_df_scatterplot_matricies,
                          'heatmap': self.show_resulted_df_heatmap,
                          'histplot': self.show_histplot_by_user_column,
                          'scatterplot': self.show_scatterplot_by_user_selected_columns,
                          'boxplot': self.show_boxplot_by_user_selected_columns,
                          'barplot': self.show_barplot_by_user_selected_columns
                          }
        self.user_selected_df_column_for_histplot = ''
        self.user_selected_x_y_columns = {'x': '', 'y': ''}

    def main(self):
        self.prepare_data()
        self.create_from_df_number_and_string_types_dframes()
        self.encode_categorical_variables()
        self.create_result_df_with_encoded_columns()
        self.execute_user_cmd()

    def execute_user_cmd(self):
        user_cmd = ''
        while user_cmd != 'e':
            self.show_user_possible_cmds()
            user_cmd = input('Enter your command[press e to exit]:')
            if user_cmd in self.user_cmds:
                self.user_cmds[user_cmd]()
        print('Bye.')

    def show_user_possible_cmds(self):
        print("Possible commands:")
        print("pairplot - to show scatterplot matricies for entire result df")
        print("heatmap - to show correlaction matrix as heatmap")
        print("histplot - to show histplot by user selected df column")
        print("scatterplot - to show scatterplot by user selected two df columns")
        print("boxplot - to show boxplot by user selected two df columns")
        print("barplot - to show barplot by user selected two df columns")

    def prepare_data(self):
        print('[INFO] Preparing dataset...')
        for df_column in self.df_columns_names:
            if self.column_dtype_need_to_change(df_column):
                self.dataframe[df_column] = pd.to_numeric(
                self.dataframe[df_column], errors = 'coerce')
                self.dataframe.dropna(inplace = True)
                print('After removing missing values...')
        print(self.dataframe)

    def show_barplot_by_user_selected_columns(self):
        self.x_y_columns_selection_by_user()
        sns.barplot(data = self.dataframe,
                    x = self.user_selected_x_y_columns['x'],
                    y = self.user_selected_x_y_columns['y'])
        plt.title(f"{self.user_selected_x_y_columns['y']} \
        ({self.user_selected_x_y_columns['x']})")
        plt.show()

    def show_boxplot_by_user_selected_columns(self):
        self.x_y_columns_selection_by_user()
        sns.boxplot(data = self.result_df_with_encoded_columns,
                    x = self.user_selected_x_y_columns['x'],
                    y = self.user_selected_x_y_columns['y'])
        plt.title(f"{self.user_selected_x_y_columns['y']} \
        ({self.user_selected_x_y_columns['x']})")
        plt.xlabel(self.user_selected_x_y_columns['x'])
        plt.ylabel(self.user_selected_x_y_columns['y'])
        plt.show()

    def show_scatterplot_by_user_selected_columns(self):
        self.x_y_columns_selection_by_user()
        sns.scatterplot(data = self.result_df_with_encoded_columns,
                        x = self.user_selected_x_y_columns['x'],
                        y = self.user_selected_x_y_columns['y']
        )
        plt.title(f"Relationship between \
        {self.user_selected_x_y_columns['x']} and \
        {self.user_selected_x_y_columns['y']}")
        plt.xlabel(self.user_selected_x_y_columns['x'])
        plt.ylabel(self.user_selected_x_y_columns['y'])
        plt.show()

    def x_y_columns_selection_by_user(self):
        while True:
            print('Dataset columns:')
            print(list(self.df_columns_names))
            print('Enter two df columns names as x, y:')
            user_column_as_x = input('df column name for X-axis[e - exit from mode]: ')
            user_column_as_y = input('df column name for Y-axis[e - exit from mode]: ')
            if user_column_as_x in self.df_columns_names and user_column_as_y in self.df_columns_names:
                self.user_selected_x_y_columns['x'] = user_column_as_x
                self.user_selected_x_y_columns['y'] = user_column_as_y
                return True
            elif user_column_for_histplot == 'e':
                print(f'Exitting from plot mode...')
                return False
            else:
                print('Wrong column(s) name(s)! Try again.')

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

    def show_histplot_by_user_column(self):
        if self.column_selection_by_user_for_histplot():
            self.build_histplot_for_selected_user_column()
        else:
            print('Histplot process terminated by user.')

    def build_histplot_for_selected_user_column(self):
        data = self.result_df_with_encoded_columns[
                self.user_selected_df_column_for_histplot]
        sns.histplot(data, bins = 30, kde = True)
        plt.title(f'Distribution of {self.user_selected_df_column_for_histplot}')
        plt.xlabel(f'{self.user_selected_df_column_for_histplot}')
        plt.ylabel('Frequency')
        plt.show()

    def column_selection_by_user_for_histplot(self):
        while True:
            print('Dataset columns:')
            print(list(self.df_columns_names))
            user_column_for_histplot = input('Enter df column name for histplot[e - exit from mode]: ')
            if user_column_for_histplot in self.df_columns_names:
                self.user_selected_df_column_for_histplot = user_column_for_histplot
                return True
            elif user_column_for_histplot == 'e':
                print('Exitting from histplot mode...')
                return False
            else:
                print('Wrong column name! Try again.')

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
    eda = Exploratory_Data_Analysis("healthcare_dataset.csv")
    eda.main()
