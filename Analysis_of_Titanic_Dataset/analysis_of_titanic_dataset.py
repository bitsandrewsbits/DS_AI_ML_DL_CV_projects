import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

class Dataset_Analyzer:
	def __init__(self, dataset_url: str):
		self.CSV_dataset_URL = dataset_url
		self.CSV_dataset = self.get_dataset_in_CSV_from_URL(self.CSV_dataset_URL)
		self.dataset_column_names = self.get_dataset_columns_names()
		self.dataset_columns_types = dict(self.CSV_dataset.dtypes)
		self.dataset_columns_with_missing_values = []
		self.missing_values_columns_with_object_dtype = []
		self.missing_values_columns_with_int_float_dtypes = []
		
		self.target_y_columns = ['Survived']
		self.features_set_X = pd.DataFrame()
		self.target_set_y = pd.DataFrame()

		self.train_set_X = pd.DataFrame()
		self.train_set_y = pd.DataFrame()
		self.test_set_X = pd.DataFrame()
		self.test_set_y = pd.DataFrame()

	def show_dataset(self):
		print(self.CSV_dataset)

	def get_dataset_columns_names(self):
		return self.CSV_dataset.columns

	def show_dataset_columns_types(self):
		print(self.dataset_columns_types)

	def show_dataset_columns(self):
		print(self.dataset_column_names)

	def get_dataset_in_CSV_from_URL(self, url: str):
		return pd.read_csv(url)

	def define_columns_with_missing_values_from_dataset(self):
		missing_values_by_bool_mapping_of_dataset = self.CSV_dataset.isna()

		for dataset_column in self.dataset_column_names:
			column_values = missing_values_by_bool_mapping_of_dataset[dataset_column].values
			if True in column_values:
				self.dataset_columns_with_missing_values.append(dataset_column)
				print(f'Missing values column - {dataset_column}!')
		return True

	def define_object_dtype_missing_value_columns(self):
		for missing_value_column in self.dataset_columns_with_missing_values:
			if self.dataset_columns_types[missing_value_column] == object:
				self.missing_values_columns_with_object_dtype.append(missing_value_column)
		return True

	def define_int_float_dtype_missing_value_columns(self):
		for missing_value_column in self.dataset_columns_with_missing_values:
			if self.dataset_columns_types[missing_value_column] != object:
				self.missing_values_columns_with_int_float_dtypes.append(missing_value_column)
		return True

	def remove_rows_with_object_dtype_columns_NaN_values(self):
		for object_dtype_column in self.missing_values_columns_with_object_dtype:
			self.CSV_dataset = self.CSV_dataset[self.CSV_dataset[object_dtype_column].notna()]

	def replace_columns_NaN_values_with_mean_values(self):
		for int_float_column in self.missing_values_columns_with_int_float_dtypes:
			column_mean_value = self.CSV_dataset[int_float_column].mean(skipna = True)
			self.CSV_dataset[int_float_column] = self.CSV_dataset[int_float_column].replace(
				to_replace = np.nan, value = column_mean_value
			)
		return True

	def define_features_set_X(self):
		for target_y_column in self.target_y_columns:
			self.features_set_X = self.CSV_dataset.drop(target_y_column, axis = 1)
		return True

	def define_target_set_y(self):
		self.target_set_y = self.CSV_dataset[self.target_y_columns]
		return True

	def split_dataset_into_train_test_parts(self):
		self.train_set_X, self.test_set_X, \
		self.train_set_y, self.test_set_y = train_test_split(
			self.features_set_X, self.target_set_y, test_size = 0.2, random_state = 42
			)
		return True

	def features_scaling(self):
		pass


if __name__ == '__main__':
	dataset_analyzer = Dataset_Analyzer("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

	dataset_analyzer.show_dataset()
	dataset_analyzer.show_dataset_columns()
	dataset_analyzer.define_columns_with_missing_values_from_dataset()
	dataset_analyzer.show_dataset_columns_types()
	dataset_analyzer.define_object_dtype_missing_value_columns()
	dataset_analyzer.define_int_float_dtype_missing_value_columns()
	dataset_analyzer.remove_rows_with_object_dtype_columns_NaN_values()
	dataset_analyzer.replace_columns_NaN_values_with_mean_values()
	print('After data cleaning and replacing:')
	dataset_analyzer.show_dataset()

	dataset_analyzer.define_features_set_X()
	dataset_analyzer.define_target_set_y()
	
	print('Splitting dataset for train and test stages.')
	dataset_analyzer.split_dataset_into_train_test_parts()
