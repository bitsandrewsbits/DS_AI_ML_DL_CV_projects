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
		self.dataset_columns_types = pd.Series()
		self.dataset_columns_with_missing_values = []
		self.features_set_X = pd.DataFrame()

	def show_dataset(self):
		print(self.CSV_dataset)

	def get_dataset_columns_names(self):
		return self.CSV_dataset.columns

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
				print(f'Missing values in {dataset_column}.')

	def remove_target_y_column_from_dataset(self):
		pass

if __name__ == '__main__':
	dataset_analyzer = Dataset_Analyzer("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

	dataset_analyzer.show_dataset()
	dataset_analyzer.show_dataset_columns()
	dataset_analyzer.define_columns_with_missing_values_from_dataset()