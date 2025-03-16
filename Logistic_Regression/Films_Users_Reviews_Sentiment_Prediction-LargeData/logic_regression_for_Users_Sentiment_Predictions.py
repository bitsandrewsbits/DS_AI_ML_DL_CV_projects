# class for init, train and test Logic Regression model for
# sentiment prediction of users films reviews

import re
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Logic_Regression_for_Sentiment_Prediction:
    def __init__(self):
        self.data_dir_files = os.listdir('data')
        self.train_datasets_files = []
        self.test_datasets_files = []
        self.current_dataset = pd.DataFrame()
        self.current_features_X = pd.DataFrame()
        self.target_y_column = 'rating'
        self.current_target_y = pd.Series()
        self.logistic_regression = LogisticRegression()

    def main(self):
        # TODO-1: read CSV file, separate features-X and target y(prediction)
        self.define_train_and_test_datasets_files()
        # TODO-2: train model
        self.train_logistic_regression_model_on_current_minidataset()
        # TODO-3: test model on test minidatasets

    def define_train_and_test_datasets_files(self):
        for data_dir_file in self.data_dir_files:
            if re.match(r'result_minidataset_[1-9][0-9]?_for_train.csv', data_dir_file):
                self.train_datasets_files.append(data_dir_file)
            elif re.match(r'result_minidataset_[1-9][0-9]?_for_test.csv', data_dir_file):
                self.test_datasets_files.append(data_dir_file)

    def read_minidataset_as_current(self, dataset_file_path: str):
        self.current_dataset = pd.read_csv(f'data/{dataset_file_path}')

    def define_current_features_X(self):
        self.current_features_X = self.current_dataset.drop(self.target_y_column, axis = 1)

    def define_current_target_y(self):
        self.current_target_y = self.current_dataset[self.target_y_column]

    def read_and_define_features_X_target_y_as_current(self,
    minidataset_file_path: str):
        self.read_minidataset_as_current(minidataset_file_path)
        self.define_current_features_X()
        self.define_current_target_y()

    def train_logistic_regression_model_on_current_minidataset(self):
        for (i, minidataset_file) in enumerate(self.train_datasets_files, 1):
            print(f'[INFO] Reading minidataset #{i} and defining features_X and target y...')
            self.read_and_define_features_X_target_y_as_current(minidataset_file)
            print('OK')
            print(f'[INFO] Training Model on minidataset #{i}...', end = ' ')
            self.logistic_regression.fit(self.current_features_X, self.current_target_y)
            print('OK')
            print('-' * 60)

    def test_logistic_regression_model(self, test_minidataset_path: str):
        for minidataset_file in self.test_datasets_files:
            self.read_and_define_features_X_target_y_as_current(minidataset_file)
            self.logistic_regression.predict()

if __name__ == '__main__':
    logic_regres_user_sentiments = Logic_Regression_for_Sentiment_Prediction()
    logic_regres_user_sentiments.main()
