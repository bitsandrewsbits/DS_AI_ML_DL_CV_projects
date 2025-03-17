# class for init, train and test Logic Regression model for
# sentiment prediction of users films reviews

import split_dataset_by_million_rows as split_ds
import minidatasets_preparing as minids_prep
import re
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
        self.predictions_for_test_minidatasets = {}
        self.classification_reports_for_test_minidatasets = {}

    def main(self):
        # read CSV file, separate features-X and target y(prediction)
        self.define_train_and_test_datasets_files()
        # train model
        self.train_logistic_regression_model_on_current_minidataset()
        # test model on test minidatasets
        self.make_logistic_regression_predictions()
        # show classification reports(metrics) for each test minidataset
        self.show_classification_reports_by_minidatasets()

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
            print(f'[INFO] Reading (TRAIN)minidataset #{i} and defining features_X and target y...')
            self.read_and_define_features_X_target_y_as_current(minidataset_file)
            print('OK')
            print(f'[INFO] Training Model on minidataset #{i}...')
            self.logistic_regression.fit(self.current_features_X, self.current_target_y)
            print('OK')
            print('-' * 60)

    def make_logistic_regression_predictions(self):
        for (i, minidataset_file) in enumerate(self.test_datasets_files, 1):
            print(f'[INFO] Reading (TEST)minidataset #{i} and defining features_X and target y...')
            self.read_and_define_features_X_target_y_as_current(minidataset_file)
            print('OK')
            print(f'[INFO] Make a prediction on Test minidataset #{i}...')
            self.predictions_for_test_minidatasets[i] = self.logistic_regression.predict(
                self.current_features_X
            )
            self.classification_reports_for_test_minidatasets[i] = classification_report(
                self.current_target_y, self.predictions_for_test_minidatasets[i],
                zero_division = 0
            )
            print('OK')
            print('->' * 30)

    def show_classification_reports_by_minidatasets(self):
        for i in self.classification_reports_for_test_minidatasets:
            print(f'[INFO] Result Metrics for Logistic Regression on test ds #{i}:')
            print(self.classification_reports_for_test_minidatasets[i])
            print('=' * 60)

if __name__ == '__main__':
    dataset_splitter = split_ds.Dataset_Splitting('data/user_reviews.csv')
    dataset_splitter.main()
    data_preparing = minids_prep.Minidatasets_Preparing()
    data_preparing.prepare_all_minidatasets()
    logic_regres_user_sentiments = Logic_Regression_for_Sentiment_Prediction()
    logic_regres_user_sentiments.main()
