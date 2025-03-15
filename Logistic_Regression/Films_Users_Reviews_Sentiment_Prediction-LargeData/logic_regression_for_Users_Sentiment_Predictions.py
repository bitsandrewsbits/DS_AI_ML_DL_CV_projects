# class for init, train and test Logic Regression model for
# sentiment prediction of users films reviews

import pandas as pd
import re
import os

class Logic_Regression_for_Sentiment_Prediction:
    def __init__(self):
        self.data_dir_files = os.listdir('data')
        self.train_datasets_files = []
        self.test_datasets_files = []
        self.current_dataset = pd.DataFrame()
        self.current_features_X = pd.DataFrame()
        self.target_y_column = 'rating'
        self.current_target_y = pd.Series()

    def main(self):
        # TODO-1: read CSV file, separate features-X and target y(prediction)
        self.define_train_and_test_datasets_files()
        # TODO-2: train model
        # TODO-3: test model on test minidatasets

    def define_train_and_test_datasets_files(self):
        for data_dir_file in self.data_dir_files:
            if re.match(r'result_minidataset_[1-9][0-9]?_for_train.csv', data_dir_file):
                self.train_datasets_files.append(data_dir_file)
            elif re.match(r'result_minidataset_[1-9][0-9]?_for_test.csv', data_dir_file):
                self.test_datasets_files.append(data_dir_file)

if __name__ == '__main__':
    logic_regres_user_sentiments = Logic_Regression_for_Sentiment_Prediction()
    logic_regres_user_sentiments.main()
