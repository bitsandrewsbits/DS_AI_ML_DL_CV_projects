# Text Classification for Sentiment Analysis

import re
import pandas as pd

dataset = pd.read_csv('customer_reviews_sentiment.csv')

def sentiment_analysis():
    prepare_data()

def show_dataset():
    print(dataset)

def prepare_data():
    if dataset_has_missing_values():
        print('[INFO] Dataset has missing values!')
    else:
        print('[INFO] Dataset has not missing values.')
    text_preprocessing()

def dataset_has_missing_values():
    dataset_after_dropna = dataset.dropna()
    print('Original dataset shape:', dataset.shape)
    print('Dataset shape after dropna operation:', dataset_after_dropna.shape)
    if (dataset.shape[0] - dataset_after_dropna.shape[0]) != 0:
        return True
    else:
        return False

def text_preprocessing():
    convert_review_column_to_lowercase()
    remove_any_spec_chars_in_review_column()

def convert_review_column_to_lowercase():
    text_column = dataset['Review']
    dataset['Review'] = text_column.str.lower()

def remove_any_spec_chars_in_review_column():
    spec_chars = [',', '.', '!']
    for i in range(len(dataset['Review'])):
        for spec_char in spec_chars:
            spec_char_regex = rf"\{spec_char}"
            target_review_text = dataset['Review'][i]
            dataset.loc[i, 'Review'] = re.sub(spec_char_regex, '', target_review_text)

if __name__ == '__main__':
    sentiment_analysis()
    show_dataset()
