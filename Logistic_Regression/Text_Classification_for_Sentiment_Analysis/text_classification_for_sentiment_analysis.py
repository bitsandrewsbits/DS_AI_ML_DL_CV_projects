# Text Classification for Sentiment Analysis

import pandas as pd

dataset = pd.read_csv('customer_reviews_sentiment.csv')

def sentiment_analysis():
    pass

def show_dataset():
    print(dataset.head())

def prepare_data():
    pass

def dataset_has_missing_values():
    if dataset.all():
        print('Dataset has missing values!')
        return True
    else:
        return False

if __name__ == '__main__':
    show_dataset()
    dataset_has_missing_values()
