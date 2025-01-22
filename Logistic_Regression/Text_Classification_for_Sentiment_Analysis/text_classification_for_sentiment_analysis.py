# Text Classification for Sentiment Analysis

import re
import pandas as pd
from nltk.corpus import stopwords

class Sentiment_Analysis:
    def __init__(self, csv_file: str):
        self.dataset = pd.read_csv(csv_file)
        self.features_X = pd.DataFrame()
        self.target_y = pd.DataFrame()
        self.unique_words_in_all_reviews = set()
        self.unique_words_frequences = {}

    def main(self):
        self.prepare_data()

    def show_dataset(self):
        print(self.dataset)

    def show_features_X(self):
        start_column_name = self.features_X.columns[1]
        print(self.features_X.loc[:, f"{start_column_name}":])

    def prepare_data(self):
        if self.dataset_has_missing_values():
            print('[INFO] Dataset has missing values!')
        else:
            print('[INFO] Dataset has not missing values.')
        self.text_preprocessing()
        self.add_review_tokens_column()
        self.remove_original_review_column()
        self.split_dataset_into_features_X_target_y()
        self.features_extraction_by_BoW()
        self.show_features_X()

    def dataset_has_missing_values(self):
        dataset_after_dropna = self.dataset.dropna()
        print('Original dataset shape:', self.dataset.shape)
        print('Dataset shape after dropna operation:', dataset_after_dropna.shape)
        if (self.dataset.shape[0] - dataset_after_dropna.shape[0]) != 0:
            return True
        else:
            return False

    def text_preprocessing(self):
        self.convert_review_column_to_lowercase()
        self.remove_any_spec_chars_in_review_column()
        self.remove_stop_words_in_review_column()

    def convert_review_column_to_lowercase(self):
        text_column = self.dataset['Review']
        self.dataset['Review'] = text_column.str.lower()

    def remove_any_spec_chars_in_review_column(self):
        spec_chars = [',', '.', '!']
        for i in range(len(self.dataset['Review'])):
            for spec_char in spec_chars:
                spec_char_regex = rf"\{spec_char}"
                target_review_text = self.dataset['Review'][i]
                self.dataset.loc[i, 'Review'] = re.sub(spec_char_regex, '', target_review_text)

    def remove_stop_words_in_review_column(self):
        en_stopwords = stopwords.words(fileids = 'english')
        for i in range(len(self.dataset['Review'])):
            for stopword in en_stopwords:
                stopword_regex = rf"^{stopword} | {stopword} "
                target_review_text = self.dataset['Review'][i]
                self.dataset.loc[i, 'Review'] = re.sub(stopword_regex, ' ', target_review_text)

    def add_review_tokens_column(self):
        review_column_without_whitespace_elem = self.dataset['Review'].str.lstrip()
        self.dataset['Review_tokens'] = review_column_without_whitespace_elem.str.split(' ')

    def remove_original_review_column(self):
        self.dataset.drop(columns = 'Review', inplace = True)

    def split_dataset_into_features_X_target_y(self):
        self.define_features_X()
        self.define_target_y()

    def define_features_X(self):
        self.features_X['Review_tokens'] = self.dataset['Review_tokens']

    def define_target_y(self):
        self.target_y['Sentiment'] = self.dataset['Sentiment']

    def features_extraction_by_BoW(self):
        self.define_unique_words_in_reviews()
        self.define_and_vectorize_unique_words_presence()

    def define_unique_words_in_reviews(self):
        unique_words_by_reviews = []
        for review_words in self.features_X['Review_tokens'].values:
            unique_words_by_reviews += list(set(review_words))
        self.unique_words_in_all_reviews = set(unique_words_by_reviews)
        print('Unique words in reviews:')
        print(self.unique_words_in_all_reviews)

    def define_and_vectorize_unique_words_presence(self):
        rows_amount = self.features_X.shape[0]
        for unique_word in self.unique_words_in_all_reviews:
            self.features_X[f"{unique_word}"] = [0] * rows_amount
            for review_words in self.features_X['Review_tokens'].values:
                if unique_word in review_words:
                    word_index = review_words.index(unique_word)
                    self.features_X.loc[word_index, f"{unique_word}"] = 1

if __name__ == '__main__':
    sentiment_analysis = Sentiment_Analysis('customer_reviews_sentiment.csv')
    sentiment_analysis.main()
