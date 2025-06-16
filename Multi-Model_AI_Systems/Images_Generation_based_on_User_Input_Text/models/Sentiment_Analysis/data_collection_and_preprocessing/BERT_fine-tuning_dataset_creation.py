# preprocessing Films Reviews Dataset files for BERT fine-tuning dataset creation.
import additional_functions_for_data_preprocessing as ad_fncs
import re

class BERT_Fune_Tuning_Dataset_Creation:
    def __init__(self, dataset_vocabulary_file_path: str,
    positive_reviews_dir_path: str, negative_reviews_dir_path: str):
        self.dataset_vocab_file_path = dataset_vocabulary_file_path
        self.vocab_word_via_index_dict = {}

        self.positive_reviews_filenames = ad_fncs.get_filenames_from_dir(positive_reviews_dir_path)
        self.negative_reviews_filenames = ad_fncs.get_filenames_from_dir(negative_reviews_dir_path)

        self.positive_reviews_info = {'review_ID': [], 'rating': [], 'review_text': []}
        self.negative_reviews_info = {'review_ID': [], 'rating': [], 'review_text': []}

    def main(self):
        self.create_dict_for_access_vocabulary_word_via_index()

    def create_dict_for_access_vocabulary_word_via_index(self):
        with open(self.dataset_vocab_file_path, 'r') as dataset_vocab_file:
            for (i, word) in enumerate(dataset_vocab_file):
                self.vocab_word_via_index_dict[i] = word[:-1]
        return True

    def get_review_ID_and_review_rating_from_filename(self, filename: str):
        review_ID_and_rating = re.split('[_.]', filename)[:-1]
        return {'review_ID': review_ID_and_rating[0], 'rating': review_ID_and_rating[1]}

if __name__ == '__main__':
    fine_tune_dataset_creation = BERT_Fune_Tuning_Dataset_Creation(
        'data/imdb.vocab', 'data/train/pos', 'data/train/neg'
    )
    fine_tune_dataset_creation.main()
    print(fine_tune_dataset_creation.get_review_ID_and_review_rating_from_filename('1169_8.txt'))
