# preprocessing Films Reviews Dataset files for DistilBERT fine-tuning dataset creation.
import additional_functions_for_data_preprocessing as ad_fncs
import re

class DistilBERT_Fune_Tuning_Dataset_Creation:
    def __init__(self, dataset_vocabulary_file_path: str,
    positive_reviews_dir_path: str, negative_reviews_dir_path: str):
        self.dataset_vocab_file_path = dataset_vocabulary_file_path
        self.vocab_word_via_index_dict = {}

        self.positive_reviews_dir_path = positive_reviews_dir_path
        self.negative_reviews_dir_path = negative_reviews_dir_path
        self.positive_reviews_filenames = ad_fncs.get_filenames_from_dir(positive_reviews_dir_path)
        self.negative_reviews_filenames = ad_fncs.get_filenames_from_dir(negative_reviews_dir_path)

        self.positive_reviews_info = {'rating': [], 'review_text': []}
        self.negative_reviews_info = {'rating': [], 'review_text': []}

    def main(self):
        self.create_dict_for_access_vocabulary_word_via_index()

    def create_dict_for_access_vocabulary_word_via_index(self):
        with open(self.dataset_vocab_file_path, 'r') as dataset_vocab_file:
            for (i, word) in enumerate(dataset_vocab_file):
                self.vocab_word_via_index_dict[i] = word[:-1]
        return True

    def get_review_rating_from_filename(self, filename: str):
        review_ID_and_rating = re.split('[_.]', filename)[:-1]
        return review_ID_and_rating[1]

    def get_review_text_from_file(self, review_dir_path: str, filename: str):
        review_text = ''
        with open(f'{review_dir_path}/{filename}', 'r') as pos_review_file:
            for file_line in pos_review_file:
                review_text += file_line
        return review_text

if __name__ == '__main__':
    fine_tune_dataset_creation = DistilBERT_Fune_Tuning_Dataset_Creation(
        'data/imdb.vocab', 'data/train/pos', 'data/train/neg'
    )
    fine_tune_dataset_creation.main()
    print('test rating:', fine_tune_dataset_creation.get_review_rating_from_filename('1169_8.txt'))
    fine_tune_dataset_creation.get_review_text_from_file('data/train/pos', '1169_8.txt')
