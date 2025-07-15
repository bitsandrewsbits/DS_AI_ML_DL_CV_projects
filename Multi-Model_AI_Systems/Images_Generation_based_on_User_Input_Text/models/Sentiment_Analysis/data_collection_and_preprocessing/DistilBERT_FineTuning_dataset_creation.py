# preprocessing Films Reviews Dataset files for DistilBERT fine-tuning dataset creation.
import additional_functions_for_data_preprocessing as ad_fncs
import re
import numpy as np
import pandas as pd

class DistilBERT_Fune_Tuning_Dataset_Creation:
    def __init__(self, positive_reviews_dir_path: str, negative_reviews_dir_path: str):
        self.positive_reviews_dir_path = positive_reviews_dir_path
        self.negative_reviews_dir_path = negative_reviews_dir_path
        self.positive_reviews_filenames = ad_fncs.get_filenames_from_dir(positive_reviews_dir_path)
        self.negative_reviews_filenames = ad_fncs.get_filenames_from_dir(negative_reviews_dir_path)

        self.result_reviews_dataset = {"text": [], "label": []}

    def main(self, class_samples_amount = 'all') -> pd.DataFrame:
        positive_reviews_dataset = pd.DataFrame(
            self.get_prepared_revews_info_from_filenames(
                self.positive_reviews_dir_path, self.positive_reviews_filenames
            )
        )
        negative_reviews_dataset = pd.DataFrame(
            self.get_prepared_revews_info_from_filenames(
                self.negative_reviews_dir_path, self.negative_reviews_filenames
            )
        )
        total_samples_amount = positive_reviews_dataset.shape[0] + negative_reviews_dataset.shape[0]

        if class_samples_amount != 'all' and 2 * class_samples_amount < total_samples_amount:
            positive_samples_amount = samples_amount // 2
            negative_samples_amount = samples_amount - positive_samples_amount
            result_pos_samples = positive_reviews_dataset.sample(
                positive_samples_amount, axis = 0
            )
            result_neg_samples = negative_reviews_dataset.sample(
                negative_samples_amount, axis = 0
            )
            result_dataset = pd.concat(
                [result_pos_samples, result_neg_samples],
                axis = 0, ignore_index = True
            )
        else:
            result_dataset = pd.concat(
                [positive_reviews_dataset, negative_reviews_dataset],
                axis = 0, ignore_index = True
            )
        result_dataset = result_dataset.sample(frac = 1).reset_index(drop = True)
        return result_dataset

    def get_prepared_revews_info_from_filenames(self,
    reviews_dir_path: str, reviews_filenames: list) -> dict[list]:
        reviews_info = {"text": [], "label": []}
        for review_filename in reviews_filenames:
            review_rating = int(self.get_review_rating_from_filename(review_filename))
            review_text = self.get_review_text_from_file(
                reviews_dir_path, review_filename
            )
            converted_review_rating = self.get_converted_rating_number_into_binary_class(
                review_rating
            )
            reviews_info["text"].append(review_text)
            reviews_info["label"].append(converted_review_rating)

        return reviews_info

    def get_converted_rating_number_into_binary_class(self, rating: int):
        if rating <= 4:
            return 0
        elif rating >= 7:
            return 1

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
        'data/train/pos', 'data/train/neg'
    )
    part_of_result_dataset = fine_tune_dataset_creation.main(5)
    print(part_of_result_dataset)
