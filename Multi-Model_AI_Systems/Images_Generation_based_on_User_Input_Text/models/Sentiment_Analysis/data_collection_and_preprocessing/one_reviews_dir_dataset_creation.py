# preprocessing Films Reviews Dataset files for DistilBERT fine-tuning dataset creation.
import re
import numpy as np
import pandas as pd

import additional_functions_for_data_preprocessing as ad_fncs
import download_datasets as load_ds
import data_preprocessing_variables as dpv

class One_Reviews_Dir_Dataset_Creator:
    def __init__(self, root_reviews_dir_path: str, target_reviews_dirname: str):
        self.root_reviews_dir_path = root_reviews_dir_path
        self.target_reviews_dirname = target_reviews_dirname
        self.reviews_dir_path = f"{self.root_reviews_dir_path}/{self.target_reviews_dirname}"
        self.reviews_filenames = ad_fncs.get_filenames_from_dir(self.reviews_dir_path)

        self.result_reviews_dataset = {"text": [], "label": []}

    def main(self, samples_amount = 'all') -> pd.DataFrame:
        reviews_dataset = pd.DataFrame(
            self.get_prepared_reviews_info_from_filenames(
                self.reviews_dir_path, self.reviews_filenames
            )
        )
        total_samples_amount = reviews_dataset.shape[0]

        if samples_amount != 'all':
            result_dataset = reviews_dataset.sample(
                samples_amount, axis = 0
            )
        else:
            result_dataset = reviews_dataset

        result_dataset = result_dataset.sample(frac = 1).reset_index(drop = True)
        return result_dataset

    def get_prepared_reviews_info_from_filenames(self,
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
    datasets_pathes = {
        "train": f"{dpv.DOWNLOADED_DATASETS_ROOT_DIR}/{load_ds.dataset['dataset_name']}/train",
        "test": f"{dpv.DOWNLOADED_DATASETS_ROOT_DIR}/{load_ds.dataset['dataset_name']}/test"
    }
    ad_fncs.create_directory(dpv.PREPROCESSED_ORIGINAL_DATASETS_DIR)
    for dataset_type in datasets_pathes:
        ad_fncs.create_directory(
            f"{dpv.PREPROCESSED_ORIGINAL_DATASETS_DIR}/{dataset_type}"
        )
        for sentiment_type in ['pos', 'neg']:
            dataset_creator = One_Reviews_Dir_Dataset_Creator(
                f"{datasets_pathes[dataset_type]}", sentiment_type
            )
            original_dataset = dataset_creator.main()
            ad_fncs.save_dataset_into_JSONL(
                original_dataset,
                f"{dpv.PREPROCESSED_ORIGINAL_DATASETS_DIR}/{dataset_type}/{sentiment_type}.jsonl"
            )