# Text cleaning for more accurate embeddings generation
# save cleaned dataset in JSON file
import sys
sys.path.append("..")
import pandas as pd
import one_reviews_dir_dataset_creation as ordds
import download_datasets as load_ds
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

unlabeled_reviews_dir_path = f"../{load_ds.downloaded_datasets_root_dir}/{load_ds.dataset['dataset_name']}/train"
unlabeled_dataset_creator = ordds.One_Reviews_Dir_Dataset_Creator(unlabeled_reviews_dir_path, 'unsup')
unlabeled_dataset = unlabeled_dataset_creator.main()

def main(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = convert_review_column_to_lowercase(dataset)
    dataset = remove_any_spec_chars_in_review_column(dataset)
    cleaned_dataset = remove_stop_words_in_review_column(dataset)
    return cleaned_dataset

def convert_review_column_to_lowercase(dataset: pd.DataFrame):
    print("[INFO] Converting reviews text to lowercase...")
    text_column = dataset['text']
    dataset['text'] = text_column.str.lower()
    return dataset

def remove_any_spec_chars_in_review_column(dataset: pd.DataFrame):
    print("[INFO] Removing special characters from reviews...")
    spec_chars = [
        ',', '.', '!', '?', ':', ';', '/', '@', '#', '\'', '(', ')', '<', '>',
        '{', '}', '*', '\\', '~', '&', '-', '$', '=', '|'
    ]
    reviews_column_for_updating = dataset['text']
    for spec_char in spec_chars:
        spec_char_regex = rf"\{spec_char}"
        reviews_column_for_updating = reviews_column_for_updating.apply(
            lambda review_text: re.sub(spec_char_regex, '', review_text)
        )
    dataset['text'] = reviews_column_for_updating
    return dataset

def remove_stop_words_in_review_column(dataset: pd.DataFrame):
    print("[INFO] Removing stop words from reviews...")
    en_stopwords = stopwords.words(fileids = 'english')
    reviews_column_for_updating = dataset['text']
    for stopword in en_stopwords:
        stopword_regex = rf"^{stopword} | {stopword} "
        reviews_column_for_updating = reviews_column_for_updating.apply(
            lambda review_text: re.sub(stopword_regex, ' ', review_text)
        )
    dataset['text'] = reviews_column_for_updating
    return dataset

def save_cleaned_dataset_to_JSONL(dataset: pd.DataFrame):
    dataset.to_json(
        "cleaned_unlabeled_reviews_dataset.jsonl",
        orient = 'records',
        lines = True
    )

if __name__ == "__main__":
    cleaned_dataset = main(unlabeled_dataset)
    print("[INFO] Saving cleaned reviews dataset into file...")
    save_cleaned_dataset_to_JSONL(cleaned_dataset)
