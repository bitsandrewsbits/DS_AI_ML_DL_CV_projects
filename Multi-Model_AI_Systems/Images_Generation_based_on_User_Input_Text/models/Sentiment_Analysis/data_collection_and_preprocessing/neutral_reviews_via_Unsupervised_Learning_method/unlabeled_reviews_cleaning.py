# Text cleaning for more accurate embeddings generation
# save cleaned dataset in JSON file
import sys
sys.path.append("..")
import pandas as pd
import one_reviews_dir_dataset_creation as ordds
import download_datasets as load_ds
import re
from nltk.corpus import stopwords

unlabeled_reviews_dir_path = f"../{load_ds.downloaded_datasets_root_dir}/{load_ds.dataset['dataset_name']}/train"
unlabeled_dataset_creator = ordds.One_Reviews_Dir_Dataset_Creator(unlabeled_reviews_dir_path, 'unsup')
unlabeled_dataset = unlabeled_dataset_creator.main()
print(unlabeled_dataset)

def main(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = convert_review_column_to_lowercase(dataset)
    dataset = remove_any_spec_chars_in_review_column(dataset)
    cleaned_dataset = remove_stop_words_in_review_column(dataset)
    return cleaned_dataset

def convert_review_column_to_lowercase(dataset: pd.DataFrame):
    text_column = dataset['text']
    dataset['text'] = text_column.str.lower()
    return dataset

def remove_any_spec_chars_in_review_column(dataset: pd.DataFrame):
    spec_chars = [',', '.', '!', '?', ':', ';', '/', '@', '#']
    for i in range(len(dataset['text'])):
        for spec_char in spec_chars:
            spec_char_regex = rf"\{spec_char}"
            target_review_text = dataset['text'][i]
            dataset.loc[i, 'text'] = re.sub(spec_char_regex, '', target_review_text)
    return dataset

def remove_stop_words_in_review_column(dataset: pd.DataFrame):
    en_stopwords = stopwords.words(fileids = 'english')
    for i in range(len(dataset['text'])):
        for stopword in en_stopwords:
            stopword_regex = rf"^{stopword} | {stopword} "
            target_review_text = dataset['text'][i]
            dataset.loc[i, 'text'] = re.sub(stopword_regex, ' ', target_review_text)
    return dataset

if __name__ == "__main__":
    cleaned_dataset = main(unlabeled_dataset)
    print('cleaned dataset:')
    print(cleaned_dataset)
