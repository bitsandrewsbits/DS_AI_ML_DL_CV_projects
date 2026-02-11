# unlabeled reviews texts embedding generation via LLM - embeddinggemma
import ollama as olm
import pandas as pd

embed_model_name = "embeddinggemma"
ollama_host = "localhost"
unlabeled_reviews_dataset_file = "cleaned_unlabeled_reviews_dataset.jsonl"

def main(dataset_filename: str):
    unlabeled_dataset = get_dataframe_from_JSONL_file(dataset_filename)
    print(unlabeled_dataset)

def get_dataframe_from_JSONL_file(filename: str):
    return pd.read_json(filename, orient = "records", lines = True)

if __name__ == "__main__":
    main(unlabeled_reviews_dataset_file)
