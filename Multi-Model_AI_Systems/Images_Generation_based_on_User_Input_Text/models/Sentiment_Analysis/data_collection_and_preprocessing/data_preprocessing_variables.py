# variables for data preprocessing scripts
DOWNLOADED_DATASETS_ROOT_DIR = "downloaded_datasets"
PREPROCESSED_ORIGINAL_DATASETS_DIR = "preprocessed_original_reviews_datasets"

MODEL_FOR_EMBEDDING_GENERATION = "mrutkows/granite-embedding:30m"
TEMP_CONTAINER_OLLAMA_HOST = "localhost"
TEMP_CONTAINER_OLLAMA_PORT = "11434"
CLEANED_UNLABELED_REVIEWS_DATASET = "cleaned_unlabeled_reviews_dataset.jsonl"

REVIEWS_EMBEDDING_DATASETS_DIR = "review_datasets_with_embeddings"
BIG_REVIEWS_EMBED_DATASET = "big_reviews_embedding_dataset.jsonl"

CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING = "classified_reviews_dataset_via_clastering.jsonl"
CLASSIFIED_REVIEWS_DATASET_PATH = f"{REVIEWS_EMBEDDING_DATASETS_DIR}/{CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING}"

MODEL_FOR_NEUTRAL_REVIEWS_EXTRACTION = "granite3.1-moe:1b"
NEUTRAL_REVIEWS_CONTAINER_OLLAMA_PORT = "11435"
NEUTRAL_REVIEWS_DATASET = "extracted_neutral_reviews.jsonl"