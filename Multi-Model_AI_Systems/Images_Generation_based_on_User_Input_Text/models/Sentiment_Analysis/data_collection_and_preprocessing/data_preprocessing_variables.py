# variables for data preprocessing scripts
DOWNLOADED_DATASETS_ROOT_DIR = "downloaded_datasets"
PREPROCESSED_ORIGINAL_DATASETS_DIR = "preprocessed_original_reviews_datasets"
USERS_SENTIMENTS_DATASET_INFO = {
    "dataset_name": "Large_Movie_Reviews_Dataset",
    "URL": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
}

MODEL_FOR_EMBEDDING_GENERATION = "mrutkows/granite-embedding:30m"
TEMP_CONTAINER_OLLAMA_HOST = "localhost"
TEMP_CONTAINER_OLLAMA_PORT = "11434"
EMBED_COMPUTE_TIME_ESTIMATION_SET_NAME = "set_for_embed_compute_time_estimation.jsonl"
CLEANED_UNLABELED_REVIEWS_DATASET = "cleaned_unlabeled_reviews_dataset.jsonl"

SMALL_SETS_FOR_EMBED_GENERATION_DIR = "datasets_for_parallel_embed_generation"
REVIEWS_EMBEDDING_DATASETS_DIR = "review_datasets_with_embeddings"
BIG_REVIEWS_EMBED_DATASET = "big_reviews_embedding_dataset.jsonl"

CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING = "classified_reviews_dataset_via_clastering.jsonl"
CLASSIFIED_REVIEWS_DATASET_PATH = f"{REVIEWS_EMBEDDING_DATASETS_DIR}/{CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING}"

MODELS_FOR_NEUTRAL_REVIEWS_EXTRACTION = ["pilardi/sentiment-analysis:llama3", "ministral-3:3b", "granite3.3:8b"]
NEUTRAL_REVIEWS_CONTAINER_OLLAMA_PORT = "11435"
NEUTRAL_REVIEWS_DATASET = "extracted_neutral_reviews.jsonl"