# unlabeled reviews texts embedding generation via LLM
import sys
sys.path.append("..")
import ollama as olm
import pandas as pd
import time

import data_preprocessing_variables as dpv

class Texts_Embedding_Dataset_Generator:
    def __init__(self, embed_model_name: str, ollama_host: str, ollama_port: str, 
    dataset_file: str, embed_dataset_path: str, computing_time_estimation_mode = False):
        self.computing_time_estimation_mode = computing_time_estimation_mode
        self.start_execution_time = 0
        self.execution_time_10_samples = 0
        self.execution_time_for_entire_dataset = 0
        
        self.embed_model_name = embed_model_name
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_client = self.get_Ollama_client()
        
        self.dataset_JSONL_file = dataset_file
        self.unlabeled_dataset = self.get_dataframe_from_JSONL_file()
        self.embed_dataset_path = embed_dataset_path
        
        if self.computing_time_estimation_mode:
            print("[WARN] Embedding-computing-time-estimation is Enabled! Testing on 10 samples...")
            self.unlabeled_embedding_dataset = self.unlabeled_dataset.copy().iloc[:10]
        else:
            self.unlabeled_embedding_dataset = self.unlabeled_dataset.copy()

    def main(self):
        self.load_embed_model_to_Ollama()
        self.start_execution_time = time.time()
        if self.computing_time_estimation_mode:
            self.add_to_dataset_embedding_column()
            self.save_text_embedding_dataset_to_JSONL()
            self.define_execution_time_for_10_samples()
            self.define_approx_execution_time_for_entire_dataset()
        else:
            self.add_to_dataset_embedding_column()
            self.save_text_embedding_dataset_to_JSONL()

    def get_dataframe_from_JSONL_file(self):
        return pd.read_json(self.dataset_JSONL_file, orient = "records", lines = True)

    def get_Ollama_client(self):
        return olm.Client(f"http://{self.ollama_host}:{self.ollama_port}")

    def load_embed_model_to_Ollama(self):
        ollama_models_names = self.get_Ollama_models_names()
        if self.embed_model_name in ollama_models_names:
            print("[INFO] Embedding model already loaded!")
        else:
            print(f"[INFO] Loading {self.embed_model_name} model for generating embedding vectors...")
            self.ollama_client.pull(self.embed_model_name)

    def get_Ollama_models_names(self):
        models_names = [model_obj.model for model_obj in dict(self.ollama_client.list())["models"]]
        return models_names

    def add_to_dataset_embedding_column(self):
        print("[INFO] Generating text embedding vectors and adding to dataset...")
        self.unlabeled_embedding_dataset["embedding_vector"] = self.unlabeled_embedding_dataset["text"].apply(
            self.get_text_embedding_vector
        )

    def get_text_embedding_vector(self, text: str):
        return self.ollama_client.embed(model = self.embed_model_name, input = text).embeddings[0]

    def save_text_embedding_dataset_to_JSONL(self):
        print("[INFO] Saving unlabeled reviews embedding dataset...")
        self.unlabeled_embedding_dataset.to_json(
            self.embed_dataset_path,
            orient = "records",
            lines = True
        )
    
    def define_execution_time_for_10_samples(self):
        end_execution_time = time.time()
        self.execution_time_10_samples = round(end_execution_time - self.start_execution_time, 2)
        print(f"[INFO] Execution time (10 samples) = {self.execution_time_10_samples} seconds")

    def define_approx_execution_time_for_entire_dataset(self):
        samples_amount = self.unlabeled_dataset.shape[0]
        self.execution_time_for_entire_dataset = round(
            (samples_amount / 10) * self.execution_time_10_samples, 2
        )
        print("[INFO] Approximated execution time for entire dataset = ", end = '') 
        print(f"{self.execution_time_for_entire_dataset} seconds or ", end = '')
        print(f"{round(self.execution_time_for_entire_dataset / 60, 1)} minutes")

if __name__ == "__main__":
    text_embed_generator = Texts_Embedding_Dataset_Generator(
        dpv.MODEL_FOR_EMBEDDING_GENERATION, dpv.TEMP_CONTAINER_OLLAMA_HOST,
        dpv.TEMP_CONTAINER_OLLAMA_PORT, dpv.CLEANED_UNLABELED_REVIEWS_DATASET,
        "test_embed_dataset_file.jsonl",
        computing_time_estimation_mode = True
    )
    text_embed_generator.main()
