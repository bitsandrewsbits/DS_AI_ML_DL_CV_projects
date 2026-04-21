# class for neutral reviews extraction via LLM:
# 1) load classified reviews embed dataset via K-means algorithm(three labels 0, 1, 2)
# 2) run Ollama container with LLM for map labels -> to sentiment classes
# 3) Sample [1-5] reviews texts for each label -> Ollama chat
# 4) Target goal - detect how good was separation between clusters(label classes must not overlapping!)
import sys
sys.path.append("..")
import os
import ollama as olm
import pandas as pd
import time

import data_preprocessing_variables as dpv
import additional_functions_for_data_preprocessing as affdp

class Neutral_Reviews_Extraction_Manager:
    def __init__(self, model_name: str, ollama_host: str, ollama_port: str, 
    classified_reviews_file_path: str, neutral_reviews_dataset_file: str):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_container_name = "neutral_reviews_extraction"
        self.delay_time_in_sec_to_ollama_ready = 5
        self.ollama_client = object
        
        self.classified_reviews_file_path = classified_reviews_file_path
        self.classified_reviews_dataset = affdp.get_dataframe_from_JSONL_file(
            self.classified_reviews_file_path
        )
        self.reviews_labels = self.get_classified_reviews_labels()
        self.samples_amount_per_label = 3
        self.samples_reviews_by_labels = {}
        self.neutral_reviews_dataset_file = neutral_reviews_dataset_file

    def main(self):
        self.run_ollama_container()
        
        self.define_Ollama_client()
        self.load_model_to_Ollama()
        self.define_reviews_samples_by_labels()

        self.stop_ollama_container()
    
    def get_classified_reviews_labels(self):
        labels_array = self.classified_reviews_dataset["sentiment_label"].unique()
        return [int(label) for label in labels_array]
    
    def run_ollama_container(self):
        os.system(f"sudo docker start {self.ollama_container_name} 2> /dev/null")
        os.system(
            f"sudo docker run --name {self.ollama_container_name} -p {self.ollama_port}:11434 -d ollama/ollama 2> /dev/null"
        )
        time.sleep(self.delay_time_in_sec_to_ollama_ready)

    def define_Ollama_client(self):
        self.ollama_client = olm.Client(host = f"http://{self.ollama_host}:{self.ollama_port}")

    def load_model_to_Ollama(self):
        ollama_models_names = self.get_Ollama_models_names()
        if self.model_name in ollama_models_names:
            print("[INFO] Model already loaded!")
        else:
            print(f"[INFO] Loading {self.model_name} model for neutral reviews extraction...")
            self.ollama_client.pull(self.model_name)

    def get_Ollama_models_names(self):
        models_names = [model_obj.model for model_obj in dict(self.ollama_client.list())["models"]]
        return models_names

    def define_reviews_samples_by_labels(self):
        for label in self.reviews_labels:
            self.samples_reviews_by_labels[label] = self.get_reviews_samples_per_label(label)
        print(self.samples_reviews_by_labels)

    def get_reviews_samples_per_label(self, label: int):
        return self.classified_reviews_dataset[
            self.classified_reviews_dataset["sentiment_label"] == label
        ]["text"].sample(self.samples_amount_per_label).values
    
    def stop_ollama_container(self):
        os.system(f"sudo docker stop {self.ollama_container_name}")

if __name__ == "__main__":
    neutral_reviews_extraction_manager = Neutral_Reviews_Extraction_Manager(
        dpv.MODEL_FOR_NEUTRAL_REVIEWS_EXTRACTION,
        dpv.TEMP_CONTAINER_OLLAMA_HOST, dpv.NEUTRAL_REVIEWS_CONTAINER_OLLAMA_PORT,
        dpv.CLASSIFIED_REVIEWS_DATASET_PATH, dpv.NEUTRAL_REVIEWS_DATASET
    )
    neutral_reviews_extraction_manager.main()