# method:
# load small datasets -> datasets as DataFrames -> compute in parallel embed generation 
# -> concatenate to one dataset with embeddings.
import os
import pandas as pd
import unlabeled_reviews_texts_embedding_generation as urteg

small_sets_dirname = "datasets_for_parallel_embed_generation"

class Parallel_Embedding_Generation_Manager:
    def __init__(self, datasets_dir: str):
        self.datasets_dir = datasets_dir
        self.datasets_files_pathes = []
        self.ollama_services_amount = 0
        self.ollama_host = "localhost"
        self.first_ollama_service_port = 11434
        self.embed_model = urteg.embed_model_name
        self.embed_dataset_generator_objects = []

    def main(self):
        if os.path.exists(self.datasets_dir):
            self.define_datasets_files_pathes()
            self.define_ollama_services_amount()
            self.create_embed_generator_objects()
            print(self.embed_dataset_generator_objects)
        else:
            print("Dir doesn't exist!")

    def define_datasets_files_pathes(self):
        datasets_files = os.listdir(self.datasets_dir)
        for set_file in datasets_files:
            self.datasets_files_pathes.append(
                f"{self.datasets_dir}/{set_file}"
            )
    
    def define_ollama_services_amount(self):
        self.ollama_services_amount = len(self.datasets_files_pathes)

    def create_embed_generator_objects(self):
        print("[INFO] Create embed dataset generator objects...")
        for (i, dataset_file) in enumerate(self.datasets_files_pathes):
            current_ollama_port = self.first_ollama_service_port + i
            self.embed_dataset_generator_objects.append(
                self.get_review_embed_generator_obj(
                    self.embed_model, self.ollama_host,
                    current_ollama_port, dataset_file
                )
            )
    
    def get_review_embed_generator_obj(self, embed_model, ollama_host, ollama_port, dataset_file):
        return urteg.Texts_Embedding_Dataset_Generator(
            embed_model, ollama_host, ollama_port,
            dataset_file
        )

if __name__ == "__main__":
    parallel_embed_gen_manager = Parallel_Embedding_Generation_Manager(small_sets_dirname)
    parallel_embed_gen_manager.main()