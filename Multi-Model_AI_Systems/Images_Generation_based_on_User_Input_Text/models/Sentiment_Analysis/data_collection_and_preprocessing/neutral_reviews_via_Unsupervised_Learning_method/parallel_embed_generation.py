# method:
# load small datasets -> datasets as DataFrames -> compute in parallel embed generation 
# -> concatenate to one dataset with embeddings.
import os
import pandas as pd
import unlabeled_reviews_texts_embedding_generation as urteg
from concurrent.futures import ProcessPoolExecutor

small_sets_dirname = "datasets_for_parallel_embed_generation"
embed_datasets_dirname = "review_datasets_with_embeddings"

class Parallel_Embedding_Generation_Manager:
    def __init__(self, datasets_dir: str, embed_dataset_dir: str):
        self.datasets_dir = datasets_dir
        self.embed_datasets_dir = embed_dataset_dir
        self.datasets_files_pathes = []
        self.ollama_services_amount = 0
        self.ollama_host = "localhost"
        self.first_ollama_service_port = 11434
        self.embed_model = urteg.embed_model_name
        self.embed_dataset_generators_params = []
        self.embed_reviews_datasets = []

    def main(self):
        if os.path.exists(self.embed_datasets_dir):
            print("[INFO] Dir for datasets with embeddings exists!")
        else:
            print("[INFO] Creating dir for datasets with embeddings...")
            os.mkdir(self.embed_datasets_dir)
            
        if os.path.exists(self.datasets_dir):
            self.define_datasets_files_pathes()
            self.define_ollama_services_amount()
            self.create_ollama_services_params()
            self.generate_embed_review_datasets_on_parallel_ollama()
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

    def start_embed_dataset_generation(self, embed_generator_params: dict):
        embed_model = embed_generator_params["embed_model"]
        ollama_host = embed_generator_params["ollama_host"]
        ollama_port = embed_generator_params["ollama_port"]
        dataset_path = embed_generator_params["dataset_path"]
        embed_dataset_path = embed_generator_params["embed_dataset_path"]
        print(f"[INFO] Start embed dataset generation on Ollama->port:{ollama_port}...")
        embed_generator = urteg.Texts_Embedding_Dataset_Generator(
            embed_model, ollama_host, ollama_port,
            dataset_path, embed_dataset_path
        )
        try:
            embed_generator.main()
        except:
            print(f'[WARN] Something wrong with ollama:{ollama_port}!')

    def create_ollama_services_params(self) -> list[dict]:
        for (i, dataset_path) in enumerate(self.datasets_files_pathes):
            current_ollama_port = self.first_ollama_service_port + i
            current_ollama_service = {}
            current_ollama_service["embed_model"] = self.embed_model
            current_ollama_service["ollama_host"] = self.ollama_host
            current_ollama_service["ollama_port"] = current_ollama_port
            current_ollama_service["dataset_path"] = dataset_path
            current_ollama_service["embed_dataset_path"] = self.get_embed_dataset_path(
                dataset_path
            )
            self.embed_dataset_generators_params.append(
                current_ollama_service
            )
    
    def get_embed_dataset_path(self, dataset_path: str):
        dataset_name = dataset_path.split('/')[1]
        return f"{self.embed_dataset_dir}/embed_{dataset_name}"
    
    def generate_embed_review_datasets_on_parallel_ollama(self):
        with ProcessPoolExecutor() as executor:
            executor.map(self.start_embed_dataset_generation, self.embed_dataset_generators_params)
        print("DONE!")

if __name__ == "__main__":
    parallel_embed_gen_manager = Parallel_Embedding_Generation_Manager(
        small_sets_dirname, embed_datasets_dirname
    )
    parallel_embed_gen_manager.main()