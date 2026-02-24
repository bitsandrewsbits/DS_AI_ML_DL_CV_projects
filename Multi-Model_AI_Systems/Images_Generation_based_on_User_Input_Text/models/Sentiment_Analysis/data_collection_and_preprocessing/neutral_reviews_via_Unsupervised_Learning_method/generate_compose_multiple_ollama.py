# script for generating compose.yaml for starting multiple Ollama container with diff ports
from unlabeled_reviews_texts_embedding_generation import *

class Compose_YAML_File_Generator:
    def __init__(self, embed_dataset_generator: object):
        self.embed_dataset_generator = embed_dataset_generator
        self.embed_dataset_generation_time_in_sec = self.get_embed_dataset_generation_time()
        self.desired_embed_generation_time_in_sec = 300
        self.ollama_instances_for_parallel_computing = self.get_ollama_instances_for_parallel_computing()

    def get_embed_dataset_generation_time(self):
        self.embed_dataset_generator.main()
        return self.embed_dataset_generator.execution_time_for_entire_dataset
    
    def get_ollama_instances_for_parallel_computing(self):
        ollama_instances = round(
            self.embed_dataset_generation_time_in_sec / self.desired_embed_generation_time_in_sec
        )
        print(f"[INFO] Approximate Ollama server instances for parallel embed computing = {ollama_instances}")
        return ollama_instances

if __name__ == "__main__":
    text_embed_generator = Texts_Embedding_Dataset_Generator(
        embed_model_name, ollama_host, ollama_port,
        unlabeled_reviews_dataset_file,
        computing_time_estimation_mode = True
    )
    compose_generator = Compose_YAML_File_Generator(text_embed_generator)