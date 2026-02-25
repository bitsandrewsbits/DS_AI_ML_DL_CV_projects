# script for generating compose.yaml for starting multiple Ollama container with diff ports
from unlabeled_reviews_texts_embedding_generation import *
import yaml

class Compose_YAML_File_Generator:
    def __init__(self, embed_dataset_generator: object):
        self.embed_dataset_generator = embed_dataset_generator
        self.embed_dataset_generation_time_in_sec = self.get_embed_dataset_generation_time()
        self.desired_embed_generation_time_in_sec = 300
        self.ollama_instances_for_parallel_computing = self.get_ollama_instances_for_parallel_computing()
        self.first_ollama_service_port = 11434
        self.compose_config = {"services": {}}
        self.compose_yaml_strings = ""

    def main(self):
        self.define_ollama_cluster_compose_config()
        self.define_compose_yaml_strings()
        self.write_yaml_strings_to_compose_file()

    def get_embed_dataset_generation_time(self):
        self.embed_dataset_generator.main()
        return self.embed_dataset_generator.execution_time_for_entire_dataset
    
    def get_ollama_instances_for_parallel_computing(self):
        ollama_instances = round(
            self.embed_dataset_generation_time_in_sec / self.desired_embed_generation_time_in_sec
        )
        print(f"[INFO] Approximate Ollama server instances for parallel embed computing = {ollama_instances}")
        return ollama_instances
    
    def define_ollama_cluster_compose_config(self):
        print("[INFO] Generating ollama cluster compose config...")
        for i in range(self.ollama_instances_for_parallel_computing):
            current_ollama_port = self.first_ollama_service_port + i
            self.compose_config["services"][f"ollama-embed-{i}"] = {
                "image": "ollama/ollama",
                "ports": [f"{current_ollama_port}:11434"]
            }
    
    def define_compose_yaml_strings(self):
        self.compose_yaml_strings = yaml.dump(self.compose_config)
    
    def write_yaml_strings_to_compose_file(self):
        print("[INFO] Saving YAML strings to compose.yaml...")
        with open("ollama_cluster_compose.yaml", "w") as cmp_f:
            cmp_f.write(self.compose_yaml_strings)

if __name__ == "__main__":
    text_embed_generator = Texts_Embedding_Dataset_Generator(
        embed_model_name, ollama_host, ollama_port,
        unlabeled_reviews_dataset_file,
        computing_time_estimation_mode = True
    )
    compose_generator = Compose_YAML_File_Generator(text_embed_generator)
    compose_generator.main()