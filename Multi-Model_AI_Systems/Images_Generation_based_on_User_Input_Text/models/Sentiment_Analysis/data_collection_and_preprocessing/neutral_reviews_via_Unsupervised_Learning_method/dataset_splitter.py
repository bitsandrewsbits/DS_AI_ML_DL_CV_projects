# dataset splitter class - split large dataset into many smaller
# small amount define by disired embed generation compute time per ollama container.
import generate_compose_multiple_ollama as gcmo
import unlabeled_reviews_texts_embedding_generation as urteg
import pandas as pd
import os

class Dataset_Splitter:
    def __init__(self, dataset_JSONL_file: str):
        self.dataset_file = dataset_JSONL_file
        self.dataset = self.get_dataframe_from_JSONL_file()
        self.text_embed_generator = self.get_text_embed_generator()
        self.compose_generator = gcmo.Compose_YAML_File_Generator(self.text_embed_generator)
        self.target_datasets_amount = self.compose_generator.ollama_instances_for_parallel_computing
        self.datasets_filenames = self.get_datasets_names()
        self.datasets_dir = "datasets_for_parallel_embed_generation"
    
    def main(self):
        if os.path.exists(self.datasets_dir):
            print("[INFO] Datasets dir for parallel embed generation exists!")
        else:
            print("[INFO] Creating datasets dir for parallel embed generation...")
            os.mkdir(self.datasets_dir)
    
    def split_dataset_into_small_sets(self):
        # TODO: create method
        pass

    def get_datasets_names(self):
        return [
            f"dataset_#{ds_number}" for ds_number in range(1, self.target_datasets_amount + 1)
        ]
    
    def get_dataframe_from_JSONL_file(self):
        return pd.read_json(self.dataset_file, orient = "records", lines = True)

    def get_text_embed_generator(self):
        return urteg.Texts_Embedding_Dataset_Generator(
            urteg.embed_model_name, urteg.ollama_host, urteg.ollama_port,
            self.dataset_file,
            computing_time_estimation_mode = True
        )

if __name__ == "__main__":
    ds_splitter = Dataset_Splitter("cleaned_unlabeled_reviews_dataset.jsonl")
    ds_splitter.main()