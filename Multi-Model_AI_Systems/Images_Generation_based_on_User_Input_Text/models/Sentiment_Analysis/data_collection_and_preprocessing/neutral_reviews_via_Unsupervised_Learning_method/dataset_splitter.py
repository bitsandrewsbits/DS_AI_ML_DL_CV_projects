# dataset splitter class - split large dataset into many smaller
# small amount define by disired embed generation compute time per ollama container.
import generate_compose_multiple_ollama as gcmo
import unlabeled_reviews_texts_embedding_generation as urteg
import data_preprocessing_variables as dpv
import pandas as pd
import os

class Dataset_Splitter:
    def __init__(self, dataset_JSONL_file: str):
        self.dataset_file = dataset_JSONL_file
        self.dataset = self.get_dataframe_from_JSONL_file()
        self.small_datasets_filenames = []
        self.small_datasets_amount = 0
        self.datasets_dir = "datasets_for_parallel_embed_generation"
        self.rows_amount_in_small_dataset = 0
    
    def main(self):
        if os.path.exists(self.datasets_dir):
            print("[INFO] Datasets dir for parallel embed generation exists!")
        else:
            print("[INFO] Creating datasets dir for parallel embed generation...")
            os.mkdir(self.datasets_dir)
        self.define_small_datasets_names()
        self.define_small_dataset_rows_amount()
        self.split_dataset_into_small_sets()
    
    def split_dataset_into_small_sets(self):
        dataset_start_index = 0
        dataset_end_index = self.rows_amount_in_small_dataset
        for (i, dataset_name) in enumerate(self.small_datasets_filenames, 1):
            current_dataset = self.dataset.iloc[dataset_start_index:dataset_end_index]
            self.save_dataset_in_JSONL(current_dataset, f"{self.datasets_dir}/{dataset_name}")
            dataset_start_index = dataset_end_index
            dataset_end_index = self.rows_amount_in_small_dataset * i
    
    def save_dataset_in_JSONL(self, dataset: pd.DataFrame, path: str):
        dataset.to_json(path, orient = "records", lines = True)
    
    def define_small_dataset_rows_amount(self):
        self.rows_amount_in_small_dataset = self.dataset.shape[0] // self.small_datasets_amount

    def define_small_datasets_names(self):
        self.small_datasets_filenames = [
            f"dataset_part_{ds_number}.jsonl" for ds_number in range(1, self.small_datasets_amount + 1)
        ]
    
    def get_dataframe_from_JSONL_file(self):
        return pd.read_json(self.dataset_file, orient = "records", lines = True)
    
    def define_small_datasets_amount(self):
        self.text_embed_generator = self.get_text_embed_generator()
        self.compose_generator = gcmo.Compose_YAML_File_Generator(self.text_embed_generator)
        self.small_datasets_amount = self.compose_generator.ollama_instances_for_parallel_computing

    def get_text_embed_generator(self):
        return urteg.Texts_Embedding_Dataset_Generator(
            dpv.MODEL_FOR_EMBEDDING_GENERATION,
            dpv.TEMP_CONTAINER_OLLAMA_HOST,
            dpv.TEMP_CONTAINER_OLLAMA_PORT,
            self.dataset_file,
            dpv.EMBED_COMPUTE_TIME_ESTIMATION_SET_NAME,
            computing_time_estimation_mode = True
        )

if __name__ == "__main__":
    ds_splitter = Dataset_Splitter(dpv.CLEANED_UNLABELED_REVIEWS_DATASET)
    ds_splitter.define_small_datasets_amount()
    ds_splitter.main()