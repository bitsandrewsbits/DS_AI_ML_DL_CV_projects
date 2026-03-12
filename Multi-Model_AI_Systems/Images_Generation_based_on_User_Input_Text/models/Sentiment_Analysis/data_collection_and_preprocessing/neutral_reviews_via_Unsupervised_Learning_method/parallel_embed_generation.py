# method:
# load small datasets -> datasets as DataFrames -> compute in parallel embed generation 
# -> concatenate to one dataset with embeddings.
import os
import pandas as pd

small_sets_dirname = "datasets_for_parallel_embed_generation"

class Parallel_Embedding_Generation_Manager:
    def __init__(self, datasets_dir: str):
        self.datasets_dir = datasets_dir
        self.datasets_files_pathes = []
        self.datasets_as_dataframes = []

    def main(self):
        if os.path.exists(self.datasets_dir):
            self.define_datasets_files_pathes()
            self.load_datasets()
        else:
            print("Dir doesn't exist!")

    def define_datasets_files_pathes(self):
        datasets_files = os.listdir(self.datasets_dir)
        for set_file in datasets_files:
            self.datasets_files_pathes.append(
                f"{self.datasets_dir}/{set_file}"
            )

    def load_datasets(self):
        print("[INFO] Loading small datasets...", end = '')
        for dataset_file_path in self.datasets_files_pathes:
            self.datasets_as_dataframes.append(
                self.get_dataframe_from_JSONL_file(dataset_file_path)
            )
        print(f"loaded {len(self.datasets_as_dataframes)} sets.")

    def get_dataframe_from_JSONL_file(self, dataset_file: str):
        return pd.read_json(dataset_file, orient = "records", lines = True)

if __name__ == "__main__":
    parallel_embed_gen_manager = Parallel_Embedding_Generation_Manager(small_sets_dirname)
    parallel_embed_gen_manager.main()