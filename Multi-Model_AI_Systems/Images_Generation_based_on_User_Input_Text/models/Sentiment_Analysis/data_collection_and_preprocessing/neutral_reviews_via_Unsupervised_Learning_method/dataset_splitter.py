# dataset splitter class - split large dataset into many smaller
# small amount define by disired embed generation compute time per ollama container.
import generate_compose_multiple_ollama as gcmo
import unlabeled_reviews_texts_embedding_generation as urteg

class Dataset_Splitter:
    def __init__(self, dataset_file: str):
        self.dataset_file = dataset_file
        self.text_embed_generator = self.get_text_embed_generator()
        self.compose_generator = gcmo.Compose_YAML_File_Generator(self.text_embed_generator)
        self.target_datasets_amount = self.compose_generator.ollama_instances_for_parallel_computing
        self.datasets_filenames = []
    
    def main(self):
        pass

    def get_text_embed_generator(self):
        return urteg.Texts_Embedding_Dataset_Generator(
            urteg.embed_model_name, urteg.ollama_host, urteg.ollama_port,
            self.dataset_file,
            computing_time_estimation_mode = True
        )

if __name__ == "__main__":
        self.compose_yaml_strings = ""
    ds_splitter = Dataset_Splitter("cleaned_unlabeled_reviews_dataset.jsonl")