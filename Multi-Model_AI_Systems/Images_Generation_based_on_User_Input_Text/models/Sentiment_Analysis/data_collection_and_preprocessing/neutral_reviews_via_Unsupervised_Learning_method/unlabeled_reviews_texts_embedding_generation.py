# unlabeled reviews texts embedding generation via LLM - embeddinggemma
import ollama as olm
import pandas as pd

embed_model_name = "embeddinggemma:latest"
ollama_host = "localhost"
ollama_port = "11434"
unlabeled_reviews_dataset_file = "cleaned_unlabeled_reviews_dataset.jsonl"

class Texts_Embedding_Dataset_Generator:
    def __init__(self, embed_model_name: str, ollama_host: str, ollama_port: str, dataset_file: str):
        self.embed_model_name = embed_model_name
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_client = self.get_Ollama_client()
        self.dataset_JSONL_file = dataset_file

    def main(self):
        unlabeled_dataset = self.get_dataframe_from_JSONL_file()
        print(unlabeled_dataset)
        self.load_embed_model_to_Ollama()
        print("test embed generation:")
        print(self.get_text_embedding_vector("How are you"))
        # TODO: think, how to generate embeddings for entire dataset(text column)
        # and what data format it will be.

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
            olm.pull(self.embed_model_name)

    def get_Ollama_models_names(self):
        models_names = [model_obj.model for model_obj in dict(olm.list())["models"]]
        return models_names

    def get_text_embedding_vector(self, text: str):
        return olm.embed(model = self.embed_model_name, input = text).embeddings[0]

if __name__ == "__main__":
    text_embed_generator = Texts_Embedding_Dataset_Generator(
        embed_model_name, ollama_host, ollama_port, unlabeled_reviews_dataset_file
    )
    text_embed_generator.main()
