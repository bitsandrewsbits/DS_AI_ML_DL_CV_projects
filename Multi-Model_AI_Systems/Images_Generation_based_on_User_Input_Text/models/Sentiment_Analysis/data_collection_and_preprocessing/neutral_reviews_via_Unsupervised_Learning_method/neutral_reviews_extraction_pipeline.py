# Neutral reviews extraction pipeline: 
# from raw unlabeled dataset -> data preprocessing + PCA -> neutral reviews embedding dataset
import sys
sys.path.append("..")
import os
import time
import data_preprocessing_variables as dpv
import download_datasets as dd
import one_reviews_dir_dataset_creation as orddc
import unlabeled_reviews_cleaning as urc
import unlabeled_reviews_texts_embedding_generation as urtmeg
import dataset_splitter as ds
import generate_compose_multiple_ollama as gcmo
import parallel_embed_generation as peg
import k_means_clustering as kmc
import neutral_reviews_extraction_via_LLM as nrevl

class Neutral_Reviews_Extraction_Pipeline:
    def __init__(self, dataset_info: dict):
        self.dataset_info = dataset_info
    
    def main(self):
        dd.download_large_movie_reviews_dataset(self.dataset_info)
        
        unlabeled_reviews_dir_path = f"../{dpv.DOWNLOADED_DATASETS_ROOT_DIR}/{dpv.USERS_SENTIMENTS_DATASET_INFO['dataset_name']}/train"
        unlabeled_dataset_creator = orddc.One_Reviews_Dir_Dataset_Creator(
            unlabeled_reviews_dir_path, 'unsup'
        )
        unlabeled_dataset = unlabeled_dataset_creator.main()
        cleaned_dataset = urc.main(unlabeled_dataset)

        for _ in range(2):
            text_embed_generator = urtmeg.Texts_Embedding_Dataset_Generator(
                dpv.MODEL_FOR_EMBEDDING_GENERATION,
                dpv.TEMP_CONTAINER_OLLAMA_HOST,
                dpv.TEMP_CONTAINER_OLLAMA_PORT,
                dpv.CLEANED_UNLABELED_REVIEWS_DATASET,
                dpv.EMBED_COMPUTE_TIME_ESTIMATION_SET_NAME,
                computing_time_estimation_mode = True
            )
        compose_generator = gcmo.Compose_YAML_File_Generator(text_embed_generator)
        compose_generator.main()
        
        os.system("sudo docker compose up")
        time.sleep(5)
        ds_splitter = ds.Dataset_Splitter(dpv.CLEANED_UNLABELED_REVIEWS_DATASET)
        ds_splitter.small_datasets_amount = compose_generator.ollama_instances_for_parallel_computing
        ds_splitter.main()
        parallel_embed_gen_manager = peg.Parallel_Embedding_Generation_Manager(
            dpv.SMALL_SETS_FOR_EMBED_GENERATION_DIR,
            dpv.REVIEWS_EMBEDDING_DATASETS_DIR
        )
        parallel_embed_gen_manager.main()
        os.system("sudo docker compose stop")

        clusters_manager = kmc.K_Means_Clustering_Manager(
            dpv.REVIEWS_EMBEDDING_DATASETS_DIR, dpv.BIG_REVIEWS_EMBED_DATASET,
            dpv.CLASSIFIED_REVIEWS_DATASET_VIA_CLUSTERING, 3
        )
        clusters_manager.main()

        neutral_reviews_extraction_manager = nrevl.Neutral_Reviews_Extraction_Manager(
            dpv.MODELS_FOR_NEUTRAL_REVIEWS_EXTRACTION,
            dpv.TEMP_CONTAINER_OLLAMA_HOST, dpv.NEUTRAL_REVIEWS_CONTAINER_OLLAMA_PORT,
            dpv.CLASSIFIED_REVIEWS_DATASET_PATH, dpv.NEUTRAL_REVIEWS_DATASET
        )
        neutral_reviews_extraction_manager.main()

if __name__ == "__main__":
    neutral_reviews_extract_pipeline = Neutral_Reviews_Extraction_Pipeline(
        dpv.USERS_SENTIMENTS_DATASET_INFO
    )
    neutral_reviews_extract_pipeline.main()