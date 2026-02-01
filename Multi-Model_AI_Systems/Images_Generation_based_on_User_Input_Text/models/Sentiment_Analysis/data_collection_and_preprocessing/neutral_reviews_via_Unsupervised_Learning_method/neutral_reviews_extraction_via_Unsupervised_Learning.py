# neutral reviews extraction via Unsupervised Learning method
# what method - I will select in future commit
import sys
sys.path.append("..")
import pandas as pd
import one_reviews_dir_dataset_creation as ordds
import download_datasets as load_ds

unlabeled_reviews_dir_path = f"../{load_ds.downloaded_datasets_root_dir}/{load_ds.dataset['dataset_name']}/train"
unlabeled_dataset_creator = ordds.One_Reviews_Dir_Dataset_Creator(unlabeled_reviews_dir_path, 'unsup')
unlabeled_dataset = unlabeled_dataset_creator.main()
print(unlabeled_dataset)
