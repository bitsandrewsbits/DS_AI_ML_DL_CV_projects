# merging images annotations and images into
# separate columns but in the same dataset.

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io

class Dataset_Creating_for_Stable_Diffusion:
    def __init__(self, dataset_dir: str,
        images_annotation_file: str, images_description_annot_file: str):
        self.dataDir = 'data'
        self.dataset_dir = dataset_dir
        self.images_dir = f'{self.dataDir}/{self.dataset_dir}/images'
        self.images_annotation_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_annotation_file}'
        self.images_description_annot_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_description_annot_file}'

        self.images_cocoAPI = COCO(self.images_annotation_file)
        self.annotations_cocoAPI = COCO(self.images_description_annot_file)

        self.images_categories_IDs = self.images_cocoAPI.getCatIds()
        self.images_IDs = []

    def define_all_images_IDs(self):
        self.images_IDs = self.images_cocoAPI.getImgIds(
            catIds = self.images_categories_IDs
        )

if __name__ == '__main__':
    stable_diffusion_ds = Dataset_Creating_for_Stable_Diffusion(
        'validation2017_for_fine-tune',
        'instances_val2017.json',
        'captions_val2017.json'
    )
    print(stable_diffusion_ds.images_categories_IDs)
    stable_diffusion_ds.define_all_images_IDs()
    print(stable_diffusion_ds.images_IDs)
