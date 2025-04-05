# merging images annotations and images into
# separate columns but in the same dataset.
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

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

        self.images_IDs_by_categories_IDs = self.images_cocoAPI.catToImgs
        self.images_IDs = self.get_all_unique_images_IDs()

        self.dataset_for_Stable_Diffusion = pd.DataFrame()

    def get_all_unique_images_IDs(self):
        unique_images_IDs = self.images_IDs_by_categories_IDs[1]
        for category_ID in self.images_IDs_by_categories_IDs:
            unique_images_IDs = (
                np.union1d(
                    unique_images_IDs,
                    self.images_IDs_by_categories_IDs[category_ID]
                )
            )
        return unique_images_IDs

    def main(self, dataset_type = "TRAIN"):
        self.add_image_ID_column_to_dataset()
        self.add_image_and_img_annotation_columns_to_dataset()
        self.remove_image_ID_column()
        self.save_dataset_to_CSV(dataset_type)

    def add_image_ID_column_to_dataset(self):
        self.dataset_for_Stable_Diffusion['image_ID'] = self.images_IDs

    def add_image_and_img_annotation_columns_to_dataset(self):
        self.dataset_for_Stable_Diffusion['image'] = \
        self.dataset_for_Stable_Diffusion['image_ID'].apply(
            self.get_image_as_arrays
        )
        self.dataset_for_Stable_Diffusion['image_annotations'] = \
        self.dataset_for_Stable_Diffusion['image_ID'].apply(
            self.get_img_annotations
        )

    def get_image_as_arrays(self, image_id: int):
        img_info = self.get_image_info_by_ID(image_id)
        image = io.imread(self.get_image_by_img_info(img_info))
        return image

    def get_img_annotations(self, img_id: int):
        image_annotation_IDs = self.get_image_annotation_IDs(img_id)
        image_annotations = self.get_image_annotations_by_IDs(image_annotation_IDs)
        return image_annotations

    def get_image_info_by_ID(self, img_id: int):
        return self.images_cocoAPI.loadImgs([img_id])[0]

    def get_image_by_img_info(self, img_info: dict):
        return f"{self.dataDir}/{self.dataset_dir}/images/{img_info['file_name']}"

    def get_image_annotation_IDs(self, img_id: int):
        return self.annotations_cocoAPI.getAnnIds(imgIds = [img_id])

    def get_image_annotations_by_IDs(self, annotations_IDs: list):
        image_annotations = []
        for annotation_info in self.annotations_cocoAPI.loadAnns(annotations_IDs):
            image_annotations.append(annotation_info['caption'])
        return image_annotations

    def remove_image_ID_column(self):
        self.dataset_for_Stable_Diffusion.drop(
            'image_ID',
            axis = 1,
            inplace = True
        )

    def save_dataset_to_CSV(self, dataset_type: str):
        self.dataset_for_Stable_Diffusion.to_csv(
            f"{self.dataDir}/{self.dataset_dir}/dataset({dataset_type})_for_Stable_Diffusion_model.csv",
            index = False
        )

if __name__ == '__main__':
    stable_diffusion_ds = Dataset_Creating_for_Stable_Diffusion(
        'validation2017_for_fine-tune',
        'instances_val2017.json',
        'captions_val2017.json'
    )
    stable_diffusion_ds.main("VALIDATION")
