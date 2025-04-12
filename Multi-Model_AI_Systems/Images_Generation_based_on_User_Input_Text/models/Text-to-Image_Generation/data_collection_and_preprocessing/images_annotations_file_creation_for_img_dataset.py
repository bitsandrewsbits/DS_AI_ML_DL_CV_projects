# merging images annotations and images into
# separate columns but in the same dataset.
from pycocotools.coco import COCO
import numpy as np
import pandas as pd

class Images_Annotations_File_Creation_For_Images_Dataset:
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

        self.images_dataset_metadata = pd.DataFrame()
        self.images_dataset_JSON_metadata = []

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

    def main(self):
        self.add_image_ID_column_to_dataset()
        self.add_image_filename_and_img_annotation_columns_to_dataset()
        self.remove_image_ID_column()
        print(self.images_dataset_metadata)
        self.convert_dataset_into_JSON_format()
        self.create_JSONL_images_metadata_file()

    def add_image_ID_column_to_dataset(self):
        self.images_dataset_metadata['image_ID'] = self.images_IDs

    def add_image_filename_and_img_annotation_columns_to_dataset(self):
        self.images_dataset_metadata["file_name"] = \
        self.images_dataset_metadata['image_ID'].apply(
            self.get_image_filename
        )
        self.images_dataset_metadata["image_annotations"] = \
        self.images_dataset_metadata['image_ID'].apply(
            self.get_img_annotations
        )

    def get_image_filename(self, img_id: int):
        image_info = self.get_image_info_by_ID(img_id)
        return image_info['file_name']

    def get_img_annotations(self, img_id: int):
        image_annotation_IDs = self.get_image_annotation_IDs(img_id)
        image_annotations = self.get_image_annotations_by_IDs(image_annotation_IDs)
        return image_annotations

    def get_image_info_by_ID(self, img_id: int):
        return self.images_cocoAPI.loadImgs([img_id])[0]

    def get_image_annotation_IDs(self, img_id: int):
        return self.annotations_cocoAPI.getAnnIds(imgIds = [img_id])

    def get_image_annotations_by_IDs(self, annotations_IDs: list):
        image_annotations = []
        for annotation_info in self.annotations_cocoAPI.loadAnns(annotations_IDs):
            image_annotations.append(annotation_info['caption'])
        return image_annotations

    def remove_image_ID_column(self):
        self.images_dataset_metadata.drop(
            'image_ID',
            axis = 1,
            inplace = True
        )

    def convert_dataset_into_JSON_format(self):
        self.images_dataset_JSON_metadata = self.images_dataset_metadata.to_json(
            orient = 'records', lines = True
        )

    def create_JSONL_images_metadata_file(self):
        with open(f'{self.images_dir}/metadata.jsonl', 'w') as imgs_metadata:
            imgs_metadata.write(self.images_dataset_JSON_metadata)

if __name__ == '__main__':
    stable_diffusion_ds = Images_Annotations_File_Creation_For_Images_Dataset(
        'validation2017_for_fine-tune',
        'instances_val2017.json',
        'captions_val2017.json'
    )
    stable_diffusion_ds.main()
