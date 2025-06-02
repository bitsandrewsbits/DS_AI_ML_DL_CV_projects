# merging images annotations and images into
# separate columns but in the same dataset.
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import re

class Images_Annotations_File_Creation_For_Images_Dataset:
    def __init__(self, dataset_dir: str,
        images_annotation_file: str, images_description_annot_file: str):
        self.dataDir = 'data'
        self.dataset_dir = dataset_dir
        self.images_dir = f'{self.dataDir}/{self.dataset_dir}/images'
        self.images_annotation_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_annotation_file}'
        self.images_description_annot_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_description_annot_file}'
        self.validation_data_dir_regex = re.compile(r'.*validation.*')
        self.validation_samples_amount = 200

        self.images_cocoAPI = COCO(self.images_annotation_file)
        self.annotations_cocoAPI = COCO(self.images_description_annot_file)

        self.categories_IDs = self.images_cocoAPI.getCatIds()
        self.categories_info = self.images_cocoAPI.loadCats(self.categories_IDs)

        self.images_IDs_by_categories_IDs = self.images_cocoAPI.catToImgs
        self.images_amount_per_categories_IDs = {}
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
        if re.match(self.validation_data_dir_regex, self.dataset_dir):
            if len(unique_images_IDs) > self.validation_samples_amount:
                return unique_images_IDs[:self.validation_samples_amount]
            return unique_images_IDs
        else:
            return unique_images_IDs

    def main(self):
        self.define_images_amount_per_categories_IDs()
        self.define_anomaly_big_imgs_categories()

        self.add_image_ID_column_to_dataset()
        self.add_image_filename_column_to_dataset()
        self.add_image_annotations_column_to_dataset()
        self.remove_image_ID_column()

        self.convert_dataset_into_JSON_format()
        self.create_JSONL_images_metadata_file()
        print(self.images_dataset_metadata)

    def define_images_amount_per_categories_IDs(self):
        for category_ID in self.images_IDs_by_categories_IDs:
            self.images_amount_per_categories_IDs[category_ID] = len(
                self.images_IDs_by_categories_IDs[category_ID]
            )
        return True

    def define_anomaly_big_imgs_categories(self):
        st_deviation = np.std(list(self.images_amount_per_categories_IDs.values()))
        mean_val = np.average(list(self.images_amount_per_categories_IDs.values()))
        print('Standart Deviation =', st_deviation)
        print('Average =', mean_val)
        for categ_ID in self.images_amount_per_categories_IDs:
            if self.images_amount_per_categories_IDs[categ_ID] > st_deviation * 2 or \
            self.images_amount_per_categories_IDs[categ_ID] > mean_val * 2:
                categ_name = self.get_category_name_by_ID(categ_ID)
                print(f'Anomaly category - {categ_name}(ID = {categ_ID}) - detected.')
                print('Images amount =', self.images_amount_per_categories_IDs[categ_ID])
                print('-' * 20)
        return True

    def get_category_name_by_ID(self, categ_ID: int):
        return self.images_cocoAPI.loadCats([categ_ID])[0]['name']

    def add_image_ID_column_to_dataset(self):
        self.images_dataset_metadata['image_ID'] = self.images_IDs

    def add_image_filename_column_to_dataset(self):
        self.images_dataset_metadata["image"] = \
        self.images_dataset_metadata['image_ID'].apply(
            self.get_image_filename_path
        )

    def add_image_annotations_column_to_dataset(self):
        self.images_dataset_metadata["image_annotations"] = \
        self.images_dataset_metadata['image_ID'].apply(
            self.get_img_annotations
        )

    def get_image_filename_path(self, img_id: int):
        image_info = self.get_image_info_by_ID(img_id)
        return f"{self.images_dir}/{image_info['file_name']}"

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
    img_annotations_ds = Images_Annotations_File_Creation_For_Images_Dataset(
        'train2017',
        'instances_train2017.json',
        'captions_train2017.json'
    )
    img_annotations_ds.main()
