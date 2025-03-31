from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import random

class Image_and_Description_Extraction:
    def __init__(self, dataset_dir: str,
        images_annotation_file: str, images_description_annot_file: str):
        self.dataDir = 'data'
        self.dataset_dir = dataset_dir
        self.images_dir = f'{self.dataDir}/{self.dataset_dir}/images'
        self.images_annotation_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_annotation_file}'
        self.images_description_annot_file = f'{self.dataDir}/{self.dataset_dir}/annotations/{images_description_annot_file}'

        self.images_cocoAPI = COCO(self.images_annotation_file)
        self.annotations_cocoAPI = COCO(self.images_description_annot_file)

        self.images_categories_names = self.get_images_catogories_names()
        self.random_images_categories = self.get_random_images_categories()
        self.random_categories_IDs = self.get_random_categories_IDs()
        self.images_IDs_of_random_categories = self.get_images_IDs_from_random_categories()

    def main(self):
        print('Random Categories:', self.random_images_categories)
        image_ID = self.get_image_ID_by_random()
        self.show_image(image_ID)
        image_annotations_IDs = self.get_image_annotation_ID(image_ID)
        print('Image annotation:')
        print(self.get_image_annotations(image_annotations_IDs))

    def get_images_catogories_names(self):
        categories_info = self.images_cocoAPI.loadCats(self.images_cocoAPI.getCatIds())
        categories_names = [category_info['name'] for category_info in categories_info]
        print(f"COCO categories: \n{categories_names}\n")
        return categories_names

    def get_random_images_categories(self, categories_amount = 1):
        random_image_categories = []
        for _ in range(categories_amount):
            random_index = random.randint(0, len(self.images_categories_names))
            random_image_categories.append(self.images_categories_names[random_index])
        return random_image_categories

    def get_random_categories_IDs(self):
        return self.images_cocoAPI.getCatIds(catNms = self.random_images_categories)

    def get_images_IDs_from_random_categories(self):
        return self.images_cocoAPI.getImgIds(catIds = self.random_categories_IDs)

    def get_image_ID_by_random(self):
        images_IDs_random_index = random.randint(0, len(self.images_IDs_of_random_categories))
        image_ID = self.images_IDs_of_random_categories[images_IDs_random_index]
        return image_ID

    def get_image_obj(self, image_id: int):
        img_info = self.get_image_info_by_ID(image_id)
        image = io.imread(self.get_image_by_img_info(img_info))
        return image

    def get_image_info_by_ID(self, img_id: int):
        return self.images_cocoAPI.loadImgs(img_id)[0]

    def get_image_by_img_info(self, img_info: dict):
        return f"{self.dataDir}/{self.dataset_dir}/images/{img_info['file_name']}"

    def show_image(self, image_id: int):
        image = self.get_image_obj(image_id)
        plt.imshow(image)
        plt.show()

    def get_image_annotation_ID(self, img_id: int):
        return self.annotations_cocoAPI.getAnnIds(imgIds = [img_id])

    def get_image_annotations(self, annotations_IDs: list):
        image_annotations = []
        for annotation_info in self.annotations_cocoAPI.loadAnns(annotations_IDs):
            image_annotations.append(annotation_info['caption'])
        return image_annotations

if __name__ == '__main__':
    image_and_description = Image_and_Description_Extraction(
        'validation2017_for_fine-tune',
        'instances_val2017.json',
        'captions_val2017.json'
    )
    image_and_description.main()
