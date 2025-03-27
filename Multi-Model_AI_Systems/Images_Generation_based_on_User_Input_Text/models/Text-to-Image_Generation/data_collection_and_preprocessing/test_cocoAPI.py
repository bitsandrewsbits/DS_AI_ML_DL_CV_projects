from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import random

#data dirs - VALIDATION data
dataDir = 'data'
validation_dir = 'validation2017_for_fine-tune'
annotation_file = f'{dataDir}/{validation_dir}/annotations/instances_val2017.json'

# initialize COCO API for instance annotations
cocoAPI = COCO(annotation_file)

# display COCO categories and supercategories
categories = cocoAPI.loadCats(cocoAPI.getCatIds())
categories_names = [category['name'] for category in categories]
print(f"COCO categories: \n{categories_names}\n")
supercategories = set([category['supercategory'] for category in categories])
print(f'COCO supercategories:\n{supercategories}')

test_categories = ['person', 'train']
test_categories_IDs = cocoAPI.getCatIds(catNms = test_categories)
print('Test choose categories IDs:')
print(test_categories_IDs)
test_images_IDs = cocoAPI.getImgIds(catIds = test_categories_IDs)
print('Test categories Images IDs:')
print(test_images_IDs)

# show test image by random ID
random_image_ID = random.randint(0, len(test_images_IDs))

def get_image_info_by_ID(img_id: int):
    return cocoAPI.loadImgs(test_images_IDs[img_id])[0]

img_info = get_image_info_by_ID(random_image_ID)
print('Image Info:')
print(img_info)

def get_image_by_img_info(img_info: dict):
    return f"{dataDir}/{validation_dir}/images/{img_info['file_name']}"

image = io.imread(get_image_by_img_info(img_info))
# print(image) # image as numpy ndarray
