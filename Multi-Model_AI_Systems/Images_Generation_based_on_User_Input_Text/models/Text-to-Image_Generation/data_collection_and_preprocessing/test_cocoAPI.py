from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

#data dirs - TEST data
dataDir = 'data'
dataType = 'test2017'
annotation_file = f'{dataDir}/{dataType}/annotations/image_info_{dataType}.json'

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
print(test_categories_IDs)
test_images_IDs = cocoAPI.getImgIds(catIds = test_categories_IDs)
print('Test categories Images IDs:')
print(test_images_IDs)
