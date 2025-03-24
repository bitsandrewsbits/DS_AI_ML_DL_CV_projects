from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

#data dirs - TRAIN data
dataDir = 'data'
dataType = 'train2017'
annotationFile = f'{dataDir}/{dataType}/annotations/instances_{dataType}.json'

# initialize COCO API for instance annotations
cocoAPI = COCO(annotationFile)

# display COCO categories and supercategories
categories = cocoAPI.loadCats(cocoAPI.getCatIds())
categories_names = [category['name'] for category in categories]
print(f"COCO categories: \n{categories_names}\n")
supercategories = set([category['supercategory'] for category in categories])
print(f'COCO supercategories:\n{supercategories}')
