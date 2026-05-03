import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D
import numpy as np

def main():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X = normalize_images_px_values(train_X)
    test_X = normalize_images_px_values(test_X)
    images_classes_amount = get_images_classes_amount(train_y)

def get_CNN_model(classes_amount: int):
    # TODO: think, how to create min model for image classification
    pass

def normalize_images_px_values(dataset_X: np.array):
    return dataset_X / 255

def get_images_classes_amount(dataset_Y: np.array):
    return len(np.unique(dataset_Y))

if __name__ == '__main__':
    main()