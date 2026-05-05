import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
import numpy as np

def main():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X = normalize_images_px_values(train_X)
    test_X = normalize_images_px_values(test_X)
    images_classes_amount = get_images_classes_amount(train_y)
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    batch_size = 32
    input_shape = (train_X.shape[1], train_X.shape[2], train_X.shape[3])
    cnn_model = get_CNN_model(input_shape, images_classes_amount)
    train_model(cnn_model, train_X, train_y, batch_size)

def train_model(model, train_set_X, train_set_y, batch_size = 16):
    model.fit(
        x = train_set_X, y = train_set_y,
        batch_size = batch_size, epochs = 10,
        validation_split = 0.1
    )        

def get_CNN_model(input_shape: tuple, classes_amount: int):
    input_layer = Input(shape = input_shape)
    print("Input shape:", input_layer.shape)
    conv_layer = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same')(input_layer)
    print("Conv2D shape:", conv_layer.shape)
    mp2d_layer = MaxPool2D(pool_size = (2, 2))(conv_layer)
    print("Global max pool shape:", mp2d_layer.shape)
    conv_layer = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(mp2d_layer)
    mp2d_layer = MaxPool2D(pool_size = (2, 2))(conv_layer)
    flatten_layer = Flatten()(mp2d_layer)
    dense_layer = Dense(256)(flatten_layer)
    output_layer = Dense(classes_amount, activation = 'softmax')(dense_layer)

    model = Model(input_layer, output_layer)
    model.compile(
        optimizer = 'adam', metrics = ['accuracy'],
        loss = 'sparse_categorical_crossentropy'
    )
    return model

def normalize_images_px_values(dataset_X: np.array):
    return dataset_X / 255

def get_images_classes_amount(dataset_Y: np.array):
    return len(np.unique(dataset_Y))

if __name__ == '__main__':
    main()