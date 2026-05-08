import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

def main():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X = normalize_images_px_values(train_X)
    test_X = normalize_images_px_values(test_X)
    images_classes_amount = get_images_classes_amount(train_y)
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    batch_size = 128
    input_shape = (train_X.shape[1], train_X.shape[2], train_X.shape[3])
    cnn_model = get_CNN_model(input_shape, images_classes_amount)
    model_train_logs = train_model_and_get_train_logs(
        cnn_model, train_X, train_y, batch_size
    )
    plot_accuracy_and_loss_per_epoch(model_train_logs)

def train_model_and_get_train_logs(model, train_set_X, train_set_y, batch_size = 16):
    return model.fit(
        x = train_set_X, y = train_set_y,
        batch_size = batch_size, epochs = 100,
        validation_split = 0.2
    )        

def get_CNN_model(input_shape: tuple, classes_amount: int):
    input_layer = Input(shape = input_shape)

    conv_layer = Conv2D(
        filters = 32, kernel_size = (3, 3),
        padding = 'same', activation = 'relu'
    )(input_layer)
    normalization_layer = BatchNormalization()(conv_layer)
    mp2d_layer = MaxPool2D(pool_size = (3, 3))(normalization_layer)
    dropout_layer = Dropout(0.3)(mp2d_layer)
    
    conv_layer = Conv2D(
        filters = 64, kernel_size = (3, 3),
        padding = 'same', activation = 'relu'
    )(dropout_layer)
    normalization_layer = BatchNormalization()(conv_layer)
    mp2d_layer = MaxPool2D(pool_size = (3, 3))(normalization_layer)
    dropout_layer = Dropout(0.2)(mp2d_layer)

    conv_layer = Conv2D(
        filters = 128, kernel_size = (3, 3), activation = 'relu'
    )(dropout_layer)
    normalization_layer = BatchNormalization()(conv_layer)

    flatten_layer = Flatten()(normalization_layer)
    dropout_layer = Dropout(0.2)(flatten_layer)
    dense_layer = Dense(1024)(dropout_layer)
    dropout_layer = Dropout(0.1)(dense_layer)
    dense_layer = Dense(32)(dropout_layer)
    output_layer = Dense(classes_amount, activation = 'softmax')(dense_layer)

    model = Model(input_layer, output_layer)
    model.compile(
        optimizer = 'adam', metrics = ['accuracy'],
        loss = 'sparse_categorical_crossentropy'
    )
    return model

def plot_accuracy_and_loss_per_epoch(train_logs):
    fig, ax = plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    plt.plot(train_logs.history['accuracy'], label = 'train_accuracy')
    plt.plot(train_logs.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(train_logs.history['loss'], label = 'train_loss')
    plt.plot(train_logs.history['val_loss'], label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()

def normalize_images_px_values(dataset_X: np.array):
    return dataset_X / 255

def get_images_classes_amount(dataset_Y: np.array):
    return len(np.unique(dataset_Y))

if __name__ == '__main__':
    main()