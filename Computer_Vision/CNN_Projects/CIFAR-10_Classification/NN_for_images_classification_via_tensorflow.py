import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Input, Dense, Conv2D

def main():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()

if __name__ == '__main__':
    main()