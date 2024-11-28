# Neural Network creation for classifying
# handwritten digits from the MNIST dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

MNIST_dataset = mnist.load_data()
(X_train, y_train), (X_test, y_test) = MNIST_dataset
# print(X_train[:1, :])
X_train_normalized = np.float32(X_train / 255)
