# Neural Network creation for classifying
# handwritten digits from the MNIST dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

MNIST_dataset = mnist.load_data()
(X_train, y_train), (X_test, y_test) = MNIST_dataset
# print(X_train[:1, :])
X_train_normalized = np.float32(X_train / 255)
X_test_normalized = np.float32(X_test / 255)

nn_model = Sequential([
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

nn_model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy']
)

# by default - batch size = 32(images amount per iteration, gradient update)
nn_model.fit(X_train_normalized, y_train, epochs = 5)

nn_prediction = nn_model.predict(X_test_normalized)
