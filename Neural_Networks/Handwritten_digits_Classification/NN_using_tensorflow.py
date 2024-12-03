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

digits_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nn_prediction = (nn_model.predict(X_test_normalized) > 0.5).astype('int32')
target_prediction_y_vector = np.dot(nn_prediction, digits_vector)

print('Classification Report:')
target_classes = ['digit 0', 'digit 1', 'digit 2', 'digit 3',
                  'digit 4', 'digit 5', 'digit 6', 'digit 7',
                  'digit 8', 'digit 9']
print(classification_report(y_test, target_prediction_y_vector, target_names = target_classes))
print('Accuracy:', accuracy_score(y_test, target_prediction_y_vector))

first_test_images_amount = 10
first_test_np_arr_images = X_test[:first_test_images_amount]

def show_subplots(images: list, subplots_columns_number = 5):
    subplots_rows_number = get_subplot_rows_number(len(images))
    plt.figure(figsize = (30, 15))
    for image_index in range(len(images)):
        plt.subplot(subplots_rows_number, subplots_columns_number, image_index + 1)
        plt.title(f"Digit Image #{image_index + 1}")
        plt.imshow(images[image_index], cmap = 'grey')
    plt.show()

def get_subplot_rows_number(list_length: int, subplots_columns_number = 5):
    if list_length % 2 != 0:
        return list_length // subplots_columns_number + 1
    elif list_length % 2 == 0:
        return list_length // subplots_columns_number

show_subplots(first_test_np_arr_images)
