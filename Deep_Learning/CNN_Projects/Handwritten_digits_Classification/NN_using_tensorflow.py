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
# verbose - level of information during training
epochs_amount = 5
training_events = nn_model.fit(X_train_normalized, y_train, epochs = epochs_amount, verbose = 2)

digits_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nn_prediction = (nn_model.predict(X_test_normalized) > 0.5).astype('int32')
target_prediction_y_vector = np.dot(nn_prediction, digits_vector)

print('Classification Report:')
target_classes = [f"prediction of digit {i}" for i in range(10)]
print(classification_report(y_test, target_prediction_y_vector, target_names = target_classes))
print('Accuracy:', accuracy_score(y_test, target_prediction_y_vector))

first_test_samples_amount = 10
first_test_np_arr_images = X_test[:first_test_samples_amount]
first_prediction_y_values = target_prediction_y_vector[:first_test_samples_amount]
first_test_y_values = y_test[:first_test_samples_amount]

def show_subplots(images: list, subplots_columns_number = 5):
    subplots_rows_number = get_subplot_rows_number(len(images))
    plt.figure(figsize = (30, 15))
    for image_index in range(len(images)):
        plt.subplot(subplots_rows_number, subplots_columns_number, image_index + 1)
        plt.title(f"Digit Image #{image_index + 1}\n\
        Prediction = {first_prediction_y_values[image_index]}\n\
        True Digit = {first_test_y_values[image_index]}")
        plt.imshow(images[image_index], cmap = 'grey')
    plt.show()

def get_subplot_rows_number(list_length: int, subplots_columns_number = 5):
    if list_length % subplots_columns_number != 0:
        return list_length // subplots_columns_number + 1
    elif list_length % subplots_columns_number == 0:
        return list_length // subplots_columns_number

show_subplots(first_test_np_arr_images)

def show_training_accuracy_loss_model_plots():
    plt.subplots_adjust(hspace = 0.5)
    create_model_accuracy_during_training_plot()
    create_model_loss_during_training_plot()
    plt.show()

def get_events_during_model_training():
    print('Training Events:')
    print(training_events.history)
    return training_events.history

loss_accuracy_during_training = get_events_during_model_training()

def get_accuracy_during_model_training():
    return loss_accuracy_during_training['accuracy']

def get_loss_during_model_training():
    return loss_accuracy_during_training['loss']

def get_epochs_numbers():
    return [i for i in range(epochs_amount)]

def create_model_accuracy_during_training_plot():
    training_accuracy_per_epochs = get_accuracy_during_model_training()
    epochs_numbers = get_epochs_numbers()
    plt.subplot(2, 1, 1)
    plt.plot(epochs_numbers, training_accuracy_per_epochs)
    plt.title("Accuracy(Epoch)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

def create_model_loss_during_training_plot():
    training_loss_per_epochs = get_loss_during_model_training()
    epochs_numbers = get_epochs_numbers()
    plt.subplot(2, 1, 2)
    plt.plot(epochs_numbers, training_loss_per_epochs)
    plt.title("Loss(Epoch)")
    plt.xlabel("epoch")
    plt.ylabel("loss")

show_training_accuracy_loss_model_plots()

# Observation: with this training parameters, model accuracy pretty well.
# Accuracy increases most of all at first 2-3 epochs. After that, it looks like
# more without big increasing steps as saturation curve.
