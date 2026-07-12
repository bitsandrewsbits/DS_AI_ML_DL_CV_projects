# program for CIFAR-10 image classification via CNN model
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path
import matplotlib.pyplot as plt

import CNN_model_architecture as cnnma
import global_variables as gv

class Images_Classification:
	def __init__(self):
		self.datasets_dir = gv.DATASETS_DIR
		self.trained_models_dir = gv.TRAINED_MODELS_DIR
		self.trained_models_path = Path(self.trained_models_dir)

		self.test_dataset = CIFAR10(
			root = self.datasets_dir,
			train = False,
			download = True,
			transform = ToTensor(),
			target_transform = None
		)
		self.classes_names = self.test_dataset.classes
		self.image_shape = self.test_dataset[0][0].shape
		self.input_shape = gv.MODEL_HYPERPARAMETERS["color_channels"]
		self.hidden_units = gv.MODEL_HYPERPARAMETERS["hidden_units"]
		self.output_shape = len(self.classes_names)
		self.batch_size = (2, self.image_shape[0], self.image_shape[1], self.image_shape[2])
		
		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model_with_architecture = cnnma.TinyVGG_Architecture_CNN(
			self.input_shape, self.hidden_units,
			self.output_shape, self.batch_size
		)

	def main(self):
		print("Loading model weights...")
		self.load_model_weights("cnn_model.pth")
		self.model_with_architecture.to(self.compute_device)
		# TODO: create random images sampling method from test dataset.
		# TODO: create method(s) for testing of loaded model on test dataset.
		# TODO: create method to show sample images and their true/predict labels
		# in different colors.

	def load_model_weights(self, model_weights_file: str):
		model_weights_path = self.trained_models_path / model_weights_file
		self.model_with_architecture.load_state_dict(
			torch.load(f = model_weights_path)
		)

if __name__ == "__main__":
	images_classification = Images_Classification()
	images_classification.main()