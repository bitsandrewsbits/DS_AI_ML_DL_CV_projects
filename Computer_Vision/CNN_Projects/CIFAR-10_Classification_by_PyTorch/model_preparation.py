# model ML flow process - model loading prepared data, training, evaluating, tuning
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import SGD

import CNN_model_architecture as cnnma

class CNN_Model_Preparation:
	def __init__(self):
		self.datasets_dir = "datasets"
		self.train_set = CIFAR10(
			root = self.datasets_dir,
			train = True,
			download = True,
			transform = ToTensor(),
			target_transform = None
		)
		self.test_set = CIFAR10(
			root = self.datasets_dir,
			train = False,
			download = True,
			transform = ToTensor(),
			target_transform = None
		)
		self.batch_size = 32
		self.train_dataloader = DataLoader(
			dataset = self.train_set,
			batch_size = self.batch_size,
			shuffle = True
		)
		self.test_dataloader = DataLoader(
			dataset = self.test_set,
			batch_size = self.batch_size
		)
		self.color_channels = 3
		self.hidden_units = 32
		self.classification_classes = self.train_set.classes
		self.classes_amount = len(self.classification_classes)
		self.image_shape = self.train_set[0][0].shape

		self.CNN_model = cnnma.TinyVGG_Architecture_CNN(
			self.color_channels, self.hidden_units,
			self.classes_amount, self.image_shape
		)
		self.loss_func = torch.nn.CrossEntropyLoss()
		self.optimizer = SGD(
			params = self.CNN_model.parameters(), lr = 0.1
		)

	def main(self):
		pass

	def get_trained_model(self):
		pass

	def train_step(self):
		# TODO: create method.
		pass

if __name__ == "__main__":
	model_preparation = CNN_Model_Preparation()
	model_preparation.main()