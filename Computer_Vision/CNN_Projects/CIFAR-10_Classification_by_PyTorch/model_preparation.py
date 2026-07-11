# model ML flow process - model loading prepared data, training, evaluating, tuning
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchmetrics import Accuracy
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
		self.batch_shape = self.get_batch_size()

		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.CNN_model = cnnma.TinyVGG_Architecture_CNN(
			self.color_channels, self.hidden_units,
			self.classes_amount, self.batch_shape
		).to(self.compute_device)

		self.loss_func = torch.nn.CrossEntropyLoss()
		self.optimizer = SGD(
			params = self.CNN_model.parameters(), lr = 0.01
		)
		self.accuracy_func = Accuracy(
			task = "multiclass",
			num_classes = self.classes_amount
		).to(self.compute_device)
		self.epochs = 70

		self.train_loss = []
		self.train_accuracy = []
		self.valid_loss = []
		self.valid_accuracy = []

		self.trained_models_dir = "trained_models"

	def main(self):
		trained_model = self.get_trained_model()
		self.show_loss_accuracy_curves()
		self.save_trained_model("cnn_model.pth")

	def get_trained_model(self):
		for epoch in range(1, self.epochs + 1):
			print(f"Epoch #{epoch}:", end = '')
			self.train_step()

	def train_step(self):
		epoch_sum_loss = 0
		epoch_sum_accuracy = 0
		self.CNN_model.train()
		for (images_batch, labels_batch) in self.train_dataloader:
			images_batch = images_batch.to(self.compute_device)
			labels_batch = labels_batch.to(self.compute_device)
			logits = self.CNN_model(images_batch)
			pred_probs = torch.softmax(logits, dim = 1)
			pred_labels = torch.argmax(pred_probs, dim = 1)

			batch_loss = self.loss_func(logits, labels_batch)
			batch_accuracy = self.accuracy_func(pred_labels, labels_batch)

			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
			
			epoch_sum_loss += batch_loss
			epoch_sum_accuracy += batch_accuracy

		epoch_loss = round(epoch_sum_loss.item() / len(self.train_dataloader), 3)
		epoch_accuracy = round(epoch_sum_accuracy.item() / len(self.train_dataloader), 3)
		print(f"train loss: {epoch_loss}, train accuracy: {epoch_accuracy}", end = '')
		self.evaluate_model()

		self.train_loss.append(epoch_loss)
		self.train_accuracy.append(epoch_accuracy)

	def evaluate_model(self):
		eval_sum_loss = 0
		eval_sum_accuracy = 0
		self.CNN_model.eval()
		with torch.inference_mode():
			for (images_batch, labels_batch) in self.test_dataloader:
				images_batch = images_batch.to(self.compute_device)
				labels_batch = labels_batch.to(self.compute_device)
				logits = self.CNN_model(images_batch)
				pred_probs = torch.softmax(logits, dim = 1)
				pred_labels = torch.argmax(pred_probs, dim = 1)

				batch_eval_loss = self.loss_func(logits, labels_batch)
				batch_eval_accuracy = self.accuracy_func(pred_labels, labels_batch)
				eval_sum_loss += batch_eval_loss
				eval_sum_accuracy += batch_eval_accuracy
			eval_loss = round(eval_sum_loss.item() / len(self.test_dataloader), 3)
			eval_accuracy = round(eval_sum_accuracy.item() / len(self.test_dataloader), 3)
		print(f", eval loss: {eval_loss}, eval accuracy: {eval_accuracy}")
		
		self.valid_loss.append(eval_loss)
		self.valid_accuracy.append(eval_accuracy)

	def get_batch_size(self):
		for batch in self.train_dataloader:
			return batch[0].shape

	def show_loss_accuracy_curves(self):
		fig, ax = plt.subplots(2, 1)
		epochs = [i for i in range(1, len(self.train_loss) + 1)]

		plt.subplot(2, 1, 1)
		plt.title("Train/Valid Loss")
		plt.plot(epochs, self.train_loss, label = 'train-loss')
		plt.plot(epochs, self.valid_loss, label = 'valid-loss')
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.legend()

		plt.subplot(2, 1, 2)
		plt.title("Train/Valid Accuracy")
		plt.plot(epochs, self.train_accuracy, label = 'train-accuracy')
		plt.plot(epochs, self.valid_accuracy, label = 'valid-accuracy')
		plt.xlabel("epoch")
		plt.ylabel("accuracy")
		plt.legend()
		
		plt.show()

	def save_trained_model(self, model_name: str):
		trained_models_dir_path = Path(self.trained_models_dir)
		trained_models_dir_path.mkdir(parents = True, exist_ok = True)
		trained_model_save_path = trained_models_dir_path / model_name
		print(f"Saving trained model to {trained_model_save_path}...")
		torch.save(
			obj = self.CNN_model.state_dict(),
			f = trained_model_save_path
		)

if __name__ == "__main__":
	model_preparation = CNN_Model_Preparation()
	model_preparation.main()