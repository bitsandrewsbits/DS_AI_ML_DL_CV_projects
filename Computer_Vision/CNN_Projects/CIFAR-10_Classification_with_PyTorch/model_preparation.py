# model ML flow process - model loading prepared data, training, evaluating, tuning
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchmetrics import Accuracy
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

import CNN_model_architecture as cnnma
import global_variables as gv

class CNN_Model_Preparation:
	def __init__(self, trained_model_dirname: str):
		self.datasets_dir = gv.DATASETS_DIR

		self.transforms = v2.Compose([
			v2.ToImage(),
			v2.RandomHorizontalFlip(p = 0.5),
			v2.ToDtype(torch.float32, scale = True)
		])

		self.train_set = CIFAR10(
			root = self.datasets_dir,
			train = True,
			download = True,
			transform = self.transforms,
			target_transform = None
		)
		self.test_set = CIFAR10(
			root = self.datasets_dir,
			train = False,
			download = True,
			transform = self.transforms,
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
		self.color_channels = gv.MODEL_HYPERPARAMETERS["color_channels"]
		self.hidden_units = gv.MODEL_HYPERPARAMETERS["hidden_units"]
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
		self.final_classification_metrics = {
			"loss": 0.0,
			"accuracy": 0.0
		}

		self.trained_models_root_path = Path(gv.TRAINED_MODELS_DIR)
		self.trained_model_dir = trained_model_dirname
		self.trained_model_path = self.create_dirs_for_trained_model()

	def main(self):
		self.train_model()
		eval_plots_path = self.create_custom_eval_plots_dir()
		eval_figure = self.get_loss_accuracy_curves_figure()
		self.save_eval_plots(eval_figure, eval_plots_path)
		final_valid_loss, final_valid_accuracy = self.evaluate_model()
		self.final_classification_metrics["loss"] = final_valid_loss
		self.final_classification_metrics["accuracy"] = final_valid_accuracy
		self.save_trained_model(self.trained_model_path)
		self.save_model_classification_metrics(self.trained_model_path)

	def create_dirs_for_trained_model(self) -> Path:
		self.trained_models_root_path.mkdir(parents = True, exist_ok = True)
		trained_model_path = self.trained_models_root_path / self.trained_model_dir
		trained_model_path.mkdir(parents = True, exist_ok = True)
		return trained_model_path

	def train_model(self):
		for epoch in range(1, self.epochs + 1):
			print(f"Epoch #{epoch}:", end = '')
			self.train_step()
			valid_loss, valid_accuracy = self.evaluate_model()
			self.valid_loss.append(valid_loss)
			self.valid_accuracy.append(valid_accuracy)

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

		self.train_loss.append(epoch_loss)
		self.train_accuracy.append(epoch_accuracy)

	def evaluate_model(self) -> tuple[float, float]:
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
		return eval_loss, eval_accuracy

	def get_batch_size(self):
		for batch in self.train_dataloader:
			return batch[0].shape

	def create_custom_eval_plots_dir(self) -> Path:
		custom_eval_plots_path = self.trained_model_path / "custom_eval_plots"
		custom_eval_plots_path.mkdir(parents = True, exist_ok = True)
		return custom_eval_plots_path

	def get_loss_accuracy_curves_figure(self, show_plots = False) -> plt.Figure:
		fig = plt.figure(figsize = (14.5, 8.5))
		fig.subplots(2, 1)
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
		
		if show_plots:
			plt.show()
		return fig

	def save_eval_plots(self, figure: plt.Figure, eval_plots_path: Path):
		figure.savefig(eval_plots_path / "loss-accuracy.png")

	def save_trained_model(self, trained_model_path: Path):
		trained_model_save_path = trained_model_path / "trained_CNN_model.pth"
		print(f"Saving trained model to {trained_model_save_path}...")
		torch.save(
			obj = self.CNN_model.state_dict(),
			f = trained_model_save_path
		)

	def save_model_classification_metrics(self, trained_model_path: Path):
		trained_model_class_metrics_path = trained_model_path / "classification_metrics.json"
		print(f"Saving trained model classification metrics to {trained_model_class_metrics_path}...")
		with open(trained_model_class_metrics_path, "w") as cl_f:
			json.dump(self.final_classification_metrics, cl_f, indent = 4)

if __name__ == "__main__":
	model_preparation = CNN_Model_Preparation(
		trained_model_dirname = "training_#1"
	)
	model_preparation.main()