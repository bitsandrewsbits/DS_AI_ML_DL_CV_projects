# model ML flow process - model loading prepared data, training, evaluating, tuning
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall
import numpy as np
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

import CNN_model_architecture as cnnma
import global_variables as gv

class CNN_Model_Preparation:
	def __init__(self, trained_model_dirname: str, model_features_extract_blocks: int,
	color_channels: int, hidden_units: int, batch_size: int, learning_rate: float, epochs: int):
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
		self.batch_size = batch_size
		self.train_dataloader = DataLoader(
			dataset = self.train_set,
			batch_size = self.batch_size,
			shuffle = True
		)
		self.test_dataloader = DataLoader(
			dataset = self.test_set,
			batch_size = self.batch_size
		)
		self.color_channels = color_channels
		self.hidden_units = hidden_units
		self.model_features_extraction_blocks = model_features_extract_blocks
		self.classification_classes = self.train_set.classes
		self.classes_amount = len(self.classification_classes)
		self.image_shape = self.train_set[0][0].shape
		self.batch_shape = self.get_batch_shape()

		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.CNN_model = cnnma.TinyVGG_Architecture_CNN(
			self.color_channels, self.hidden_units,
			self.classes_amount, self.batch_shape,
			self.model_features_extraction_blocks
		).to(self.compute_device)

		self.loss_func = torch.nn.CrossEntropyLoss()
		self.learning_rate = learning_rate
		self.optimizer = SGD(
			params = self.CNN_model.parameters(), lr = self.learning_rate
		)
		self.accuracy_func = Accuracy(
			task = "multiclass", num_classes = self.classes_amount
		).to(self.compute_device)
		self.precision_func = Precision(
			task = "multiclass", num_classes = self.classes_amount
		).to(self.compute_device)
		self.recall_func = Recall(
			task = "multiclass", num_classes = self.classes_amount
		).to(self.compute_device)
		self.epochs = epochs

		self.train_metrics = {
			"train_loss": [], "valid_loss": [],
			"train_accuracy": [], "valid_accuracy": [],
			"train_precision": [], "valid_precision": [],
			"train_recall": [], "valid_recall": []
		}
		self.final_classification_metrics = {
			"loss": 0.0,
			"accuracy": 0.0,
			"precision": 0.0,
			"recall": 0.0
		}

		self.trained_models_root_path = Path(gv.TRAINED_MODELS_DIR)
		self.trained_model_dir = trained_model_dirname
		self.trained_model_path = self.create_dirs_for_trained_model()
		self.log_writer = SummaryWriter(
			log_dir = self.trained_model_path / "train_logs"
		)

	def main(self):
		self.train_model()
		eval_plots_path = self.create_custom_eval_plots_dir()
		eval_figure = self.get_loss_accuracy_curves_figure()
		self.save_eval_plots(eval_figure, eval_plots_path)
		self.define_final_eval_classification_metrics()
		self.save_trained_model()
		model_train_config = self.get_model_train_config()
		self.save_model_train_config(model_train_config)
		self.save_model_classification_metrics()

	def create_dirs_for_trained_model(self) -> Path:
		self.trained_models_root_path.mkdir(parents = True, exist_ok = True)
		trained_model_path = self.trained_models_root_path / self.trained_model_dir
		trained_model_path.mkdir(parents = True, exist_ok = True)
		return trained_model_path

	def train_model(self):
		for epoch in range(1, self.epochs + 1):
			print(f"Epoch #{epoch}:", end = '')
			train_loss, train_accur_precis_recall = self.train_step()
			valid_loss, valid_accur_precis_recall = self.evaluate_model()

			self.update_train_epoch_metrics(
				losses = [train_loss, valid_loss],
				train_accuracy_precision_recall = train_accur_precis_recall,
				valid_accuracy_precision_recall = valid_accur_precis_recall
			)

			self.log_writer.add_scalars(
				main_tag = "Loss",
				tag_scalar_dict = {
					"train": train_loss,
					"valid": valid_loss
				},
				global_step = epoch
			)
			self.log_writer.add_scalars(
				main_tag = "Accuracy",
				tag_scalar_dict = {
					"train": train_accur_precis_recall[0],
					"valid": valid_accur_precis_recall[0]
				},
				global_step = epoch
			)
			self.log_writer.add_scalars(
				main_tag = "Precision",
				tag_scalar_dict = {
					"train": train_accur_precis_recall[1],
					"valid": valid_accur_precis_recall[1]
				}
			)
			self.log_writer.add_scalars(
				main_tag = "Recall",
				tag_scalar_dict = {
					"train": train_accur_precis_recall[2],
					"valid": valid_accur_precis_recall[2]
				}
			)
		self.log_writer.close()

	def train_step(self):
		epoch_sum_loss = 0
		epoch_accur_precis_recall_sums = np.array([0.0, 0.0, 0.0])
		self.CNN_model.train()
		for (images_batch, labels_batch) in self.train_dataloader:
			images_batch = images_batch.to(self.compute_device)
			labels_batch = labels_batch.to(self.compute_device)
			logits = self.CNN_model(images_batch)
			pred_probs = torch.softmax(logits, dim = 1)
			pred_labels = torch.argmax(pred_probs, dim = 1)

			batch_loss = self.loss_func(logits, labels_batch)
			batch_accur_precis_recall = self.get_batch_accur_precis_recall(
				pred_labels, labels_batch
			)

			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
			
			epoch_sum_loss += batch_loss
			epoch_accur_precis_recall_sums += batch_accur_precis_recall

		epoch_loss = round(epoch_sum_loss.item() / len(self.train_dataloader), 3)
		epoch_accur_precis_recall = epoch_accur_precis_recall_sums / len(self.train_dataloader)
		epoch_accuracy = round(epoch_accur_precis_recall[0], 3)
		print(f"train loss: {epoch_loss}, train accuracy: {epoch_accuracy}", end = '')
		return epoch_loss, epoch_accur_precis_recall

	def evaluate_model(self) -> tuple[float, float]:
		eval_sum_loss = 0
		eval_accur_precis_recall_sums = np.array([0.0, 0.0, 0.0])
		self.CNN_model.eval()
		with torch.inference_mode():
			for (images_batch, labels_batch) in self.test_dataloader:
				images_batch = images_batch.to(self.compute_device)
				labels_batch = labels_batch.to(self.compute_device)
				logits = self.CNN_model(images_batch)
				pred_probs = torch.softmax(logits, dim = 1)
				pred_labels = torch.argmax(pred_probs, dim = 1)

				batch_eval_loss = self.loss_func(logits, labels_batch)
				batch_accur_precis_recall = self.get_batch_accur_precis_recall(
					pred_labels, labels_batch
				)
				eval_sum_loss += batch_eval_loss
				eval_accur_precis_recall_sums += batch_accur_precis_recall

			eval_loss = round(eval_sum_loss.item() / len(self.test_dataloader), 3)
			eval_accur_precis_recall = eval_accur_precis_recall_sums / len(self.test_dataloader)
			eval_accuracy = round(eval_accur_precis_recall[0], 3)
		print(f"eval loss: {eval_loss}, eval accuracy: {eval_accuracy}")
		return eval_loss, eval_accur_precis_recall

	def get_batch_accur_precis_recall(self, pred_labels: torch.Tensor, labels_batch: torch.Tensor):
		batch_accur_precis_recall = np.array([
			self.accuracy_func(pred_labels, labels_batch).cpu(),
			self.precision_func(pred_labels, labels_batch).cpu(),
			self.recall_func(pred_labels, labels_batch).cpu()
		])
		return batch_accur_precis_recall

	def update_train_epoch_metrics(self, losses: list,
	train_accuracy_precision_recall: np.array, valid_accuracy_precision_recall: np.array):
		self.train_metrics["train_loss"].append(losses[0])
		self.train_metrics["valid_loss"].append(losses[1])
		self.train_metrics["train_accuracy"].append(train_accuracy_precision_recall[0])
		self.train_metrics["train_precision"].append(train_accuracy_precision_recall[1])
		self.train_metrics["train_recall"].append(train_accuracy_precision_recall[2])
		self.train_metrics["valid_accuracy"].append(valid_accuracy_precision_recall[0])
		self.train_metrics["valid_precision"].append(valid_accuracy_precision_recall[1])
		self.train_metrics["valid_recall"].append(valid_accuracy_precision_recall[2])

	def define_final_eval_classification_metrics(self):
		print('-' * 40)
		print("Final evaluation loss/accuray/precision/recall:")
		final_valid_loss, final_valid_accur_precis_recall = self.evaluate_model()
		self.final_classification_metrics["loss"] = final_valid_loss
		self.final_classification_metrics["accuracy"] = final_valid_accur_precis_recall[0]
		self.final_classification_metrics["precision"] = final_valid_accur_precis_recall[1]
		self.final_classification_metrics["recall"] = final_valid_accur_precis_recall[2]
		print(f"precision: {final_valid_accur_precis_recall[1]}, recall: {final_valid_accur_precis_recall[2]}")
		print('-' * 40)

	def get_batch_shape(self):
		for batch in self.train_dataloader:
			return batch[0].shape

	def create_custom_eval_plots_dir(self) -> Path:
		custom_eval_plots_path = self.trained_model_path / "custom_eval_plots"
		custom_eval_plots_path.mkdir(parents = True, exist_ok = True)
		return custom_eval_plots_path

	def get_loss_accuracy_curves_figure(self, show_plots = False) -> plt.Figure:
		fig = plt.figure(figsize = (14.5, 8.5))
		fig.subplots(2, 1)
		epochs = [i for i in range(1, len(self.train_metrics["train_loss"]) + 1)]

		plt.subplot(2, 1, 1)
		plt.title("Train/Valid Loss")
		plt.plot(epochs, self.train_metrics["train_loss"], label = 'train-loss')
		plt.plot(epochs, self.train_metrics["valid_loss"], label = 'valid-loss')
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.legend()

		plt.subplot(2, 1, 2)
		plt.title("Train/Valid Accuracy")
		plt.plot(epochs, self.train_metrics["train_accuracy"], label = 'train-accuracy')
		plt.plot(epochs, self.train_metrics["valid_accuracy"], label = 'valid-accuracy')
		plt.xlabel("epoch")
		plt.ylabel("accuracy")
		plt.legend()
		
		if show_plots:
			plt.show()
		return fig

	def get_model_train_config(self) -> dict:
		model_train_config = {
			"model_features_extract_blocks": self.model_features_extraction_blocks,
			"color_channels": self.color_channels,
			"hidden_units": self.hidden_units,
			"batch_size": self.batch_size,
			"learning_rate": self.learning_rate,
			"epochs": self.epochs
		}
		return model_train_config

	def save_model_train_config(self, config: dict):
		train_model_config_save_path = self.trained_model_path / "model_train_config.json"
		print(f"Saving model/train hyperparameters to {train_model_config_save_path}...")
		with open(train_model_config_save_path, "w") as mtc_f:
			json.dump(config, mtc_f, indent = 4)

	def save_eval_plots(self, figure: plt.Figure, eval_plots_path: Path):
		figure.savefig(eval_plots_path / "loss-accuracy.png")

	def save_trained_model(self):
		trained_model_save_path = self.trained_model_path / "trained_CNN_model.pth"
		print(f"Saving trained model to {trained_model_save_path}...")
		torch.save(
			obj = self.CNN_model.state_dict(),
			f = trained_model_save_path
		)

	def save_model_classification_metrics(self):
		trained_model_class_metrics_path = self.trained_model_path / "classification_metrics.json"
		print(f"Saving trained model classification metrics to {trained_model_class_metrics_path}...")
		with open(trained_model_class_metrics_path, "w") as cl_f:
			json.dump(self.final_classification_metrics, cl_f, indent = 4)

if __name__ == "__main__":
	model_preparation = CNN_Model_Preparation(
		trained_model_dirname = "training_#2",
		model_features_extract_blocks = 1,
		color_channels = 3,
		hidden_units = 25,
		batch_size = 32,
		learning_rate = 0.01,
		epochs = 3
	)
	model_preparation.main()