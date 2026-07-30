# program for CIFAR-10 image classification via CNN model
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path
import matplotlib.pyplot as plt
import random

import CNN_model_architecture as cnnma
import global_variables as gv

class Images_Classification:
	def __init__(self, trained_model_name: str):
		self.datasets_dir = gv.DATASETS_DIR
		self.trained_models_dir = gv.TRAINED_MODELS_DIR
		self.trained_model_name = trained_model_name
		self.trained_models_path = Path(self.trained_models_dir)
		self.trained_model_path = self.trained_models_path / self.trained_model_name

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
		self.test_samples = self.get_random_samples()
		self.test_samples_evaluation = {
			"model_name": self.trained_model_name,
			"eval_results": []
		}

		self.subplots_amount = gv.EVALUATION_SAMPLES_AMOUNT
		self.subplots_rows = gv.EVALUATION_SUBPLOTS_ROWS
		self.subplots_columns = self.subplots_amount // self.subplots_rows

	def main(self):
		print("Loading model weights...")
		self.load_model_weights()
		self.model_with_architecture.to(self.compute_device)
		self.update_samples_eval_data()
		self.make_sample_evaluation()
		self.show_test_samples_evaluation()

	def load_model_weights(self):
		self.model_with_architecture.load_state_dict(
			torch.load(f = self.trained_model_path)
		)

	def get_random_samples(self):
		return random.sample(
			list(self.test_dataset),
			k = gv.EVALUATION_SAMPLES_AMOUNT
		)

	def update_samples_eval_data(self):
		for (image, true_label) in self.test_samples:
			sample_eval_results = {
				"image": image,
				"true_label": true_label,
				"true_class": self.classes_names[true_label],
				"predict_label": '',
				"predict_class": ''
			}
			self.test_samples_evaluation["eval_results"].append(
				sample_eval_results
			)

	def make_sample_evaluation(self):
		print("Making evaluation for test samples...")
		for sample_info in self.test_samples_evaluation["eval_results"]:
			sample_info["predict_label"] = self.get_predicted_label(
				sample_info["image"]
			)
			sample_info["predict_class"] = self.classes_names[sample_info["predict_label"]]

	def get_predicted_label(self, image: torch.Tensor):
		self.model_with_architecture.eval()
		with torch.inference_mode():
			image_as_batch = image.unsqueeze(dim = 0)
			image_as_batch = image_as_batch.to(self.compute_device)
			pred_logits = self.model_with_architecture(image_as_batch)
			pred_probs = torch.softmax(pred_logits, dim = 1)
			pred_label = torch.argmax(pred_probs, dim = 1)
			return pred_label.item()

	def show_test_samples_evaluation(self):
		fig, ax = plt.subplots(self.subplots_rows, self.subplots_columns)
		for (i, sample_info) in enumerate(self.test_samples_evaluation["eval_results"], 1):
			plt.subplot(self.subplots_rows, self.subplots_columns, i)
			if sample_info["true_class"] == sample_info["predict_class"]:
				plt.title(
					f"True: {sample_info["true_class"]}/"
					f"Pred: {sample_info["predict_class"]}",
					fontsize = 9, c = "green"
				)
			else:
				plt.title(
					f"True: {sample_info["true_class"]}/"
					f"Pred: {sample_info["predict_class"]}",
					fontsize = 9, c = "red"
				)
			plt.imshow(torch.permute(sample_info["image"], (1, 2, 0)))
			plt.axis(False)
		plt.show()

if __name__ == "__main__":
	images_classification = Images_Classification("cnn_model.pth")
	images_classification.main()