# test trained model - make an inference on test images
# using custom trained models or pretrained models from torchvision lib
import torch
from pathlib import Path

class Face_Detector:
	def __init__(self, trained_models_path: Path = None,
	model_version: int, test_images_dir: str):
		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.trained_models_path = trained_models_path
		self.trained_model_version = model_version
		self.test_images_dir = test_images_dir
		self.trained_model = object

	def load_model(self):
		if self.trained_models_path:
			pass