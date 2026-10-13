# test trained model - make an inference on test images
# using custom trained models or pretrained models from torchvision lib
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os

import CNN_one_face_detection_model as ofdm
import utils as u

class Face_Detector:
	def __init__(self, model_version: int, test_images_path: Path = None,
		cv_task: str = "one_face", trained_models_path: Path = None,
		load_pretrained_model: bool = False, test_image_from_file: bool = True):
		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.cv_task = cv_task
		self.trained_model_version = model_version
		self.trained_models_path = trained_models_path
		self.load_pretrained_model = load_pretrained_model
		self.test_image_from_file = test_image_from_file
		self.trained_model_path = self.get_trained_model_path()
		self.test_images_path = test_images_path
		self.test_images_pathes = []
		self.trained_model_weights_file = "trained_model.pth"
		self.train_config_file = "train_config.json"
		self.train_config_params = {}
		self.trained_model = object
		self.transforms_for_test_image = object

	def main(self):
		if self.trained_model_path:
			self.load_custom_trained_model()
			self.define_test_image_transforms()
		elif self.load_pretrained_model:
			pass
		else:
			print("[WARN] Model was not loaded! exitting...")
			return None

		if self.test_image_from_file:
			if self.test_images_path:
				self.test_images_pathes = list(self.test_images_path.glob("*/*.jpg"))
				test_image_path = self.get_test_image_path()
				test_image_tensor = self.get_image_tensor(test_image_path)
				self.make_inference_on_image(test_image_tensor)
		else:
			pass

	def get_trained_model_path(self) -> Path:
		if self.trained_models_path:
			model_path = (
				self.trained_models_path / self.cv_task / 
				f"training_#_{self.trained_model_version}"
			)
			return model_path
		else:
			print("[WARN] Path to manually created-trained CNN was not defined.")
			return None

	def load_custom_trained_model(self):
		if self.trained_model_weights_file in os.listdir(self.trained_model_path):
			train_config_path = self.trained_model_path / self.train_config_file
			self.train_config_params = u.get_train_config_params(train_config_path)
			
			self.trained_model = ofdm.One_Face_Detection_CNN(
				self.train_config_params["input_shape"], self.train_config_params["hidden_units"],
				self.train_config_params["output_shape"], self.train_config_params["batch_size"],
				(self.train_config_params["image_width"], self.train_config_params["image_height"])
			)
			self.trained_model.load_state_dict(
				state_dict = torch.load(
					f = self.trained_model_path / self.trained_model_weights_file
				)
			)

	def define_test_image_transforms(self):
		image_size = (
			self.train_config_params["image_width"], 
			self.train_config_params["image_height"]
		)
		self.transforms_for_test_image = v2.Compose([
			v2.Resize(size = image_size)
		])

	def get_test_image_path(self) -> Path:
		return random.sample(population = self.test_images_pathes, k = 1)[0]

	def get_image_tensor(self, image_path: Path) -> torch.Tensor:
		image_obj = Image.open(image_path)
		image_tensor = pil_to_tensor(image_obj)
		image_tensor = image_tensor.to(dtype = torch.float32)
		transformed_image_tensor = self.transforms_for_test_image(image_tensor)
		return transformed_image_tensor

	def make_inference_on_image(self, image: torch.Tensor):
		print("[INFO] Making one bounding box prediction on image...")
		pred_bbx_params = self.get_pred_bounding_box_params(image)
		image_with_bbx = self.get_image_with_pred_bounding_box(image, pred_bbx_params)
		self.show_image_with_bounding_box(image_with_bbx)

	def get_pred_bounding_box_params(self, image: torch.Tensor):
		self.trained_model.eval()
		with torch.inference_mode():
			image_batch = image.unsqueeze(dim = 0)
			inference_result = self.trained_model(image_batch)
			x, y, w, h = inference_result[0]
		return int(x.item()), int(y.item()), int(w.item()), int(h.item())

	def get_image_with_pred_bounding_box(self, image: torch.Tensor, bbx_params: tuple):
		x1 = bbx_params[0]
		y1 = bbx_params[1]
		x2 = x1 + bbx_params[2]
		y2 = y1 + bbx_params[3]
		image = image.permute(1, 2, 0).contiguous().cpu().numpy()
		image = image.astype(np.int32)
		cv2.rectangle(
		    image, (x1, y1), (x2, y2),
		    (0, 255, 0), 2
		)
		return image

	def show_image_with_bounding_box(self, image_with_bbx: torch.Tensor):
		plt.title("Image with pred face bounding box")
		plt.imshow(image_with_bbx)
		plt.show()

if __name__ == "__main__":
	test_images_path = Path("data/WIDER_sets/WIDER_test/images")
	trained_models_root_path = Path("trained_models")
	face_detector = Face_Detector(
		model_version = 2,
		test_images_path = test_images_path,
		cv_task = "one_face",
		trained_models_path = trained_models_root_path,
		load_pretrained_model = False,
		test_image_from_file = True
	)
	face_detector.main()