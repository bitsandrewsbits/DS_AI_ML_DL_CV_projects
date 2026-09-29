# test trained model - make an inference on test images
# using custom trained models or pretrained models from torchvision lib
import torch
from pathlib import Path

class Face_Detector:
	def __init__(self, trained_models_path: Path = None,
	model_version: int, test_images_dir: str, cv_task: str):
		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.trained_models_path = trained_models_path
		self.trained_model_version = model_version
		self.test_images_dir = test_images_dir
		self.cv_task = cv_task
		self.trained_model = object

	def make_inference_on_image(self, image: torch.Tensor):
		pred_bbx_params = self.get_pred_bounding_box_params(image)
		image_with_bbx = self.get_image_with_pred_bounding_box(image, pred_bbx_params)
		self.show_image_with_bounding_box(image_with_bbx)

	def get_pred_bounding_box_params(self, image: torch.Tensor):
		self.face_detect_model.eval()
		with torch.inference_mode():
			image_batch = image.unsqueeze(dim = 0)
			inference_result = self.face_detect_model(image_batch)
			x, y, w, h = inference_result[0]
		return int(x.item()), int(y.item()), int(w.item()), int(h.item())

	def get_image_with_pred_bounding_box(self, image: torch.Tensor, bbx_params: tuple):
		x1 = bbx_params[0]
		y1 = bbx_params[1]
		x2 = x1 + bbx_params[2]
		y2 = y1 + bbx_params[3]
		image = image.permute(1, 2, 0).contiguous().cpu().numpy()
		cv2.rectangle(
		    image, (x1, y1), (x2, y2),
		    (0, 255, 0), 2
		)
		return image

	def show_image_with_bounding_box(self, image_with_bbx: torch.Tensor):
		plt.title("Image with pred face bounding box")
		plt.imshow(image_with_bbx)
		plt.show()