# dataset class for face detection task
# target - tuple of bounding boxes(bbx) tuples 
# targets - in wider face annotation file
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import re

class Wider_Face_Detection_Dataset(Dataset):
	def __init__(self, target_path: Path, dataset_annotation_path: Path, transform = None):
		self.paths = list(target_path.glob("*/*.jpg"))
		self.dataset_annotation_path = dataset_annotation_path
		self.dataset_annotation_text = self.get_annotation_file_text()
		self.splitted_annotation_lines = self.get_splitted_annotation_by_new_line()
		self.path_to_bbx_info = {}

	def load_image(self, index: int):
		image_path = self.paths[index]
		return Image.open(image_path)

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[tuple]]:
		pass

	# TODO: think, how to parce wider_face annotation.txt file with
	# bounding boxes parameters and add to image filename as target labels.
	def define_path_to_bbx_dataset_info(self):
		self.path_to_bbx_info["path"] = path
		self.path_to_bbx_info["bbx"] = self.get_image_bbxs(path)

	def get_annotation_file_text(self):
		with open(self.dataset_annotation_path, "rt") as ant_f:
			return ant_f.read()
	
	def get_image_filename(self, path: Path):
		print("Image path:", path)
		image_filename = path.split('/')[-1]
		print("image filename:", image_filename)
		return image_filename

	def get_splitted_annotation_by_new_line(self):
		return self.dataset_annotation_text.split('\n')

	def get_image_bbxs(self, path: Path):
		# from annotation lines
		pass

if __name__ == "__main__":
	face_detection_dataset = Wider_Face_Detection_Dataset(
		Path("data/WIDER_sets/WIDER_val/images"),
		Path("data/WIDER_sets/wider_face_split/wider_face_val_bbx_gt.txt")
	)