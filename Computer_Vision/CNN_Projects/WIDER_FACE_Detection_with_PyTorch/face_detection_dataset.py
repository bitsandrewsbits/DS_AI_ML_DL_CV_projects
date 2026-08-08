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
		print(type(self.dataset_annotation_text))
		self.path_to_bbx_info = {}

	def load_image(self, index: int):
		image_path = self.paths[index]
		return Image.open(image_path)

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[tuple]]:
		# TODO: create method - add transform obj
		# return image tensor, [bbx1[x1, y1, w, h], ..., bbxn[x1, y1, w, h]] as target
		pass

	def define_path_to_bbx_dataset_info(self):
		self.path_to_bbx_info["path"] = path
		self.path_to_bbx_info["bbx"] = self.get_image_bbxs(path)

	def get_annotation_file_text(self):
		with open(self.dataset_annotation_path, "rt") as ant_f:
			return ant_f.read()

	def get_image_bbxs(self, path: Path):
		image_bbxs_text = self.get_image_bbx_text(path)
		splitted_bbxs_text_by_lines = image_bbxs_text.split('\n')[:-1]
		bbxs_x1_y1_w_h = []
		for bbx_line in splitted_bbxs_text_by_lines:
			bbx_parameters = bbx_line.split(' ')
			bbx_x1_y1_w_h = [int(param) for param in bbx_parameters[:4]]
			bbxs_x1_y1_w_h.append(bbx_x1_y1_w_h)
		return bbxs_x1_y1_w_h

	def get_image_bbx_text(self, path: Path):
		image_filename = self.get_image_filename_without_extension(path)
		image_bbxs_lines_start_end_regex = rf"({image_filename}\.jpg\n.*?\n)([0-9 \n]*?)([0-9]{1,}?--.*?\.jpg\n)"
		find_result = re.findall(image_bbxs_lines_start_end_regex, self.dataset_annotation_text)
		return find_result[0][1:-1][0]

	def get_image_filename_without_extension(self, path: Path):
		image_filename = str(path).split('/')[-1]
		print("Target file:", image_filename)
		return image_filename.split('.')[0]

if __name__ == "__main__":
	face_detection_dataset = Wider_Face_Detection_Dataset(
		Path("data/WIDER_sets/WIDER_val/images"),
		Path("data/WIDER_sets/wider_face_split/wider_face_val_bbx_gt.txt")
	)
	face_detection_dataset.define_path_to_bbx_dataset_info()