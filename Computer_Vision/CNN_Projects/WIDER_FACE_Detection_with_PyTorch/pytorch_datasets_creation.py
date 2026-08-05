# data preparation, PyTorch-datasets creation
import os
from pathlib import Path

class Datasets_Praparation:
	def __init__(self):
		self.data_root_path = Path("data/WIDER_sets")
		self.data_pathes = {
			"train": self.data_root_path / "WIDER_train" / "images",
			"val": self.data_root_path / "WIDER_val" / "images",
			"test": self.data_root_path / "WIDER_test" / "images",
		}
		self.data_annotations_root_path = self.data_root_path / "wider_face_split"
		
		# TODO: create custom dataset class from 
		# DatasetCustom output format - image_tensor_transformed, (bbx1(x1, y1, h, w), bbx2(x2, y2, h, w), ... bbxn(xm, yn, h, w))

	def main(self):
		for dataset_path in self.data_pathes.values():
			self.get_dataset_images_abs_pathes(dataset_path)

	def get_dataset_images_abs_pathes(self, root_data_path: Path):
		paths = list(root_data_path.glob("*/*.jpg"))
		return paths

	# TODO: think, how to parce wider_face annotation.txt file with
	# bounding boxes parameters and add to image filename as target labels.
	def get_image_path(self):
		# TODO: get image path via regex
		pass

	def get_image_bounding_boxes_parameters(self):
		# TODO: get bounding boxes params - x1, y1, h, w
		# from .txt file
		pass

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation()
	datasets_prep.main()