# data preparation, PyTorch-datasets creation
import os
import pathlib

class Datasets_Praparation:
	def __init__(self):
		self.data_root_path = pathlib.Path("data/WIDER_sets")
		self.data_pathes = {
			"train": self.data_root_path / "WIDER_train",
			"validation": self.data_root_path / "WIDER_val",
			"test": self.data_root_path / "WIDER_test"
		}

	def main(self):
		for dataset_path in self.data_pathes.values():
			self.move_images_to_datasets_root_dir(dataset_path)

	def move_images_to_datasets_root_dir(self, root_data_path: pathlib.Path):
		for path_obj in root_data_path.walk():
			temp_dir_objects = os.listdir(path_obj[0])
			for dir_obj in temp_dir_objects:
				dir_obj_path = path_obj[0] / dir_obj

	# TODO: think, how to parce wider_face annotation.txt file with
	# bounding boxes parameters and add to image filename as target labels.
	def get_image_path(self)
		# TODO: get image path via regex
		pass

	def get_image_bounding_boxes_parameters(self):
		# TODO: get bounding boxes params - x1, y1, h, w
		# from .txt file
		pass

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation()
	datasets_prep.main()