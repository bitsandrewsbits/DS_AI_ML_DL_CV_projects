# data preparation, PyTorch-datasets creation
import os
from pathlib import Path
import torch
from torchvision.transforms import v2

from face_detection_dataset import Wider_Face_Detection_Dataset

class Datasets_Praparation:
	def __init__(self):
		self.data_root_path = Path("data/WIDER_sets")
		self.data_pathes = {
			"train": self.data_root_path / "WIDER_train" / "images",
			"val": self.data_root_path / "WIDER_val" / "images",
			"test": self.data_root_path / "WIDER_test" / "images",
		}
		self.data_annotation_paths = {
			"train": self.data_root_path / "wider_face_split" / "wider_face_train_bbx_gt.txt",
			"val": self.data_root_path / "wider_face_split" / "wider_face_val_bbx_gt.txt",
			"test": None
		}
		self.image_transforms = v2.Compose([
			v2.ToImage(),
			v2.Resize(size = (224, 224)),
			v2.RandomHorizontalFlip(p = 0.5),
			v2.ToDtype(torch.float32, scale = True)
		])
		self.pytorch_datasets = {
			"train": '',
			"val": '',
			"test": ''
		}

	def main(self):
		for dataset_type, data_path in self.data_pathes.items():
			self.pytorch_datasets[dataset_type] = Wider_Face_Detection_Dataset(
				data_path, self.data_annotation_paths[dataset_type],
				self.image_transforms
			)
		# TODO: think, how to create dataloaders for each dataset.
		# TODO: create dataloaders to access them from external side(from classifier file)

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation()
	datasets_prep.main()