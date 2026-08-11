# data preparation, PyTorch-datasets creation
import os
from pathlib import Path
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

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
		self.batch_size = 2
		self.dataloaders = {
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
		train_sample = self.pytorch_datasets["train"][0]
		print("train sample shape:", train_sample[0].shape, train_sample[1])
		for dataset_type in self.dataloaders.keys():
			if dataset_type != "test":
				self.dataloaders[dataset_type] = DataLoader(
					dataset = self.pytorch_datasets[dataset_type],
					batch_size = self.batch_size,
					collate_fn = self.collate_func,
					shuffle = True
				)
			else:
				self.dataloaders[dataset_type] = DataLoader(
					dataset = self.pytorch_datasets[dataset_type],
					batch_size = self.batch_size
				)
		sample_train_batch = next(iter(self.dataloaders["train"]))
		print(len(sample_train_batch))
		print(sample_train_batch[0].shape)
		print("Target len:", len(sample_train_batch[1]))
		print("Target shape:", sample_train_batch[1][0].shape)

		# TODO: think, how to create dataloaders for each dataset.
		# TODO: create dataloaders to access them from external side(from classifier file)

	def collate_func(self, batch: tuple[torch.Tensor, list]):
		image_batch = []
		label_batch = []
		# TODO: create collate_fn()

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation()
	datasets_prep.main()