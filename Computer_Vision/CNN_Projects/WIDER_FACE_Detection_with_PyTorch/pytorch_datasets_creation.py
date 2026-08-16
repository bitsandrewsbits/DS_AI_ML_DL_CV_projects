# data preparation, PyTorch-datasets creation
import os
from pathlib import Path
import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import DataLoader

from face_detection_dataset import Wider_Face_Detection_Dataset

class Datasets_Praparation:
	def __init__(self, image_size = (64, 64), batch_size = 32, task = "one_face"):
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
			v2.Resize(size = image_size),
			v2.RandomHorizontalFlip(p = 0.5),
			v2.ToDtype(torch.float32, scale = True)
		])
		self.pytorch_datasets = {
			"train": '',
			"val": '',
			"test": ''
		}
		self.batch_size = batch_size
		self.dataloaders = {
			"train": '',
			"val": '',
			"test": ''
		}
		self.detection_task = task

	def main(self):
		for dataset_type, data_path in self.data_pathes.items():
			self.pytorch_datasets[dataset_type] = Wider_Face_Detection_Dataset(
				data_path, self.data_annotation_paths[dataset_type],
				self.image_transforms
			)
		train_sample = self.pytorch_datasets["train"][0]
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

	def collate_func(self, batch: tuple[torch.Tensor, BoundingBoxes]):
		image_batch = []
		label_batch = []
		for image, bbxs_obj in batch:
			image_batch.append(image)
			label_batch.append(bbxs_obj)
		return torch.stack(image_batch), torch.stack(label_batch)

	def inspect_train_dataloader_struct(self):
		sample_train_batch = next(iter(self.dataloaders["train"]))
		print("Batch amount in train-loader:", len(self.dataloaders["train"]))
		print(sample_train_batch[0])
		print(sample_train_batch[1])

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation(
		image_size = (224, 224),
		batch_size = 2,
		task = "one_face"
	)
	datasets_prep.main()
	datasets_prep.inspect_train_dataloader_struct()