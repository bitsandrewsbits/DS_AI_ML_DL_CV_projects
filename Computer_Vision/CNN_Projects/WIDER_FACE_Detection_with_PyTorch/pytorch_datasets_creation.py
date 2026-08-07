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
		self.data_annotation_paths = {
			"train": self.data_root_path / "wider_face_split" / "wider_face_train_bbx_gt.txt",
			"val": self.data_root_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
		}

	def main(self):
		pass

if __name__ == "__main__":
	datasets_prep = Datasets_Praparation()
	datasets_prep.main()