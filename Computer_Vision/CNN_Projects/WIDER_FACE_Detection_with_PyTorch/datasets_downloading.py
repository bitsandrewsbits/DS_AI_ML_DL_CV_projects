# downloading WIDER FACE train/val/test datasets
import requests
import zipfile
import os
from pathlib import Path

def main(datasets_info: dict):
	datasets_path = create_datasets_dirs()
	for dataset_type, url in datasets_info.items():
		if f"WIDER_{dataset_type}.zip" in os.listdir(datasets_path):
			print(f"Dataset WIDER_{dataset_type}.zip already downloaded! skipping.")
		else:
			download_dataset_zip(datasets_path, dataset_type, url)

		if f"WIDER_{dataset_type}" in os.listdir(datasets_path):
			print(f"Dataset WIDER_{dataset_type} already extracted! skipping.")
		else:
			unzip_dataset(datasets_path, dataset_type)

def create_datasets_dirs() -> Path:
	root_path = Path("data")
	datasets_path = root_path / "WIDER_sets"
	if root_path.is_dir():
		print("Datasets dir already exist!")
	else:
		print("Creating datasets dir...")
		datasets_path.mkdir(parents = True, exist_ok = True)
	return datasets_path

def download_dataset_zip(datasets_path: Path, dataset_type: str, url: str):
	print(f"Downloading {dataset_type} WIDER dataset...")
	with open(datasets_path / f"WIDER_{dataset_type}.zip", "wb") as f:
		response = requests.get(url)
		f.write(response.content)

def unzip_dataset(datasets_path: Path, dataset_type: str):
	print("Extracting data...")
	with zipfile.ZipFile(datasets_path / f"WIDER_{dataset_type}.zip", "r") as r_f:
		r_f.extractall(datasets_path)

if __name__ == "__main__":
	datasets_URLs = {
		"train": "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip",
		"val": "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip",
		"test": "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_test.zip",
		"annotation": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
	}
	main(datasets_URLs)