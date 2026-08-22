# main file for train model for face(s) detection task.
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import argparse

import pytorch_datasets_creation as pdc
import CNN_one_face_detection_model as cofdm
import training_loop as tl

def main():
	input_args = argparse.ArgumentParser()
	input_args.add_argument(
		"-t", "--cv_task",
		choices = ["one_face", "all_image_faces"],
		help = "Select computer vision detection task."
	)
	input_args.add_argument(
		"-imw", "--image_width",
		help = "Enter an image width in px."
	)
	input_args.add_argument(
		"-imh", "--image_height",
		help = "Enter an image height in px."
	)
	input_args.add_argument(
		"-bsz", "--batch_size",
		help = "Enter an image batch size."
	)
	input_args.add_argument(
		"-lr", "--learning-rate",
		help = "Enter a learning rate for optimizer."
	)
	input_args.add_argument(
		"-eps", "--epochs",
		help = "Enter a train epochs amount."
	)
	input_args.add_argument(
		"-hus", "--hidden_units",
		help = "Enter a hidden units amount in hidden layer."
	)
	
	received_args = input_args.parse_args()

main()