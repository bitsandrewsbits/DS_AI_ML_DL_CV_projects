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
		help = "Select computer vision detection task.",
		required = True
	)
	input_args.add_argument(
		"-imw", "--image_width",
		help = "Enter an image width in px.",
		required = True
	)
	input_args.add_argument(
		"-imh", "--image_height",
		help = "Enter an image height in px.",
		required = True
	)
	input_args.add_argument(
		"-tss", "--train_set_size",
		help = "Enter a size of train set."
	)
	input_args.add_argument(
		"-vss", "--validation_set_size",
		help = "Enter a size of validation set."
	)
	input_args.add_argument(
		"-ttss", "--test_set_size",
		help = "Enter a size of test set."
	)
	input_args.add_argument(
		"-bsz", "--batch_size",
		help = "Enter an image batch size.",
		required = True
	)
	input_args.add_argument(
		"-lr", "--learning-rate",
		help = "Enter a learning rate for optimizer.",
		required = True
	)
	input_args.add_argument(
		"-eps", "--epochs",
		help = "Enter a train epochs amount.",
		required = True
	)
	input_args.add_argument(
		"-hus", "--hidden_units",
		help = "Enter a hidden units amount in hidden layer.",
		required = True
	)
	
	received_args = input_args.parse_args()
	cv_task = received_args.cv_task
	
	batch_size = int(received_args.batch_size)
	image_width = int(received_args.image_width)
	image_height = int(received_args.image_height)

	if received_args.train_set_size:
		train_size = int(received_args.train_set_size)
	else:
		train_size = "all"
	if received_args.validation_set_size:
		valid_size = int(received_args.validation_set_size)
	else:
		valid_size = "all"
	if received_args.test_set_size:
		test_size = int(received_args.test_set_size)
	else:
		test_size = "all"

	learning_rate = float(received_args.learning_rate)
	hidden_units = int(received_args.hidden_units)
	
	epochs = int(received_args.epochs)

	datasets_prep = pdc.Datasets_Praparation(
		image_size = (image_width, image_height),
		batch_size = batch_size,
		task = cv_task,
		datasets_sizes = {
			"train": train_size,
			"val": valid_size,
			"test": test_size
		}
	)
	datasets_prep.main()
	face_detect_dataloaders = datasets_prep.dataloaders

	if cv_task == "one_face":
		face_detect_model = cofdm.One_Face_Detection_CNN(
			input_shape = 3,
			hidden_units = hidden_units,
			output_shape = 4,
			batch_size = batch_size,
			image_wh = (image_width, image_height)
		)

		model_train = tl.Model_Training(
			model_obj = face_detect_model,
			dataloaders = face_detect_dataloaders,
			learning_rate = learning_rate,
			epochs = epochs
		)
	
	model_train.train_model()

main()