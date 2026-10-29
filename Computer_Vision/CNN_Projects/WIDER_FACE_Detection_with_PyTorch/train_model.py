# main file for train model for face(s) detection task.
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import pytorch_datasets_creation as pdc
import CNN_one_face_detection_model as cofdm
import training_loop as tl
import utils

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

	ROOT_TRAINED_MODELS_DIRNAME = "trained_models"
	root_trained_models_dir_path = Path(ROOT_TRAINED_MODELS_DIRNAME)
	save_trained_model_path = utils.get_trained_model_dir(
		trained_models_root_path = root_trained_models_dir_path,
		cv_task = cv_task
	)
	
	if cv_task == "one_face":
		INPUT_SHAPE = 3
		OUTPUT_SHAPE = 4
		face_detect_model = cofdm.One_Face_Detection_CNN(
			input_shape = INPUT_SHAPE,
			hidden_units = hidden_units,
			output_shape = OUTPUT_SHAPE,
			batch_size = batch_size,
			image_wh = (image_width, image_height)
		)

		model_train = tl.Model_Training(
			save_trained_model_path = save_trained_model_path,
			model_obj = face_detect_model,
			dataloaders = face_detect_dataloaders,
			learning_rate = learning_rate,
			epochs = epochs
		)

	model_train.train_model()

	loss_fig = utils.get_loss_curves_figure(
		model_train.epochs_nums,
		{
			"train": model_train.train_losses,
			"eval": model_train.eval_losses
		}
	)

	MAE_fig = utils.get_MAE_curves_figure(
		model_train.epochs_nums,
		{
			"train": model_train.train_MAEs,
			"eval": model_train.eval_MAEs
		}
	)
	
	train_params = {
		"cv_task": cv_task, "batch_size": batch_size,
		"image_width": image_width, "image_height": image_height,
		"train_size": train_size, "valid_size": valid_size,
		"test_size": test_size, "learning_rate": learning_rate,
		"input_shape": INPUT_SHAPE, "output_shape": OUTPUT_SHAPE,
		"hidden_units": hidden_units, "epochs": epochs
	}

	utils.save_train_parameters(
		parameters = train_params,
		train_config_path = save_trained_model_path
	)

	CUSTOM_PLOTS_DIRNAME = "custom_plots"
	utils.save_loss_curves_plot(
		figure = loss_fig,
		trained_model_dir_path = save_trained_model_path,
		custom_plots_dirname = CUSTOM_PLOTS_DIRNAME
	)
	utils.save_MAE_curves_plot(
		figure = MAE_fig,
		trained_model_dir_path = save_trained_model_path,
		custom_plots_dirname = CUSTOM_PLOTS_DIRNAME
	)
	
	utils.save_model_weights(
		trained_model_dir_path = save_trained_model_path,
		model = model_train.face_detect_model
	)

	ROTATION_DIRS_AMOUNT = 5
	STORED_TRAINED_MODELS_AMOUNT = 10
	utils.remove_old_trained_models(
		trained_models_root_path = root_trained_models_dir_path,
		cv_task = cv_task,
		latest_trained_models_amount = STORED_TRAINED_MODELS_AMOUNT,
		rotation_dirs_amount = ROTATION_DIRS_AMOUNT
	)

if __name__ == "__main__":
	main()