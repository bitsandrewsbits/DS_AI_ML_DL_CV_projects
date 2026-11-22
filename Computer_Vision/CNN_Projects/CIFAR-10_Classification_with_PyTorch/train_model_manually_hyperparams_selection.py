# script for running model training process manually
import argparse

import model_preparation as mp

def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"-tmd", "--trained_model_dirname",
		help = "Dirname for trained model",
		required = True
	)
	arg_parser.add_argument(
		"-mfeb", "--model_features_extract_blocks",
		help = "Features extraction blocks in CNN model.",
		required = True
	)
	arg_parser.add_argument(
		"-ch", "--color_channels",
		help = "Color channels",
		required = True
	)
	arg_parser.add_argument(
		"-hus", "--hidden_units",
		help = "Hidden units amount",
		required = True
	)
	arg_parser.add_argument(
		"-bsz", "--batch_size",
		help = "Image batch size",
		required = True
	)
	arg_parser.add_argument(
		"-lr", "--learning_rate",
		help = "Learning rate",
		required = True
	)
	arg_parser.add_argument(
		"-eps", "--epochs",
		help = "Epochs amount",
		required = True
	)

	received_args = arg_parser.parse_args()
	trained_model_dirname = received_args.trained_model_dirname
	model_features_extract_blocks = int(received_args.model_features_extract_blocks)
	color_channels = int(received_args.color_channels)
	hidden_units_amount = int(received_args.hidden_units)
	batch_size = int(received_args.batch_size)
	learning_rate = float(received_args.learning_rate)
	train_epochs = int(received_args.epochs)

	train_model = mp.CNN_Model_Preparation(
		trained_model_dirname = trained_model_dirname,
		model_features_extract_blocks = model_features_extract_blocks,
		color_channels = color_channels,
		hidden_units = hidden_units_amount,
		batch_size = batch_size,
		learning_rate = learning_rate,
		epochs = train_epochs
	)
	train_model.main()

if __name__ == "__main__":
	main()