# CNN for one-face-detection task based on TinyVGG architecture
import torch
from torch import nn

class Face_Detection_CNN(nn.Module):
	def __init__(self, input_shape, hidden_units, output_shape = 4, batch_size = 32, image_wh = (64, 64)):
		super().__init__()
		self.input_size = input_shape
		self.image_width_height = image_wh
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(
				in_channels = input_shape,
				out_channels = hidden_units,
				kernel_size = (3, 3),
				padding = 1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = hidden_units,
				out_channels = hidden_units,
				kernel_size = (3, 3),
				padding = 1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size = (2, 2),
				stride = 1
			)
		)
		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(
				in_channels = hidden_units,
				out_channels = hidden_units,
				kernel_size = (3, 3),
				padding = 1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = hidden_units,
				out_channels = hidden_units,
				kernel_size = (3, 3),
				padding = 1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size = (3, 3),
				stride = 1
			)
		)
		self.batch_size = batch_size
		self.flatter_size = self.get_input_features_size_for_linear_block()
		self.face_detection_block = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features = self.flatter_size, out_features = output_shape)
		)

	def get_input_features_size_for_linear_block(self):
		test_set = torch.zeros(
			size = (
				self.batch_size, self.input_size,
				self.image_width_height[0],
				self.image_width_height[1]
			)
		)
		conv_block_1_out = self.conv_block_1(test_set)
		conv_block_2_out = self.conv_block_2(conv_block_1_out)
		result_flatten_shape = 1
		conv_block_2_out_shape = conv_block_2_out.shape[1:]
		for one_dim_size in conv_block_2_out_shape:
			result_flatten_shape *= one_dim_size
		return result_flatten_shape

	def forward(self, set_x):
		conv_block_1_out = self.conv_block_1(set_x)
		conv_block_2_out = self.conv_block_2(conv_block_1_out)
		face_bbx_result = self.face_detection_block(conv_block_2_out)
		return face_bbx_result

if __name__ == "__main__":
	INPUT_SHAPE = 3
	HIDDEN_UNITS = 10
	OUTPUT_SHAPE = 4
	BATCH_SIZE = 10
	IMAGE_W_H = (224, 224)
	cnn_face_detect = Face_Detection_CNN(
		INPUT_SHAPE, HIDDEN_UNITS,
		OUTPUT_SHAPE, BATCH_SIZE,
		IMAGE_W_H
	)
	test_set = torch.zeros(
		size = (
			BATCH_SIZE, INPUT_SHAPE,
			IMAGE_W_H[0],
			IMAGE_W_H[1]
		)
	)
	test_result = cnn_face_detect(test_set)
	print("Test pred results:", test_result)