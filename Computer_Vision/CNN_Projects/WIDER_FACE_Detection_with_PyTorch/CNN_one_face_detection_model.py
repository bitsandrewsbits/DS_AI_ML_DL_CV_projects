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
		for one_dim_size in conv_block_2_out.shape:
			result_flatten_shape *= one_dim_size
		return result_flatten_shape

if __name__ == "__main__":
	cnn_face_detect = Face_Detection_CNN(3, 10, 4, 32, (224, 224))
