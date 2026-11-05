# CNN for one-face-detection task based on TinyVGG architecture
import torch
from torch import nn

class One_Face_Detection_CNN(nn.Module):
	def __init__(self, input_shape, hidden_units, output_shape = 4, batch_size = 32, image_wh = (64, 64)):
		super().__init__()
		self.input_size = input_shape
		self.image_width_height = image_wh
		self.conv_block_1 = self.get_Conv_MaxPool_block(
			conv_layers = 3, input_shape = input_shape,
			hidden_units = hidden_units, conv_kernel_size = 4,
			max_pool_kernel_size = 2, conv_padding = 1,
			max_pool_stride = 1, first_block = True
		)
		self.conv_block_2 = self.get_Conv_MaxPool_block(
			conv_layers = 2, input_shape = input_shape,
			hidden_units = hidden_units, conv_kernel_size = 5,
			max_pool_kernel_size = 2, conv_padding = 1,
			max_pool_stride = 1, first_block = False
		)
		self.batch_size = batch_size
		self.flatter_size = self.get_input_features_size_for_linear_block()
		self.face_detection_block = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features = self.flatter_size, out_features = 256),
			nn.Linear(in_features = 256, out_features = 128),
			nn.Dropout(0.3),
			nn.Linear(in_features = 128, out_features = output_shape)
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

	def get_Conv_MaxPool_block(self, conv_layers: int, input_shape: int,
	hidden_units: int, conv_kernel_size: int, max_pool_kernel_size: int,
	conv_padding: int, max_pool_stride: int, first_block = False) -> nn.Sequential:
		layers_sequence = self.get_layers_sequence_in_block(conv_layers)
		conv_pool_block = nn.Sequential()
		first_conv_layer = True
		for layer in layers_sequence:
			if layer == "conv":
				if first_block and first_conv_layer:
					block_layer = nn.Conv2d(
						in_channels = input_shape,
						out_channels = hidden_units,
						kernel_size = (conv_kernel_size, conv_kernel_size),
						padding = conv_padding
					)
					first_conv_layer = False
				else:
					block_layer = nn.Conv2d(
						in_channels = hidden_units,
						out_channels = hidden_units,
						kernel_size = (conv_kernel_size, conv_kernel_size),
						padding = conv_padding
					)
			elif layer == "activation-f":
				block_layer = nn.ReLU()
			elif layer == "max_pool":
				block_layer = nn.MaxPool2d(
					kernel_size = (max_pool_kernel_size, max_pool_kernel_size),
					stride = max_pool_stride
				)
			conv_pool_block.append(block_layer)
		return conv_pool_block

	def get_layers_sequence_in_block(self, conv_layers: int) -> list[str]:
		layers_sequence = []
		for _ in range(conv_layers):
			layers_sequence.append("conv")
			layers_sequence.append("activation-f")
		layers_sequence.append("max_pool")
		return layers_sequence

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
	cnn_face_detect = One_Face_Detection_CNN(
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