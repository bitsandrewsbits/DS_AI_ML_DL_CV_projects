# CNN model architecture - TinyVGG
import torch
from torch import nn

class TinyVGG_Architecture_CNN(nn.Module):
	def __init__(self, input_shape: int, hidden_units: int, output_shape: int, data_point_shape: torch.Tensor):
		super().__init__()
		self.input_shape = input_shape
		self.hidden_units = hidden_units
		self.output_shape = output_shape
		self.data_point_shape = data_point_shape

		self.input_conv_pool_block = nn.Sequential(
			nn.Conv2d(
				in_channels = input_shape, out_channels = hidden_units,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = hidden_units, out_channels = hidden_units,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
		)
		self.conv_pool_block = nn.Sequential(
			nn.Conv2d(
				in_channels = hidden_units, out_channels = hidden_units,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = hidden_units, out_channels = hidden_units,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
		)
		self.features_extraction_blocks = self.get_features_extraction_CNN_part(2)
		self.linear_classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features = self.get_flatten_size(), out_features = output_shape)
		)

	def get_features_extraction_CNN_part(self, blocks_n: int, include_input_block = True) -> nn.ModuleList:
		features_extraction_blocks = nn.ModuleList()
		if include_input_block:
			features_extraction_blocks.append(self.input_conv_pool_block)
			blocks_n -= 1
		for _ in range(blocks_n):
			features_extraction_blocks.append(self.conv_pool_block)
		return features_extraction_blocks

	def forward(self, set_x):
		features_ext_block_out = set_x
		for features_ext_block in self.features_extraction_blocks:
			features_ext_block_out = features_ext_block(features_ext_block_out)

		lin_classifier_result = self.linear_classifier(features_ext_block_out)
		return lin_classifier_result

	def get_flatten_size(self):
		test_tensor = torch.zeros(
			size = self.data_point_shape
		)
		for features_ext_block in self.features_extraction_blocks:
			test_tensor = features_ext_block(test_tensor)
		features_ext_block_output_shape = test_tensor.shape[1:]
		flatten_size = 1
		for dim_size in features_ext_block_output_shape:
			flatten_size *= dim_size
		return flatten_size

if __name__ == "__main__":
	color_channels = 3
	classes_amount = 10
	test_batch_data = torch.randn(size = (32, color_channels, 128, 128))
	model = TinyVGG_Architecture_CNN(color_channels, 64, classes_amount, test_batch_data.shape)
	test_model_inference_logits = model(test_batch_data)
	print("Output logits shape:", test_model_inference_logits.shape)
