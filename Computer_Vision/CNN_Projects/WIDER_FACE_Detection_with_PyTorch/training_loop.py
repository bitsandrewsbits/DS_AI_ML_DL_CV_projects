# model training loop
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import pytorch_datasets_creation as pdc
import CNN_one_face_detection_model as cofdm

class Model_Training:
	def __init__(self, model_obj, dataloaders: dict[torch.utils.data.DataLoader],
	learning_rate = 0.01, epochs = 3):
		self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
		self.face_detect_model = model_obj.to(self.compute_device)
		self.dataloaders = dataloaders
		self.loss_func = CrossEntropyLoss()
		self.learning_rate = learning_rate
		self.optimizer = Adam(
			params = self.face_detect_model.parameters(),
			lr = self.learning_rate
		)
		self.epochs = epochs

	def train_model(self):
		for epoch in range(1, self.epochs + 1):
			epoch_loss = self.train_step()
			epoch_val_loss = self.validation_step()
			print(
				f"[INFO] Epoch #{epoch}: train_loss = {epoch_loss}, valid_loss = {epoch_val_loss}"
			)

	def train_step(self):
		self.face_detect_model.train()
		epoch_loss = 0
		for image_batch, label_batch in self.dataloaders["train"]:
			image_batch = image_batch.to(self.compute_device)
			label_batch = label_batch.to(self.compute_device).squeeze().to(torch.float16)
			pred_bbx_x_y_w_h = self.face_detect_model(image_batch)
			batch_loss = self.loss_func(pred_bbx_x_y_w_h, label_batch)
			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()
			epoch_loss += batch_loss
		epoch_loss = round(epoch_loss.item() / len(self.dataloaders["train"]), 3)
		return epoch_loss

	def validation_step(self):
		self.face_detect_model.eval()
		valid_loss = 0
		with torch.inference_mode():
			for image_batch, label_batch in self.dataloaders["val"]:
				image_batch = image_batch.to(self.compute_device)
				label_batch = label_batch.to(self.compute_device).squeeze().to(torch.float16)
				pred_bbx_x_y_w_h = self.face_detect_model(image_batch)
				val_batch_loss = self.loss_func(pred_bbx_x_y_w_h, label_batch)
				valid_loss += val_batch_loss
		valid_loss = round(valid_loss.item() / len(self.dataloaders["val"]), 3)
		return valid_loss

if __name__ == "__main__":
	BATCH_SIZE = 32
	datasets_prep = pdc.Datasets_Praparation(
		image_size = (224, 224),
		batch_size = BATCH_SIZE,
		task = "one_face",
		datasets_sizes = {"train": 1000, "val": 50, "test": 100}
	)
	datasets_prep.main()
	face_detect_dataloaders = datasets_prep.dataloaders

	face_detect_model = cofdm.One_Face_Detection_CNN(
		input_shape = 3,
		hidden_units = 32,
		output_shape = 4,
		batch_size = BATCH_SIZE,
		image_wh = (224, 224)
	)

	model_train = Model_Training(
		model_obj = face_detect_model,
		dataloaders = face_detect_dataloaders,
		learning_rate = 0.1,
		epochs = 20
	)
	model_train.train_model()