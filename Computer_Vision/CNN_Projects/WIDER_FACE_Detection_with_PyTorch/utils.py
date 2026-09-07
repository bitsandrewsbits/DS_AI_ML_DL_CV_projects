# additional functions after training models
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_loss_curves_figure(epochs: list, losses: dict[list]):
	fig = plt.figure(figsize = (16.5, 8.5))
	for loss_type in losses:
		show_loss_curve_plot(epochs, losses[loss_type], loss_type)
	return fig

def get_MAE_curves_figure(epochs: list, MAEs: dict[list]):
	fig = plt.figure(figsize = (16.5, 8.5))
	for MAE_type in MAEs:
		show_MAE_curve_plot(epochs, MAEs[MAE_type], MAE_type)
	return fig

def show_loss_curve_plot(epochs: list, loss: list, loss_type: str):
	plt.plot(epochs, loss, label = f'{loss_type}-loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.grid(True)

def show_MAE_curve_plot(epochs: list, MAE: list, MAE_type: str):
	plt.plot(epochs, MAE, label = f'{MAE_type}-MAE')
	plt.xlabel('epoch')
	plt.ylabel('MAE')
	plt.legend()
	plt.grid(True)

def get_trained_model_dir(trained_models_root_path: Path, cv_task: str) -> Path:
	trained_models_for_cv_task_root_path = trained_models_root_path / cv_task
	trained_models_for_cv_task_root_path.mkdir(parents = True, exist_ok = True)
	
	trained_model_dirname_pattern = "training_#_"
	trained_models_cv_task = os.listdir(trained_models_for_cv_task_root_path)
	
	if trained_models_cv_task != 0:
		current_training_num = len(trained_models_cv_task) + 1
		trained_model_dirname = f"{trained_model_dirname_pattern}{current_training_num}"
	else:
		trained_model_dirname = f"{trained_model_dirname_pattern}0";
	
	trained_model_dir_path = trained_models_for_cv_task_root_path / trained_model_dirname
	trained_model_dir_path.mkdir(parents = True, exist_ok = True)
	return trained_model_dir_path

def save_loss_curves_plot(figure: plt.Figure, trained_model_dir_path: Path):
	plot_path = trained_model_dir_path / "losses.png"
	figure.savefig(plot_path)

def save_MAE_curves_plot(figure: plt.Figure, trained_model_dir_path: Path):
	plot_path = trained_model_dir_path / "MAE_curves.png"
	figure.savefig(plot_path)

def save_model_weights(trained_model_dir_path: Path, model: torch.nn.Module):
	model_name = "trained_model.pth"
	saved_model_path = trained_model_dir_path / model_name
	print(f"[INFO] Saving model to {saved_model_path}.")
	torch.save(obj = model.state_dict(), f = saved_model_path)