# additional functions after training models
import matplotlib.pyplot as plt
from pathlib import Path
import os

def show_loss_curve_plot(epochs: list, losses: list, loss_type: str):
	plt.plot(epochs, losses, label = f'{loss_type}-loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
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

# TODO: create and test
def save_model_weights(model: torch.nn.Module, model_name: str):
	pass