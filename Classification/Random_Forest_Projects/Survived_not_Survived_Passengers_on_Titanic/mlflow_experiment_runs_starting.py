# MLFlow experiment environment initiating and running
import mlflow as mlf
import os

class MLFlow_Experiment_Runner:
	def __init__(self, classifier_model_name: str):
		self.classifier_model_name = classifier_model_name
		self.experiment_name = self.get_experiment_name()
		self.active_experiment = self.set_experiment_as_active()
		self.tracking_server_IP = os.getenv('MLFLOW_IP')
		self.tracking_server_port = 5000
	
	def main(self):
		self.set_mlflow_server_URI()
		if self.experiment_exist():
			print(f'[INFO] Experiment - {self.experiment_name} - active!')
			self.set_experiment_as_active()
		else:
			print('[INFO] Experiment - {self.experiment_name} does not exist!')
			self.create_experiment(self.experiment_name)

	def set_mlflow_server_URI(self):
		mlf.set_tracking_uri(f"http://{self.tracking_server_IP}:{self.tracking_server_port}")
		return True
	
	def get_experiment_name(self):
		return f"{self.classifier_model_name}_classifier"

	# TODO: think, 1)how to check and 2)create new/set existed experiment as current
	def experiment_exist(self):
		if mlf.get_experiment_by_name(self.experiment_name):
			print(f'[INFO] Experiment {self.experiment_name} already exist!')
			return True
		return False

	def create_new_experiment(self):
		print(f'[INFO] Creating experiment - {self.experiment_name}...')
		mlf.create_experiment(self.experiment_name)

	def set_experiment_as_active(self):
		return mlf.set_experiment(self.experiment_name)

if __name__ == '__main__':
	mlflow_experiment = MLFlow_Experiment_Runner('random_forest')
	mlflow_experiment.main()