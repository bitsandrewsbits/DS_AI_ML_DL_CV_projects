# MLFlow experiment environment initiating and running
import mlflow as mlf
import os

class MLFlow_Experiment_Runner:
	def __init__(self, classifier_model_name: str):
		self.classifier_model_name = classifier_model_name
		self.experiment_name = self.get_experiment_name()
		self.tracking_server_IP = os.getenv('MLFLOW_IP')
		self.tracking_server_port = 5000
	
	def main(self):
		self.set_tracking_uri()

	def set_mlflow_server_URI(self):
		mlf.set_tracking_uri(f"http://{self.tracking_server_IP}:{self.tracking_server_port}")
	
	def get_experiment_name(self):
		return f"{self.classifier_model_name}_classifier"

	# TODO: think, 1)how to check and 2)create new/set existed experiment as current
	def experiment_exist(self):
		pass

if __name__ == '__main__':
	mlflow_experiment = MLFlow_Experiment_Runner('random_forest')
	print(mlflow_experiment.experiment_name)