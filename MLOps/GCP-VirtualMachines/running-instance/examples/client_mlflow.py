import mlflow
from dotenv import dotenv_values
from mlflow import MlflowClient

# Configuration of tracking server
# config = dotenv_values(".env")
config = dotenv_values(".env.testing")
mlfow_server_ip = config["MLFOW_SERVER_IP"]
mlflow.set_tracking_uri(mlfow_server_ip)
client = MlflowClient()


# Start experiment
experiment_name = "iris_logistic_model"

# Fetch experiment metadata information
experiment = client.get_experiment_by_name(experiment_name)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
