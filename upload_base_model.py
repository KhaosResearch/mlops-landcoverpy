import mlflow
import os
from prefect.filesystems import S3

from training.landcoverpy_mlflow_wrapper import LancoverpyMlflowWrapper

s3_block = S3.load("khaos-minio")

os.environ["AWS_ACCESS_KEY_ID"] = s3_block.aws_access_key_id.get_secret_value()
os.environ["AWS_SECRET_ACCESS_KEY"] = s3_block.aws_secret_access_key.get_secret_value()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://<S3-IP>:<S3-PORT>"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

model_file = "training/base_model/model.joblib"

mlflow.set_tracking_uri("http://<CLUSTER-IP>:<MLFLOW-PORT>")

experiment_id = mlflow.get_experiment_by_name("landcoverpy").experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    mlflow.pyfunc.log_model(artifact_path="",
                            registered_model_name="landcoverpy",
                            code_path=["training/landcoverpy_mlflow_wrapper.py"],
                            pip_requirements=["landcoverpy","mlflow==2.3.1"],
                            python_model=LancoverpyMlflowWrapper(),
                            artifacts= {"model_file": model_file}
                            )