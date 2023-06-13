import mlflow
import json
import os
import numpy as np
import pandas as pd
from prefect.filesystems import S3

from training.landcover_mlflow_wrapper import LancoverMlflowWrapper

def confusion_matrix_to_metrics(confusion_matrix_path: np.ndarray) -> dict:

    confusion_matrix = pd.read_csv(confusion_matrix_path, index_col=0)

    classes = confusion_matrix.columns.tolist()
    matrix_array = confusion_matrix.values.astype(int)
    metrics = {}

    for i, class_name in enumerate(classes):
        values = matrix_array[i]

        tp = values[i]
        fn = np.sum(values) - tp
        fp = np.sum(matrix_array[:, i]) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        metrics[f'precision_{class_name}'] = precision
        metrics[f'recall_{class_name}'] = recall
        metrics[f'f1_score_{class_name}'] = f1_score

    total_tp = np.sum(np.diagonal(matrix_array))
    total_fp = np.sum(matrix_array, axis=0) - np.diagonal(matrix_array)

    global_accuracy = total_tp / (total_tp + np.sum(total_fp))
    metrics['accuracy'] = global_accuracy

    return metrics

s3_block = S3.load("khaos-minio")

os.environ["AWS_ACCESS_KEY_ID"] = s3_block.aws_access_key_id.get_secret_value()
os.environ["AWS_SECRET_ACCESS_KEY"] = s3_block.aws_secret_access_key.get_secret_value()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://<S3-IP>:<S3-PORT>"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

model_file = "base_model/model.joblib"
confusion_matrix_png = "base_model/confusion_matrix.png"
metadata_file = "base_model/metadata.json"
training_data = "base_model/training_dataset.csv"
testing_data = "base_model/testing_dataset.csv"
confusion_matrix_csv = "base_model/confusion_matrix.csv"

mlflow.set_tracking_uri("http://<CLUSTER-IP>:<MLFLOW-PORT>")

experiment_id = mlflow.get_experiment_by_name("landcoverpy").experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    mlflow.pyfunc.log_model(artifact_path="model",
                            registered_model_name="landcoverpy",
                            code_path=["landcoverpy_mlflow_wrapper.py", "landcoverpy_model.py"],
                            pip_requirements=["landcoverpy","mlflow==2.3.1"],
                            python_model=LancoverMlflowWrapper(),
                            artifacts={"model_file": model_file, "confusion_matrix": confusion_matrix_png, "training_data": training_data, "testing_data": testing_data, "metadata_file": metadata_file}
                            )
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    mlflow.log_params(metadata)

    metrics = confusion_matrix_to_metrics(confusion_matrix_csv)
    mlflow.log_metrics(metrics)