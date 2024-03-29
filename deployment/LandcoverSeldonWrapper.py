import json
import os

import mlflow

from landcoverpy.landcover_model import LandcoverModel

class LandcoverSeldonWrapper:

    def __init__(self, model_version="latest", file_path="files"):
        
        self.model_version = model_version
        self.file_path = file_path
        self.ready = False

    def load_model(self):
        model_uri = f"models:/landcoverpy/{self.model_version}"
        download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=self.file_path)

        model_file = os.path.join(download_path, "artifacts", "model.joblib")
        metadata_file = os.path.join(download_path, "artifacts", "metadata.json")

        with open(metadata_file) as f:
            metadata = json.load(f)
        used_columns = metadata["used_columns"]

        self.predictor = LandcoverModel(model_file=model_file, used_columns=used_columns)

        self.ready = True

    def predict(self, X, features_names=None):
        if not self.ready:
            self.load_model()
        
        download_url, prediction_metrics = self.predictor.predict(X)

        self.prediction_metrics = prediction_metrics

        return {'result' : download_url}
    
    def metrics(self):
        custom_gauge_metrics = [{"type": "GAUGE", "key": k, "value": self.prediction_metrics[k]} for k in self.prediction_metrics]
        return custom_gauge_metrics
