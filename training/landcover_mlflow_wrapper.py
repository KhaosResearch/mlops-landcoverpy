import json
import os

import mlflow
from landcoverpy.landcover_model import LandcoverModel

class LancoverMlflowWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_file = context.artifacts["model_file"]
        metadata_file = context.artifacts["metadata_file"]
        with open(metadata_file) as f:
            metadata = json.load(f)
        used_columns = metadata["used_columns"]
        self.landcoverpy_model = LandcoverModel(model_file, used_columns)

    def predict(self, context, model_input_dict):
        url, _ = self.landcoverpy_model.predict(**model_input_dict)
        return url