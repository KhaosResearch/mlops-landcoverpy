import os

import mlflow
from training.landcoverpy_model import LandcoverpyModel

class LancoverpyMlflowWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_file = context.artifacts["model_file"]
        self.landcoverpy_model = LandcoverpyModel()

    def predict(self, context, model_input_dict):
        out_df = self.landcoverpy_model.predict(**model_input_dict)
        return out_df