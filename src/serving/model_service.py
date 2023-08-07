import mlflow
import bentoml
import pandas as pd
from bentoml.io import PandasDataFrame
import os

# Set the path to mlrun
mlruns_dir = './../../data/mlflow/mlruns'
# Initialize MLFlow
mlflow.set_tracking_uri(mlruns_dir)

# Load the trained MLFLow model using the run ID
def get_run_requirements(run_id):
    model_uri = f'runs:/{run_id}/anom_weight_10_fold_2/model'
    model = mlflow.pyfunc.load_model(model_uri = model_uri)
    requirements = mlflow.pyfunc.load_requirements(model_uri = model_uri, format = 'pip')
    return model, requirements

model, requirements = get_run_requirements(run_id = '63d821254d6646508bcfa379865ffd32')

@bentoml.environment_variable(infer_pip_packages = False, pip_packages = requirements)
@bentoml.artifacts([bentoml.PickleArtifact('model')])
class FraudDetector(bentoml.BentoService):
    @bentoml.api(input = PandasDataFrame(), output = PandasDataFrame())
    def predict(self, input_data):
        input_df = pd.DataFrame(input_data)
        predictions = self.artifacts.model.predict(input_df)
        return predictions

# Initialize BentoService
bento_svc = FraudDetector()
bento_svc.pack('model', model)

# Save the BentoService to a directory
saved_path = bento_svc.save()