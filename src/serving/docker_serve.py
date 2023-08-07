import os
import mlflow

# Set the path to the mlruns directory relative to the current working directory
mlruns_dir = os.path.join('../../data/mlflow', 'mlruns')

# Initialize MLFlow with the relative path
mlflow.set_tracking_uri(mlruns_dir)

# Load the MLflow model using the run_id
model_uri = "runs:/63d821254d6646508bcfa379865ffd32/model"
loaded_model = mlflow.pyfunc.load_model(model_uri = model_uri)
