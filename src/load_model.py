import mlflow
import mlflow.sklearn
import joblib
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Set MLflow Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Experiment and Model Name
experiment_name = "IMDB"
model_name = "sentiment-model"

# Get the Experiment ID
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found!")
experiment_id = experiment.experiment_id

# Fetch all model runs from the experiment
runs = mlflow.search_runs(
    experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"]
)

if runs.empty:
    raise ValueError(f"No runs found for experiment '{experiment_name}'.")

# Get the best model run (highest accuracy)
best_run = runs.iloc[0]  # First row has the highest accuracy due to sorting
best_run_id = best_run["run_id"]
best_accuracy = best_run["metrics.accuracy"]

# Get the model URI
best_model_uri = f"runs:/{best_run_id}/model_artifact"
print(f"Best Model URI: {best_model_uri}")
print(f"Best Model Accuracy: {best_accuracy}")

# Load the best model
best_model = mlflow.sklearn.load_model(best_model_uri)

# Save the best model locally
joblib.dump(best_model, "models/best_model.pkl")
print("Best model saved as 'models/best_model.pkl'.")

# Verify the model type
print("Model Type:", type(best_model))
