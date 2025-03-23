import mlflow
import mlflow.sklearn
import joblib
import json
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import os
from dotenv import load_dotenv

# Define paths
model_path = Path("models/model.pkl")
vectorizer_path = Path("models/vectorizer.pkl")
dataset_source_url = Path("data/clean/clean_dataset.csv")
metrics_path = Path("reports/models/metrics.json")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Set Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("IMDB")

# Start MLflow Run
run_description = "RFC sentiment analysis workflow"
with mlflow.start_run(description=run_description):

    # Load Dataset
    df = pd.read_csv(dataset_source_url)

    # Log Dataset as Input
    dataset = mlflow.data.from_pandas(
        df,
        source=str(dataset_source_url),
        name="clean_dataset.csv",
        targets="sentiment",
    )
    mlflow.log_input(dataset, context="training")

    # Transform Data
    X = vectorizer.transform(df["review"])

    # Log vectorizer and model
    mlflow.log_artifact(str(vectorizer_path))

    # Infer Signature
    signature = infer_signature(X, model.predict(X))

    # Log Model with MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_artifact",
        registered_model_name="sentiment-model",
        pip_requirements=[
            "dvc==3.59.1",
            "mlflow==2.21.0",
            "pandas==2.2.3",
            "numpy==2.2.4",
            "matplotlib==3.10.1",
            "scikit-learn==1.6.1",
            "purifytext==0.1.0",
        ],
        signature=signature,
        input_example=df.sample(5),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        pyfunc_predict_fn="predict",
    )

    # Set Tags
    mlflow.set_tag("labels", json.dumps(["RandomForestClassifier", "classification"]))
    mlflow.set_tag("meta", json.dumps({"framework": "scikit-learn"}))

    # Log Metrics
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        mlflow.log_metrics(metrics)

    # Log Parameters
    model_params = model.get_params()
    mlflow.log_params(model_params)

print("Model, metadata, metrics, and input file logged successfully!")
