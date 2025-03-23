import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import sys
import os


def evaluate_model():
    model = joblib.load("models/model.pkl")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.feature_engineering import preprocess_data

    _, X_test, _, y_test = preprocess_data()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics = {"accuracy": acc}
    with open("reports/models/metrics.json", "w") as f:
        json.dump(metrics, f)
    print(metrics)


if __name__ == "__main__":
    evaluate_model()
