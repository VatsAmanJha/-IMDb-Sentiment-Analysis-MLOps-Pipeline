from sklearn.ensemble import RandomForestClassifier
import yaml
import joblib
import sys
import os

# Load parameters
params = yaml.safe_load(open("params.yaml"))["training"]
n_estimators, random_state = params["n_estimators"], params["random_state"]


def build_model():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return model


def train_model():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.feature_engineering import preprocess_data
    X_train, X_test, y_train, y_test = preprocess_data()
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, "models/model.pkl")
    print("Model saved as models/model.pkl")


if __name__ == "__main__":
    train_model()
