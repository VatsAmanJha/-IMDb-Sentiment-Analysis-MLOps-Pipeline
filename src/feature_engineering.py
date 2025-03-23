import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import yaml

# Load parameters
params = yaml.safe_load(open("params.yaml"))["feature_engineering"]
max_features = params["vocab_size"]
params = yaml.safe_load(open("params.yaml"))["data_split"]
test_size = params["test_size"]


def preprocess_data():
    df = pd.read_csv("data/clean/clean_dataset.csv")

    # Apply TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["review"])
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["sentiment"], test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
