import zipfile
from pathlib import Path
import pandas as pd
from purifytext import clean_text

data_dir = Path("data")


def load_dataset():
    file_path = data_dir / "raw" / "extracted" / "IMDB Dataset.csv"
    return pd.read_csv(file_path)


def clean_data(df):
    df.dropna(inplace=True)
    df = clean_text(
        dataframe=df, column_name="review", stemming=True, lemmatizing=False
    )
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    out_dir = data_dir / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "clean_dataset.csv", index=False)


if __name__ == "__main__":
    df = load_dataset()
    clean_data(df)
