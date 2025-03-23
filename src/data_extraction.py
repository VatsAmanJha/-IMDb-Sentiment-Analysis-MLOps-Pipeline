import zipfile
from pathlib import Path

data_dir = Path("data")


def extract_data():
    raw_zip = data_dir / "raw" / "dataset.zip"
    extract_path = data_dir / "raw" / "extracted"
    if not raw_zip.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_zip}")
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)


if __name__ == "__main__":
    extract_data()
