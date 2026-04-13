from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import urllib.request

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import DATASET_PATH

UCI_ZIP_URL = (
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)


def ensure_directories() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_from_official_zip() -> Path:
    zip_path = DATASET_PATH.parent / "ai4i2020.zip"
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with ZipFile(zip_path, "r") as zip_file:
        csv_members = [member for member in zip_file.namelist() if member.endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError("No CSV file found inside downloaded dataset archive.")
        with zip_file.open(csv_members[0]) as source, open(DATASET_PATH, "wb") as target:
            target.write(source.read())

    if zip_path.exists():
        zip_path.unlink()

    return DATASET_PATH


def download_dataset_if_needed() -> Path:
    """Download the UCI AI4I dataset once and cache it locally."""
    ensure_directories()

    if DATASET_PATH.exists():
        return DATASET_PATH

    try:
        dataset = fetch_ucirepo(id=601)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()

        dataframe = pd.concat([features, targets], axis=1)
        dataframe.to_csv(DATASET_PATH, index=False)
        return DATASET_PATH
    except Exception:
        return download_from_official_zip()


def load_dataset(local_path: Path | None = None) -> pd.DataFrame:
    source_path = local_path if local_path else download_dataset_if_needed()
    dataframe = pd.read_csv(source_path)
    return dataframe
