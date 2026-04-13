from __future__ import annotations

import re

import pandas as pd


def normalize_column_name(column: str) -> str:
    clean = column.strip().lower()
    clean = clean.replace("[", "_").replace("]", "")
    clean = clean.replace("%", "percent")
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_")


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip()

    df = df.drop_duplicates().reset_index(drop=True)
    return df

