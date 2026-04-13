from __future__ import annotations

import pandas as pd

from src.config import DROP_COLUMNS, TARGET_COLUMN


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    df["temperature_difference_k"] = (
        df["process_temperature_k"] - df["air_temperature_k"]
    )
    df["power_proxy"] = df["rotational_speed_rpm"] * df["torque_nm"]
    df["wear_torque_interaction"] = df["tool_wear_min"] * df["torque_nm"]
    df["wear_speed_ratio"] = df["tool_wear_min"] / df["rotational_speed_rpm"].clip(
        lower=1
    )

    return df


def prepare_model_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = dataframe.copy()
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN].astype(int)
    return X, y

