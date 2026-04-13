from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DATASET_FILENAME = "ai4i2020.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_ai4i2020.csv"
MODEL_PATH = MODELS_DIR / "predictive_maintenance_model.joblib"

TARGET_COLUMN = "machine_failure"
DROP_COLUMNS = [
    "udi",
    "product_id",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

CATEGORICAL_COLUMNS = ["type"]
NUMERIC_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temperature_difference_k",
    "power_proxy",
    "wear_torque_interaction",
    "wear_speed_ratio",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

