from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series):
    models = build_candidate_models()
    scorecard: dict[str, float] = {}
    trained_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        scorecard[name] = f1_score(y_train, predictions)
        trained_models[name] = model

    best_model_name = max(scorecard, key=scorecard.get)
    best_model = trained_models[best_model_name]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_selection.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_model": best_model_name,
                "training_f1_scores": scorecard,
            },
            file,
            indent=2,
        )

    return best_model_name, best_model, scorecard


def save_model(model: Pipeline, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_saved_model(model_path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)

