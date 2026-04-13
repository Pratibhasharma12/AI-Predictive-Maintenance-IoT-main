from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alerts import generate_alerts
from src.config import PROCESSED_DATA_PATH, REPORTS_DIR
from src.data_loader import load_dataset
from src.evaluate import evaluate_model
from src.features import engineer_features, prepare_model_data
from src.model import save_model, split_data, train_and_select_model
from src.preprocess import clean_dataset
from src.visualize import (
    plot_confusion_matrix,
    plot_failure_distribution,
    plot_probability_distribution,
    plot_top_alerts,
)


def save_processed_dataset(dataframe: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(PROCESSED_DATA_PATH, index=False)


def save_preview_artifacts(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.head(10).to_csv(REPORTS_DIR / "dataset_preview.csv", index=False)

    summary_lines = [
        "Preprocessing Summary",
        "=" * 24,
        f"Raw shape: {raw_df.shape}",
        f"Processed shape: {feature_df.shape}",
        "Engineered features:",
        "- temperature_difference_k",
        "- power_proxy",
        "- wear_torque_interaction",
        "- wear_speed_ratio",
        "Leakage-safe dropped columns during modeling:",
        "- udi",
        "- product_id",
        "- twf",
        "- hdf",
        "- pwf",
        "- osf",
        "- rnf",
    ]
    with open(
        REPORTS_DIR / "preprocessing_summary.txt", "w", encoding="utf-8"
    ) as file:
        file.write("\n".join(summary_lines))


def save_summary(
    metrics: dict,
    selected_model_name: str,
    model_scores: dict,
    alerts: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "AI-Powered Predictive Maintenance Pipeline Summary",
        "=" * 52,
        f"Selected model: {selected_model_name}",
        f"Training F1 scores: {model_scores}",
        f"Accuracy: {metrics['accuracy']}",
        f"Precision: {metrics['precision']}",
        f"Recall: {metrics['recall']}",
        f"ROC-AUC: {metrics['roc_auc']}",
        "",
        "Top 5 maintenance alerts:",
    ]

    for _, row in alerts.head(5).iterrows():
        summary_lines.append(
            (
                f"- Type={row['type']}, tool_wear={row['tool_wear_min']}, "
                f"torque={row['torque_nm']}, probability={row['failure_probability']}, "
                f"alert={row['alert_level']}"
            )
        )

    with open(REPORTS_DIR / "summary.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))


def run_full_pipeline(alert_threshold: float = 0.45, output_dir: Path | None = None):
    raw_df = load_dataset()
    clean_df = clean_dataset(raw_df)
    feature_df = engineer_features(clean_df)
    save_processed_dataset(feature_df)
    save_preview_artifacts(raw_df, feature_df)

    X, y = prepare_model_data(feature_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    selected_model_name, model, model_scores = train_and_select_model(X_train, y_train)
    save_model(model)

    metrics = evaluate_model(model, X_test, y_test)
    predictions = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    alerts = generate_alerts(model, X_test, y_test, threshold=alert_threshold)

    plot_failure_distribution(feature_df, output_dir=output_dir)
    plot_confusion_matrix(model, X_test, y_test, output_dir=output_dir)
    plot_probability_distribution(predictions, output_dir=output_dir)
    plot_top_alerts(alerts, output_dir=output_dir)

    save_summary(metrics, selected_model_name, model_scores, alerts)

    print("Pipeline completed successfully.")
    print(f"Selected model: {selected_model_name}")
    print(
        f"Accuracy={metrics['accuracy']} | Precision={metrics['precision']} | "
        f"Recall={metrics['recall']} | ROC-AUC={metrics['roc_auc']}"
    )
    print(f"Reports saved to: {REPORTS_DIR}")
