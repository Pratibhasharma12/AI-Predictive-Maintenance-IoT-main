from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import FIGURES_DIR


sns.set_theme(style="whitegrid")


def ensure_figure_dir(output_dir: Path | None = None) -> Path:
    figure_dir = FIGURES_DIR if output_dir is None else output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def plot_failure_distribution(dataframe: pd.DataFrame, output_dir: Path | None = None):
    figure_dir = ensure_figure_dir(output_dir)
    plt.figure(figsize=(7, 5))
    sns.countplot(data=dataframe, x="machine_failure", hue="machine_failure", legend=False)
    plt.title("Failure vs Non-Failure Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_dir / "failure_distribution.png", dpi=300)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, output_dir: Path | None = None):
    figure_dir = ensure_figure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_probability_distribution(
    predictions: pd.DataFrame, output_dir: Path | None = None
):
    figure_dir = ensure_figure_dir(output_dir)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        predictions["failure_probability"],
        bins=25,
        kde=True,
        color="#d95f02",
    )
    plt.title("Predicted Failure Probability Distribution")
    plt.xlabel("Failure Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figure_dir / "failure_probability_distribution.png", dpi=300)
    plt.close()


def plot_top_alerts(alerts: pd.DataFrame, output_dir: Path | None = None):
    figure_dir = ensure_figure_dir(output_dir)
    chart_data = alerts.head(10).copy()
    chart_data["sample_id"] = chart_data.index.astype(str)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=chart_data,
        x="failure_probability",
        y="sample_id",
        hue="alert_level",
        palette={"high": "#d62728", "medium": "#ff7f0e", "normal": "#2ca02c"},
    )
    plt.title("Top Predicted Risky Machines")
    plt.xlabel("Predicted Failure Probability")
    plt.ylabel("Sample Rank")
    plt.tight_layout()
    plt.savefig(figure_dir / "top_risky_machines.png", dpi=300)
    plt.close()

