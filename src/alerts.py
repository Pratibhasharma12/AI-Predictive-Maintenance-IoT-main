from __future__ import annotations

import pandas as pd

from src.config import REPORTS_DIR


def probability_to_alert(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.75):
        return "high"
    if probability >= threshold:
        return "medium"
    return "normal"


def generate_alerts(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    alert_frame = X_test.copy()
    alert_frame["actual_failure"] = y_test.to_numpy()
    alert_frame["predicted_failure"] = predictions
    alert_frame["failure_probability"] = probabilities.round(4)
    alert_frame["alert_level"] = alert_frame["failure_probability"].apply(
        lambda value: probability_to_alert(value, threshold)
    )

    top_alerts = (
        alert_frame.sort_values("failure_probability", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top_alerts.to_csv(REPORTS_DIR / "top_alerts.csv", index=False)
    return top_alerts

