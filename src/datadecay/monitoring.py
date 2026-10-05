"""
monitoring.py — Model reliability monitoring for DataDecay-X.

Connects drift to model performance by evaluating each temporal window
independently. Tracks F1, degradation, and health score over time.

IMPORTANT: Drift and model reliability are SEPARATE signals.
Drift does not necessarily imply model failure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import evaluate_model, predict_model
from .preprocessing import prepare_features, transform_with_preprocessor
from .utils import safe_float, safe_round

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window-level evaluation
# ---------------------------------------------------------------------------

def evaluate_model_by_window(
    model: Any,
    windows: List[pd.DataFrame],
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    preprocessor: Any,
    label_encoder: Any,
    metric: str = "f1",
) -> pd.DataFrame:
    """
    Evaluate a fitted model on each temporal window independently.

    Parameters
    ----------
    model          : fitted sklearn estimator
    windows        : list of DataFrames, one per temporal window
    target_column  : name of the target column
    numerical_cols : numerical feature columns
    categorical_cols : categorical feature columns
    preprocessor   : fitted ColumnTransformer
    label_encoder  : fitted LabelEncoder for target
    metric         : primary reliability metric (default 'f1')

    Returns
    -------
    pd.DataFrame with per-window metrics:
        window, n_rows, accuracy, precision, recall, f1, primary_metric
    """
    rows = []

    for win_idx, win_df in enumerate(windows):
        try:
            X_win, y_win = prepare_features(
                win_df,
                target_column=target_column,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
            )

            if len(X_win) == 0 or len(y_win) == 0:
                continue

            X_trans = transform_with_preprocessor(
                X_win, preprocessor, numerical_cols, categorical_cols
            )

            # Encode target
            y_enc = np.array([
                int(label_encoder.transform([str(v)])[0])
                if str(v) in label_encoder.classes_ else -1
                for v in y_win
            ])

            # Filter out unknown labels
            valid_mask = y_enc != -1
            if valid_mask.sum() < 2:
                continue

            X_valid = X_trans.values[valid_mask]
            y_valid = y_enc[valid_mask]

            y_pred = predict_model(model, X_valid)
            metrics = evaluate_model(y_valid, y_pred)

            primary = metrics.get(metric, metrics.get("f1", 0.0))

            rows.append({
                "window": win_idx,
                "n_rows": int(valid_mask.sum()),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "primary_metric": primary,
            })

        except Exception as exc:
            logger.warning("Window %d evaluation failed: %s", win_idx, exc)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability degradation
# ---------------------------------------------------------------------------

def calculate_reliability_degradation(
    reliability_timeline: pd.DataFrame,
    baseline_metric: float,
    metric_col: str = "f1",
) -> Dict[str, Any]:
    """
    Calculate how much model reliability has degraded from the baseline.

    Parameters
    ----------
    reliability_timeline : window-level metrics DataFrame
    baseline_metric      : reference/baseline metric value
    metric_col           : column to track (default 'f1')

    Returns
    -------
    dict with degradation statistics
    """
    if reliability_timeline.empty or metric_col not in reliability_timeline.columns:
        return {
            "baseline": baseline_metric,
            "current": baseline_metric,
            "minimum": baseline_metric,
            "maximum": baseline_metric,
            "degradation_pct": 0.0,
            "absolute_degradation": 0.0,
            "trend": "stable",
        }

    values = reliability_timeline[metric_col].values
    current = float(values[-1]) if len(values) > 0 else baseline_metric
    minimum = float(values.min()) if len(values) > 0 else baseline_metric
    maximum = float(values.max()) if len(values) > 0 else baseline_metric

    absolute_degradation = baseline_metric - current
    degradation_pct = (absolute_degradation / max(baseline_metric, 1e-6)) * 100

    # Trend: compare last 3 windows vs first 3 windows
    n = len(values)
    if n >= 6:
        early_mean = float(np.mean(values[:3]))
        recent_mean = float(np.mean(values[-3:]))
        if recent_mean < early_mean - 0.02:
            trend = "declining"
        elif recent_mean > early_mean + 0.02:
            trend = "improving"
        else:
            trend = "stable"
    elif n >= 2:
        if values[-1] < values[0] - 0.02:
            trend = "declining"
        elif values[-1] > values[0] + 0.02:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "baseline": round(baseline_metric, 4),
        "current": round(current, 4),
        "minimum": round(minimum, 4),
        "maximum": round(maximum, 4),
        "degradation_pct": round(degradation_pct, 2),
        "absolute_degradation": round(absolute_degradation, 4),
        "trend": trend,
    }


def calculate_health_score(
    current_metric: float,
    baseline_metric: float,
    threshold: float = 0.70,
) -> float:
    """
    Compute a normalized health score in [0, 1].

    Formula:
        health = (current - threshold) / (baseline - threshold)
        clipped to [0, 1]

    A score of 1.0 means the model is at baseline performance.
    A score of 0.0 means the model is at or below the threshold.
    """
    denominator = baseline_metric - threshold
    if abs(denominator) < 1e-6:
        return 1.0 if current_metric >= threshold else 0.0

    health = (current_metric - threshold) / denominator
    return float(np.clip(health, 0.0, 1.0))


def calculate_time_below_threshold(
    reliability_timeline: pd.DataFrame,
    threshold: float,
    metric_col: str = "f1",
) -> Dict[str, Any]:
    """
    Count how many windows have metric below the reliability threshold.

    Returns
    -------
    dict with n_windows_below, consecutive_below, first_crossing_window
    """
    if reliability_timeline.empty or metric_col not in reliability_timeline.columns:
        return {
            "n_windows_below": 0,
            "consecutive_below": 0,
            "first_crossing_window": None,
            "pct_below": 0.0,
        }

    values = reliability_timeline[metric_col].values
    windows = reliability_timeline["window"].values if "window" in reliability_timeline.columns else \
              np.arange(len(values))

    below_mask = values < threshold
    n_below = int(below_mask.sum())
    n_total = len(values)

    # First crossing window
    first_crossing = None
    crossings = np.where(below_mask)[0]
    if len(crossings) > 0:
        first_crossing = int(windows[crossings[0]])

    # Max consecutive windows below threshold
    max_consecutive = 0
    current_consecutive = 0
    for b in below_mask:
        if b:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return {
        "n_windows_below": n_below,
        "consecutive_below": max_consecutive,
        "first_crossing_window": first_crossing,
        "pct_below": round(n_below / n_total * 100, 1) if n_total > 0 else 0.0,
    }


def build_reliability_timeline(
    window_metrics: pd.DataFrame,
    baseline_metric: float,
    threshold: float,
    drift_severity_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build an enriched reliability timeline DataFrame combining
    model performance and drift information.

    Parameters
    ----------
    window_metrics         : per-window metrics from evaluate_model_by_window
    baseline_metric        : baseline reference performance value
    threshold              : reliability threshold
    drift_severity_table   : optional window-level drift severity DataFrame

    Returns
    -------
    pd.DataFrame with columns:
        window, f1, accuracy, precision, recall,
        health_score, below_threshold, degradation_pct,
        [drift_severity_index, severity_label] (if drift provided)
    """
    if window_metrics.empty:
        return pd.DataFrame()

    timeline = window_metrics.copy()

    # Health score per window
    timeline["health_score"] = timeline["f1"].apply(
        lambda x: round(calculate_health_score(x, baseline_metric, threshold), 4)
    )
    timeline["below_threshold"] = timeline["f1"] < threshold
    timeline["degradation_pct"] = timeline["f1"].apply(
        lambda x: round((baseline_metric - x) / max(baseline_metric, 1e-6) * 100, 2)
    )

    # Merge drift data if available
    if drift_severity_table is not None and not drift_severity_table.empty:
        drift_cols = ["window", "drift_severity_index", "severity_label"]
        available_drift_cols = [c for c in drift_cols if c in drift_severity_table.columns]
        if len(available_drift_cols) > 1:
            timeline = timeline.merge(
                drift_severity_table[available_drift_cols],
                on="window",
                how="left",
            )

    return timeline


def compare_model_reliability(
    model_results_by_name: Dict[str, Any],
    windows: List[pd.DataFrame],
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    preprocessor: Any,
    label_encoder: Any,
    metric: str = "f1",
) -> Dict[str, pd.DataFrame]:
    """
    Compare reliability timelines across multiple models.

    Returns
    -------
    dict keyed by model_name → reliability timeline DataFrame
    """
    timelines: Dict[str, pd.DataFrame] = {}

    for model_name, result in model_results_by_name.items():
        model = result.get("model")
        if model is None:
            continue

        baseline_metrics = result.get("metrics", {})
        baseline_val = baseline_metrics.get(metric, 0.0)

        try:
            window_metrics = evaluate_model_by_window(
                model=model,
                windows=windows,
                target_column=target_column,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                preprocessor=preprocessor,
                label_encoder=label_encoder,
                metric=metric,
            )
            timelines[model_name] = window_metrics
        except Exception as exc:
            logger.warning("Reliability comparison failed for %s: %s", model_name, exc)

    return timelines
