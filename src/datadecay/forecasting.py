"""
forecasting.py — Lightweight reliability forecasting for DataDecay-X.

Implements:
  1. Persistence forecast (naive baseline)
  2. Linear trend forecast
  3. XGBoost regression forecaster (optional)

TTD (Time-to-Decay) is an ESTIMATE, not a guaranteed countdown.
All forecasts are labeled as estimates throughout.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

EPS = 1e-8


# ---------------------------------------------------------------------------
# Persistence forecast
# ---------------------------------------------------------------------------

def forecast_persistence(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Persistence forecast: predict the last observed value for all future steps.

    Simple but robust baseline — useful when the series is stable.

    Parameters
    ----------
    history : array of observed reliability values (chronological)
    horizon : number of future windows to forecast

    Returns
    -------
    array of length horizon
    """
    if len(history) == 0:
        return np.full(horizon, 0.0)
    last_val = float(history[-1])
    return np.full(horizon, last_val)


# ---------------------------------------------------------------------------
# Linear trend forecast
# ---------------------------------------------------------------------------

def forecast_linear_trend(
    history: np.ndarray,
    horizon: int,
    min_points: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit a linear trend to historical reliability and extrapolate.

    Parameters
    ----------
    history    : array of observed reliability values
    horizon    : windows to forecast ahead
    min_points : minimum observations required; fallback to persistence if fewer

    Returns
    -------
    (forecast array of length horizon, trend_metadata dict)
    """
    n = len(history)
    if n < min_points:
        # Fallback to persistence
        return forecast_persistence(history, horizon), {
            "slope": 0.0, "intercept": float(history[-1]) if n > 0 else 0.0,
            "r_squared": 0.0, "method": "persistence_fallback",
        }

    x = np.arange(n, dtype=float)
    y = np.array(history, dtype=float)

    # Least-squares linear fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Compute R²
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / (ss_tot + EPS)

    # Forecast
    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = np.clip(np.polyval(coeffs, future_x), 0.0, 1.0)

    trend_meta = {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "method": "linear_trend",
    }

    return forecast, trend_meta


# ---------------------------------------------------------------------------
# XGBoost forecaster (optional)
# ---------------------------------------------------------------------------

def forecast_xgboost(
    history: np.ndarray,
    horizon: int,
    n_lags: int = 3,
    random_seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    XGBoost-based autoregressive reliability forecaster.

    Requires xgboost to be installed. Falls back to linear trend if unavailable.

    Parameters
    ----------
    history    : observed reliability values
    horizon    : number of future windows
    n_lags     : number of lag features
    random_seed: reproducibility

    Returns
    -------
    (forecast array, metadata dict)
    """
    if not _XGB_AVAILABLE or len(history) < n_lags + 2:
        logger.info("XGBoost forecaster falling back to linear trend.")
        return forecast_linear_trend(history, horizon)

    y = np.array(history, dtype=float)
    n = len(y)

    # Build lagged feature matrix
    X_rows, y_rows = [], []
    for i in range(n_lags, n):
        X_rows.append(y[i - n_lags: i])
        y_rows.append(y[i])

    if len(X_rows) < 3:
        return forecast_linear_trend(history, horizon)

    X_arr = np.array(X_rows)
    y_arr = np.array(y_rows)

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_seed,
        verbosity=0,
    )
    model.fit(X_arr, y_arr)

    # Iterative forecasting
    window = list(y[-n_lags:])
    preds = []
    for _ in range(horizon):
        x_pred = np.array([window[-n_lags:]])
        p = float(model.predict(x_pred)[0])
        p = float(np.clip(p, 0.0, 1.0))
        preds.append(p)
        window.append(p)

    return np.array(preds), {"method": "xgboost", "n_lags": n_lags}


# ---------------------------------------------------------------------------
# Main forecast dispatcher
# ---------------------------------------------------------------------------

def forecast_reliability(
    history: np.ndarray,
    horizon: int,
    method: str = "linear_trend",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Forecast future reliability values.

    Parameters
    ----------
    history    : array of observed reliability values (e.g., window F1 scores)
    horizon    : number of future windows to forecast
    method     : 'persistence', 'linear_trend', 'xgboost'
    random_seed: reproducibility

    Returns
    -------
    dict with:
        forecast_values (array)
        forecast_windows (array)
        method
        trend_metadata
        history_length
    """
    history = np.array(history, dtype=float)
    n = len(history)
    future_windows = np.arange(n, n + horizon)

    if method == "persistence":
        forecast_vals = forecast_persistence(history, horizon)
        meta = {"method": "persistence", "slope": 0.0}

    elif method == "xgboost" and _XGB_AVAILABLE:
        forecast_vals, meta = forecast_xgboost(history, horizon, random_seed=random_seed)

    else:
        if method not in ("linear_trend", "persistence") and method != "xgboost":
            logger.warning("Unknown method '%s'; using linear_trend.", method)
        forecast_vals, meta = forecast_linear_trend(history, horizon)

    return {
        "forecast_values": forecast_vals,
        "forecast_windows": future_windows,
        "method": meta.get("method", method),
        "trend_metadata": meta,
        "history_length": n,
        "horizon": horizon,
        "history_values": history,
        "history_windows": np.arange(n),
    }


# ---------------------------------------------------------------------------
# Threshold crossing (TTD)
# ---------------------------------------------------------------------------

def estimate_threshold_crossing(
    history: np.ndarray,
    forecast_values: np.ndarray,
    threshold: float,
    start_window_offset: int = 0,
) -> Dict[str, Any]:
    """
    Estimate the first window where reliability is predicted to cross below
    the configured threshold.

    IMPORTANT: This is an ESTIMATE. Forecasts are not guarantees.

    Parameters
    ----------
    history              : observed reliability history
    forecast_values      : predicted future reliability values
    threshold            : reliability threshold (e.g. 0.70)
    start_window_offset  : the actual window index of the first forecast step

    Returns
    -------
    dict with:
        crossing_detected (bool)
        crossing_window (int or None)
        warning_lead_windows (int)
        ttd_estimate (str)
        current_reliability (float)
        last_observed_window (int)
    """
    n_history = len(history)
    current = float(history[-1]) if n_history > 0 else 0.0
    already_below = current < threshold

    crossing_window = None
    for i, val in enumerate(forecast_values):
        if val < threshold:
            crossing_window = start_window_offset + i
            break

    # Warning lead time: how many forecast steps before crossing
    if crossing_window is not None:
        warn_lead = crossing_window - start_window_offset
    else:
        warn_lead = len(forecast_values)

    # TTD human-readable string
    if already_below:
        ttd_str = f"Already below threshold (current={current:.3f} < {threshold})"
    elif crossing_window is not None:
        ttd_str = (
            f"Estimated threshold crossing at window {crossing_window} "
            f"(~{warn_lead} window(s) from now) [ESTIMATE]"
        )
    else:
        ttd_str = (
            f"No threshold crossing within forecast horizon of {len(forecast_values)} windows "
            f"[ESTIMATE — not a guarantee]"
        )

    return {
        "crossing_detected": crossing_window is not None or already_below,
        "crossing_window": crossing_window,
        "already_below": already_below,
        "warning_lead_windows": int(warn_lead),
        "ttd_estimate": ttd_str,
        "current_reliability": round(current, 4),
        "last_observed_window": n_history - 1,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Forecast error calculation
# ---------------------------------------------------------------------------

def calculate_forecast_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate MAE and RMSE between actual and predicted reliability.

    Only valid when actual future values are known (e.g., retrospective evaluation).
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    min_len = min(len(actual), len(predicted))
    if min_len == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_compared": 0}

    actual = actual[:min_len]
    predicted = predicted[:min_len]

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_compared": min_len,
    }


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

def compare_forecasters(
    history: np.ndarray,
    horizon: int,
    actual_future: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run multiple forecast methods and compare their errors.

    If actual_future is provided, compute MAE and RMSE.

    Returns
    -------
    pd.DataFrame comparing methods
    """
    methods = ["persistence", "linear_trend"]
    if _XGB_AVAILABLE and len(history) >= 5:
        methods.append("xgboost")

    rows = []
    for method in methods:
        result = forecast_reliability(history, horizon, method=method, random_seed=random_seed)
        row = {
            "method": method,
            "forecast_final": round(float(result["forecast_values"][-1]), 4),
            "forecast_min": round(float(result["forecast_values"].min()), 4),
        }

        if actual_future is not None:
            errs = calculate_forecast_error(actual_future, result["forecast_values"])
            row.update(errs)
        else:
            row.update({"mae": float("nan"), "rmse": float("nan")})

        rows.append(row)

    return pd.DataFrame(rows)
