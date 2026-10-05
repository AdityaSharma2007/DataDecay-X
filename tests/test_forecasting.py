"""
test_forecasting.py — Tests for reliability forecasting.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.forecasting import (
    calculate_forecast_error,
    compare_forecasters,
    estimate_threshold_crossing,
    forecast_linear_trend,
    forecast_persistence,
    forecast_reliability,
)


# ---------------------------------------------------------------------------
# Persistence forecast
# ---------------------------------------------------------------------------

def test_persistence_flat():
    history = np.array([0.80, 0.80, 0.80])
    result = forecast_persistence(history, horizon=3)
    assert len(result) == 3
    assert all(v == pytest.approx(0.80) for v in result)


def test_persistence_uses_last():
    history = np.array([0.90, 0.85, 0.70])
    result = forecast_persistence(history, horizon=2)
    assert all(v == pytest.approx(0.70) for v in result)


def test_persistence_empty():
    result = forecast_persistence(np.array([]), horizon=3)
    assert len(result) == 3
    assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# Linear trend forecast
# ---------------------------------------------------------------------------

def test_linear_trend_declining():
    history = np.array([0.85, 0.82, 0.78, 0.74, 0.70])
    forecast, meta = forecast_linear_trend(history, horizon=3)
    assert len(forecast) == 3
    # Declining trend: forecasted values should be below last observed
    assert forecast[-1] < history[-1] or forecast[-1] < history[0]


def test_linear_trend_stable():
    history = np.array([0.80, 0.80, 0.80, 0.80, 0.80])
    forecast, meta = forecast_linear_trend(history, horizon=3)
    assert len(forecast) == 3
    assert all(abs(v - 0.80) < 0.05 for v in forecast)


def test_linear_trend_metadata():
    history = np.array([0.80, 0.78, 0.76, 0.74, 0.72])
    _, meta = forecast_linear_trend(history, horizon=2)
    assert "slope" in meta
    assert meta["slope"] < 0  # negative slope for declining trend


def test_linear_trend_clipped():
    # Values should be clipped to [0, 1]
    history = np.array([0.10, 0.05, 0.02])
    forecast, _ = forecast_linear_trend(history, horizon=5)
    assert all(0.0 <= v <= 1.0 for v in forecast)


def test_linear_trend_insufficient_data():
    # Falls back to persistence
    history = np.array([0.75])
    forecast, meta = forecast_linear_trend(history, horizon=2)
    assert len(forecast) == 2
    assert meta["method"] == "persistence_fallback"


# ---------------------------------------------------------------------------
# forecast_reliability dispatcher
# ---------------------------------------------------------------------------

def test_forecast_reliability_persistence():
    history = np.array([0.82, 0.80, 0.78])
    result = forecast_reliability(history, horizon=3, method="persistence")
    assert result["method"] == "persistence"
    assert len(result["forecast_values"]) == 3


def test_forecast_reliability_linear():
    history = np.array([0.85, 0.82, 0.79, 0.76])
    result = forecast_reliability(history, horizon=3, method="linear_trend")
    assert "forecast_values" in result
    assert len(result["forecast_values"]) == 3
    assert "forecast_windows" in result
    assert result["forecast_windows"][0] == 4  # starts after last observed


# ---------------------------------------------------------------------------
# estimate_threshold_crossing
# ---------------------------------------------------------------------------

def test_no_crossing():
    history = np.array([0.85, 0.84, 0.83, 0.82])
    forecast = np.array([0.82, 0.81, 0.80])
    result = estimate_threshold_crossing(history, forecast, threshold=0.70)
    assert result["crossing_detected"] == False
    assert result["crossing_window"] is None
    assert "No threshold crossing" in result["ttd_estimate"]


def test_crossing_detected():
    history = np.array([0.80, 0.76, 0.72])
    forecast = np.array([0.69, 0.65, 0.60])
    result = estimate_threshold_crossing(
        history, forecast, threshold=0.70, start_window_offset=3
    )
    assert result["crossing_detected"] == True
    assert result["crossing_window"] == 3
    assert "ESTIMATE" in result["ttd_estimate"]


def test_already_below_threshold():
    history = np.array([0.80, 0.75, 0.65])
    forecast = np.array([0.60, 0.55, 0.50])
    result = estimate_threshold_crossing(history, forecast, threshold=0.70)
    assert result["already_below"] == True


# ---------------------------------------------------------------------------
# calculate_forecast_error
# ---------------------------------------------------------------------------

def test_forecast_error():
    actual = np.array([0.80, 0.77, 0.74])
    predicted = np.array([0.79, 0.76, 0.73])
    result = calculate_forecast_error(actual, predicted)
    assert result["mae"] == pytest.approx(0.01, abs=0.001)
    assert result["rmse"] >= result["mae"]


def test_forecast_error_empty():
    result = calculate_forecast_error(np.array([]), np.array([]))
    assert result["n_compared"] == 0


# ---------------------------------------------------------------------------
# compare_forecasters
# ---------------------------------------------------------------------------

def test_compare_forecasters():
    history = np.array([0.85, 0.82, 0.79, 0.76, 0.73])
    df = compare_forecasters(history, horizon=3)
    assert len(df) >= 2  # at least persistence and linear_trend
    assert "method" in df.columns
    assert "forecast_final" in df.columns
