"""
test_monitoring.py — Tests for model reliability monitoring.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.monitoring import (
    build_reliability_timeline,
    calculate_health_score,
    calculate_reliability_degradation,
    calculate_time_below_threshold,
    evaluate_model_by_window,
)


# ---------------------------------------------------------------------------
# calculate_health_score
# ---------------------------------------------------------------------------

def test_health_score_at_baseline():
    score = calculate_health_score(current_metric=0.85, baseline_metric=0.85, threshold=0.70)
    assert score == pytest.approx(1.0, abs=0.01)


def test_health_score_at_threshold():
    score = calculate_health_score(current_metric=0.70, baseline_metric=0.85, threshold=0.70)
    assert score == pytest.approx(0.0, abs=0.01)


def test_health_score_below_threshold():
    score = calculate_health_score(current_metric=0.50, baseline_metric=0.85, threshold=0.70)
    assert score == 0.0


def test_health_score_above_baseline():
    score = calculate_health_score(current_metric=0.90, baseline_metric=0.85, threshold=0.70)
    # Should clip to 1.0
    assert score >= 1.0 or score == pytest.approx(1.0, abs=0.1)


def test_health_score_same_baseline_threshold():
    # Edge case: baseline == threshold
    score = calculate_health_score(current_metric=0.70, baseline_metric=0.70, threshold=0.70)
    assert score in (0.0, 1.0)


# ---------------------------------------------------------------------------
# calculate_reliability_degradation
# ---------------------------------------------------------------------------

def test_degradation_no_change():
    timeline = pd.DataFrame({
        "window": [0, 1, 2],
        "f1": [0.80, 0.80, 0.80],
    })
    result = calculate_reliability_degradation(timeline, baseline_metric=0.80)
    assert result["degradation_pct"] == pytest.approx(0.0, abs=0.1)
    assert result["trend"] == "stable"


def test_degradation_declining():
    timeline = pd.DataFrame({
        "window": list(range(6)),
        "f1": [0.85, 0.82, 0.78, 0.74, 0.70, 0.65],
    })
    result = calculate_reliability_degradation(timeline, baseline_metric=0.85)
    assert result["degradation_pct"] > 0
    assert result["trend"] == "declining"
    assert result["current"] == pytest.approx(0.65, abs=0.01)


def test_degradation_empty_timeline():
    result = calculate_reliability_degradation(pd.DataFrame(), baseline_metric=0.80)
    assert result["baseline"] == 0.80
    assert result["trend"] == "stable"


# ---------------------------------------------------------------------------
# calculate_time_below_threshold
# ---------------------------------------------------------------------------

def test_time_below_threshold_none_below():
    timeline = pd.DataFrame({
        "window": [0, 1, 2, 3],
        "f1": [0.80, 0.82, 0.85, 0.83],
    })
    result = calculate_time_below_threshold(timeline, threshold=0.70)
    assert result["n_windows_below"] == 0
    assert result["first_crossing_window"] is None


def test_time_below_threshold_some_below():
    timeline = pd.DataFrame({
        "window": [0, 1, 2, 3, 4],
        "f1": [0.80, 0.75, 0.65, 0.60, 0.72],
    })
    result = calculate_time_below_threshold(timeline, threshold=0.70)
    assert result["n_windows_below"] == 2
    assert result["first_crossing_window"] == 2


def test_time_below_threshold_consecutive():
    timeline = pd.DataFrame({
        "window": [0, 1, 2, 3, 4],
        "f1": [0.60, 0.58, 0.55, 0.72, 0.80],
    })
    result = calculate_time_below_threshold(timeline, threshold=0.70)
    assert result["consecutive_below"] == 3


# ---------------------------------------------------------------------------
# build_reliability_timeline
# ---------------------------------------------------------------------------

def test_build_reliability_timeline():
    window_metrics = pd.DataFrame({
        "window": [0, 1, 2, 3],
        "n_rows": [100, 100, 100, 100],
        "accuracy": [0.82, 0.80, 0.75, 0.70],
        "precision": [0.80, 0.78, 0.73, 0.68],
        "recall": [0.80, 0.78, 0.73, 0.68],
        "f1": [0.80, 0.78, 0.73, 0.68],
        "primary_metric": [0.80, 0.78, 0.73, 0.68],
    })
    timeline = build_reliability_timeline(
        window_metrics, baseline_metric=0.82, threshold=0.70
    )
    assert "health_score" in timeline.columns
    assert "below_threshold" in timeline.columns
    assert "degradation_pct" in timeline.columns
    assert len(timeline) == 4
    # Window 3 (f1=0.68) should be below threshold
    assert timeline.iloc[3]["below_threshold"] == True


def test_build_timeline_with_drift():
    window_metrics = pd.DataFrame({
        "window": [0, 1, 2],
        "n_rows": [100, 100, 100],
        "accuracy": [0.80, 0.77, 0.74],
        "precision": [0.80, 0.77, 0.74],
        "recall": [0.80, 0.77, 0.74],
        "f1": [0.80, 0.77, 0.74],
        "primary_metric": [0.80, 0.77, 0.74],
    })
    drift_df = pd.DataFrame({
        "window": [0, 1, 2],
        "drift_severity_index": [0.0, 0.1, 0.5],
        "severity_label": ["none", "low", "moderate"],
    })
    timeline = build_reliability_timeline(
        window_metrics, baseline_metric=0.80, threshold=0.70,
        drift_severity_table=drift_df,
    )
    assert "drift_severity_index" in timeline.columns
