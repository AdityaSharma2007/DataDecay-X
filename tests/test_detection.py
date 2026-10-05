"""
test_detection.py — Tests for drift detection (PSI, KS, Wasserstein).
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.detection import (
    aggregate_drift_severity,
    calculate_ks,
    calculate_psi,
    calculate_psi_categorical,
    calculate_wasserstein,
    detect_drift_across_windows,
    detect_feature_drift,
)


@pytest.fixture
def reference_array():
    rng = np.random.default_rng(42)
    return rng.normal(50, 10, 500)


@pytest.fixture
def identical_array(reference_array):
    return reference_array.copy()


@pytest.fixture
def shifted_array():
    rng = np.random.default_rng(42)
    return rng.normal(80, 10, 500)  # large mean shift


@pytest.fixture
def ref_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "num_a": rng.normal(0, 1, 300),
        "num_b": rng.uniform(0, 10, 300),
        "cat_x": rng.choice(["A", "B", "C"], 300),
    })


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def test_psi_identical_distributions(reference_array, identical_array):
    psi = calculate_psi(reference_array, identical_array)
    assert psi < 0.01, f"PSI for identical distributions should be near 0, got {psi}"


def test_psi_large_shift(reference_array, shifted_array):
    psi = calculate_psi(reference_array, shifted_array)
    assert psi > 0.20, f"PSI for large shift should exceed 0.20, got {psi}"


def test_psi_non_negative(reference_array):
    rng = np.random.default_rng(5)
    cur = rng.normal(55, 12, 500)
    psi = calculate_psi(reference_array, cur)
    assert psi >= 0.0


def test_psi_empty_arrays():
    psi = calculate_psi(np.array([]), np.array([]))
    assert psi == 0.0


def test_psi_categorical():
    ref = np.array(["A", "A", "B", "C"] * 100)
    cur_same = np.array(["A", "A", "B", "C"] * 100)
    cur_diff = np.array(["C", "C", "C", "C"] * 100)
    assert calculate_psi_categorical(ref, cur_same) < 0.05
    assert calculate_psi_categorical(ref, cur_diff) > 0.10


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------

def test_ks_identical(reference_array, identical_array):
    ks_stat, p_val = calculate_ks(reference_array, identical_array)
    assert ks_stat < 0.05
    assert p_val > 0.05  # not significant


def test_ks_shifted(reference_array, shifted_array):
    ks_stat, p_val = calculate_ks(reference_array, shifted_array)
    assert ks_stat > 0.5
    assert p_val < 0.05  # significant


def test_ks_insufficient_data():
    ks_stat, p_val = calculate_ks(np.array([1, 2]), np.array([1, 2, 3]))
    assert ks_stat == 0.0
    assert p_val == 1.0


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def test_wasserstein_identical(reference_array, identical_array):
    w = calculate_wasserstein(reference_array, identical_array)
    assert w < 0.01


def test_wasserstein_shifted(reference_array, shifted_array):
    w = calculate_wasserstein(reference_array, shifted_array)
    assert w > 1.0  # normalised shift should be large


def test_wasserstein_non_negative(reference_array):
    rng = np.random.default_rng(7)
    cur = rng.normal(52, 10, 500)
    w = calculate_wasserstein(reference_array, cur)
    assert w >= 0.0


# ---------------------------------------------------------------------------
# detect_feature_drift
# ---------------------------------------------------------------------------

def test_detect_feature_drift_stable(ref_df):
    result = detect_feature_drift(
        reference_df=ref_df,
        current_df=ref_df.copy(),
        numerical_cols=["num_a", "num_b"],
        categorical_cols=["cat_x"],
    )
    assert not result.empty
    assert set(["feature", "psi", "ks_statistic", "severity"]).issubset(result.columns)
    # Identical data → all near zero PSI
    assert result["psi"].max() < 0.05


def test_detect_feature_drift_shifted(ref_df):
    rng = np.random.default_rng(10)
    drifted = ref_df.copy()
    drifted["num_a"] = drifted["num_a"] + 5.0  # add large shift
    result = detect_feature_drift(
        reference_df=ref_df,
        current_df=drifted,
        numerical_cols=["num_a", "num_b"],
        categorical_cols=[],
    )
    num_a_row = result[result["feature"] == "num_a"].iloc[0]
    assert num_a_row["psi"] > 0.10 or num_a_row["ks_drift"] == True


# ---------------------------------------------------------------------------
# aggregate_drift_severity
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    result = aggregate_drift_severity(pd.DataFrame())
    assert result["drift_severity_index"] == 0.0
    assert result["severity_label"] == "none"


def test_aggregate_severe():
    df = pd.DataFrame([
        {"feature": "a", "psi": 0.5, "ks_statistic": 0.8, "ks_drift": True,
         "wasserstein": 0.5, "severity": "severe"},
        {"feature": "b", "psi": 0.3, "ks_statistic": 0.6, "ks_drift": True,
         "wasserstein": 0.3, "severity": "severe"},
    ])
    result = aggregate_drift_severity(df)
    assert result["drift_severity_index"] > 0.4
    assert result["n_drifted_features"] == 2


# ---------------------------------------------------------------------------
# detect_drift_across_windows
# ---------------------------------------------------------------------------

def test_detect_across_windows(ref_df):
    windows = [ref_df.iloc[:50].copy(), ref_df.iloc[50:100].copy(), ref_df.iloc[100:].copy()]
    result = detect_drift_across_windows(
        reference_df=ref_df,
        windows=windows,
        numerical_cols=["num_a", "num_b"],
        categorical_cols=["cat_x"],
    )
    assert "window_drift_table" in result
    assert len(result["window_drift_table"]) == 3
    assert "overall_severity" in result
