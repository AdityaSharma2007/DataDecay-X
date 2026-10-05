"""
test_simulator.py — Tests for the temporal drift simulation module.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.simulator import (
    DRIFT_ABRUPT,
    DRIFT_GRADUAL,
    DRIFT_RECURRING,
    DRIFT_STABLE,
    build_drift_schedule,
    create_temporal_windows,
    simulate_drift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """Small synthetic tabular dataset for simulation tests."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="D"),
        "feat_a": rng.normal(50, 10, n),
        "feat_b": rng.normal(100, 20, n),
        "category": rng.choice(["X", "Y", "Z"], size=n),
        "target": rng.integers(0, 2, size=n),
    })
    # Assign windows
    window_size = n // 12
    df["__ddx_window"] = (np.arange(n) // window_size).clip(0, 11)
    return df


# ---------------------------------------------------------------------------
# build_drift_schedule
# ---------------------------------------------------------------------------

def test_build_schedule_stable():
    sched = build_drift_schedule(12, DRIFT_STABLE, 4, 4, 2.0)
    assert all(v == 0.0 for v in sched)
    assert len(sched) == 12


def test_build_schedule_abrupt():
    sched = build_drift_schedule(12, DRIFT_ABRUPT, 4, 4, 2.0)
    assert sched[3] == 0.0
    for w in range(4, 8):
        assert sched[w] == 2.0
    assert sched[8] == 0.0


def test_build_schedule_gradual():
    sched = build_drift_schedule(12, DRIFT_GRADUAL, 4, 4, 2.0)
    assert sched[3] == 0.0
    assert sched[4] > 0.0
    assert sched[7] == pytest.approx(2.0, abs=0.01)
    # Gradual: monotonically increasing in drift region
    for i in range(4, 7):
        assert sched[i] <= sched[i + 1]


def test_build_schedule_recurring():
    sched = build_drift_schedule(12, DRIFT_RECURRING, 2, 4, 1.5)
    # First burst
    assert sched[2] == 1.5
    # Second burst
    assert sched[6] == 1.5
    assert len(sched) == 12


# ---------------------------------------------------------------------------
# simulate_drift — stable scenario
# ---------------------------------------------------------------------------

def test_simulate_stable(simple_df):
    result = simulate_drift(
        df=simple_df,
        numerical_cols=["feat_a", "feat_b"],
        categorical_cols=["category"],
        n_windows=12,
        drift_type=DRIFT_STABLE,
        start_window=4,
        duration=4,
        magnitude=2.0,
        random_seed=42,
    )
    original = simple_df["feat_a"].values
    drifted = result["drifted_df"]["feat_a"].values
    # Stable: no change applied
    np.testing.assert_array_almost_equal(original, drifted)


# ---------------------------------------------------------------------------
# simulate_drift — gradual scenario
# ---------------------------------------------------------------------------

def test_simulate_gradual_shifts_features(simple_df):
    result = simulate_drift(
        df=simple_df,
        numerical_cols=["feat_a", "feat_b"],
        categorical_cols=[],
        n_windows=12,
        drift_type=DRIFT_GRADUAL,
        start_window=4,
        duration=4,
        magnitude=3.0,
        random_seed=42,
    )
    gt = result["ground_truth"]
    drifted = result["drifted_df"]

    # Drift ground truth should capture the type
    assert gt["drift_type"] == DRIFT_GRADUAL
    assert gt["start_window"] == 4

    # At least one affected feature should differ after drift
    orig_mean = simple_df["feat_a"].mean()
    drifted_win6 = drifted[drifted["__ddx_window"] == 6]["feat_a"].mean()
    assert drifted_win6 != orig_mean


# ---------------------------------------------------------------------------
# simulate_drift — abrupt scenario
# ---------------------------------------------------------------------------

def test_simulate_abrupt(simple_df):
    result = simulate_drift(
        df=simple_df,
        numerical_cols=["feat_a"],
        categorical_cols=[],
        n_windows=12,
        drift_type=DRIFT_ABRUPT,
        start_window=5,
        duration=3,
        magnitude=5.0,
        random_seed=42,
    )
    gt = result["ground_truth"]
    assert gt["drift_type"] == DRIFT_ABRUPT
    assert gt["magnitude"] == 5.0

    drifted = result["drifted_df"]
    pre_drift_mean = drifted[drifted["__ddx_window"] < 5]["feat_a"].mean()
    during_drift_mean = drifted[drifted["__ddx_window"] == 6]["feat_a"].mean()
    assert during_drift_mean > pre_drift_mean  # mean should shift up


# ---------------------------------------------------------------------------
# simulate_drift — recurring scenario
# ---------------------------------------------------------------------------

def test_simulate_recurring(simple_df):
    result = simulate_drift(
        df=simple_df,
        numerical_cols=["feat_a"],
        categorical_cols=[],
        n_windows=12,
        drift_type=DRIFT_RECURRING,
        start_window=2,
        duration=4,
        magnitude=2.0,
        random_seed=42,
    )
    gt = result["ground_truth"]
    assert gt["drift_type"] == DRIFT_RECURRING
    # Window metadata should have at least some drifted windows
    drifted_meta = [m for m in result["window_metadata"] if m["is_drifted"]]
    assert len(drifted_meta) > 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(simple_df):
    kwargs = dict(
        df=simple_df,
        numerical_cols=["feat_a", "feat_b"],
        categorical_cols=["category"],
        n_windows=12,
        drift_type=DRIFT_GRADUAL,
        start_window=4,
        duration=4,
        magnitude=2.0,
        random_seed=99,
    )
    r1 = simulate_drift(**kwargs)
    r2 = simulate_drift(**kwargs)
    pd.testing.assert_frame_equal(
        r1["drifted_df"].reset_index(drop=True),
        r2["drifted_df"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# create_temporal_windows
# ---------------------------------------------------------------------------

def test_create_temporal_windows(simple_df):
    windows = create_temporal_windows(simple_df, n_windows=12)
    assert len(windows) == 12
    total = sum(len(w) for w in windows)
    assert total == len(simple_df)


def test_window_ordering_preserved(simple_df):
    windows = create_temporal_windows(simple_df, n_windows=6)
    # Each successive window should have higher or equal __ddx_window values
    for i, w in enumerate(windows):
        assert (w["__ddx_window"] == i).all()
