"""
test_recovery.py — Tests for recovery strategy engine.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.recovery import (
    STRATEGY_HYBRID,
    STRATEGY_NONE,
    STRATEGY_REAL,
    STRATEGY_SYNTHETIC,
    compare_recovery_strategies,
    run_hybrid_recovery,
    run_no_recovery,
    run_real_only_recovery,
    run_synthetic_only_recovery,
)
from src.datadecay.preprocessing import (
    chronological_split,
    encode_target,
    fit_preprocessor,
    prepare_features,
    sort_by_time,
)
from src.datadecay.models import create_model, train_model


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(42)
    n = 800
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = (X1 + X2 + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({
        "feat_a": X1, "feat_b": X2, "target": y
    })
    return df


NUM_COLS = ["feat_a", "feat_b"]
CAT_COLS = []
TARGET = "target"


@pytest.fixture
def trained_pipeline(classification_df):
    split = chronological_split(classification_df, n_windows=8, test_window_count=2)
    ref = split["reference_df"]
    future = split["future_df"]

    X_ref, y_ref = prepare_features(ref, TARGET, NUM_COLS, CAT_COLS)
    preprocessor, X_trans = fit_preprocessor(X_ref, NUM_COLS, CAT_COLS)
    y_enc, le = encode_target(y_ref)

    clf = create_model("logistic_regression", random_seed=42)
    clf, _ = train_model(clf, X_trans.values, y_enc)

    return {
        "model": clf,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "ref_df": ref,
        "future_df": future,
        "full_df": classification_df,
    }


# ---------------------------------------------------------------------------
# No recovery
# ---------------------------------------------------------------------------

def test_no_recovery_returns_future_metrics(trained_pipeline):
    result = run_no_recovery(
        baseline_model=trained_pipeline["model"],
        future_df=trained_pipeline["future_df"],
        target_column=TARGET,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        preprocessor=trained_pipeline["preprocessor"],
        label_encoder=trained_pipeline["label_encoder"],
    )
    assert result["strategy"] == STRATEGY_NONE
    assert "future_metrics" in result
    assert result["future_metrics"]["f1"] >= 0.0


# ---------------------------------------------------------------------------
# Real only recovery
# ---------------------------------------------------------------------------

def test_real_only_recovery(trained_pipeline):
    result = run_real_only_recovery(
        baseline_model=trained_pipeline["model"],
        recent_real_df=trained_pipeline["ref_df"],
        future_df=trained_pipeline["future_df"],
        target_column=TARGET,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        model_name="logistic_regression",
        random_seed=42,
    )
    assert result["strategy"] == STRATEGY_REAL
    assert result["future_metrics"]["f1"] >= 0.0
    assert result["n_real"] > 0
    assert result["n_synthetic"] == 0


# ---------------------------------------------------------------------------
# Synthetic only recovery
# ---------------------------------------------------------------------------

def test_synthetic_only_recovery(trained_pipeline):
    # Build simple synthetic data
    rng = np.random.default_rng(7)
    n = 200
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = (X1 + X2 > 0).astype(int)
    synthetic = pd.DataFrame({"feat_a": X1, "feat_b": X2, "target": y})

    result = run_synthetic_only_recovery(
        baseline_model=trained_pipeline["model"],
        synthetic_df=synthetic,
        future_df=trained_pipeline["future_df"],
        target_column=TARGET,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        model_name="logistic_regression",
        random_seed=42,
    )
    assert result["strategy"] == STRATEGY_SYNTHETIC
    assert result["n_synthetic"] == 200
    assert result["n_real"] == 0


# ---------------------------------------------------------------------------
# Hybrid recovery
# ---------------------------------------------------------------------------

def test_hybrid_recovery(trained_pipeline):
    rng = np.random.default_rng(8)
    n = 150
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = (X1 + X2 > 0).astype(int)
    synthetic = pd.DataFrame({"feat_a": X1, "feat_b": X2, "target": y})

    result = run_hybrid_recovery(
        baseline_model=trained_pipeline["model"],
        recent_real_df=trained_pipeline["ref_df"],
        synthetic_df=synthetic,
        future_df=trained_pipeline["future_df"],
        target_column=TARGET,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        model_name="logistic_regression",
        random_seed=42,
    )
    assert result["strategy"] == STRATEGY_HYBRID
    assert result["n_real"] > 0
    assert result["n_synthetic"] == 150
    assert result["intervention_size"] == result["n_real"] + result["n_synthetic"]


# ---------------------------------------------------------------------------
# compare_recovery_strategies
# ---------------------------------------------------------------------------

def test_compare_strategies():
    fake_results = {
        STRATEGY_NONE: {
            "strategy": STRATEGY_NONE,
            "future_metrics": {"f1": 0.70, "accuracy": 0.72},
            "training_time_s": 0,
            "n_synthetic": 0, "n_real": 0, "intervention_size": 0,
        },
        STRATEGY_REAL: {
            "strategy": STRATEGY_REAL,
            "future_metrics": {"f1": 0.75, "accuracy": 0.76},
            "training_time_s": 0.5,
            "n_synthetic": 0, "n_real": 300, "intervention_size": 300,
        },
        STRATEGY_SYNTHETIC: {
            "strategy": STRATEGY_SYNTHETIC,
            "future_metrics": {"f1": 0.72, "accuracy": 0.73},
            "training_time_s": 0.3,
            "n_synthetic": 500, "n_real": 0, "intervention_size": 500,
        },
    }
    df = compare_recovery_strategies(fake_results, baseline_future_f1=0.70)
    assert len(df) == 3
    assert "recovery_gain" in df.columns
    assert "decision" in df.columns
    # Best strategy should be sorted first
    assert df.iloc[0]["future_f1"] >= df.iloc[1]["future_f1"]

def test_recovery_gain_calculation():
    fake_results = {
        STRATEGY_NONE: {
            "strategy": STRATEGY_NONE,
            "future_metrics": {"f1": 0.70, "accuracy": 0.72},
            "training_time_s": 0, "n_synthetic": 0, "n_real": 0, "intervention_size": 0,
        },
        STRATEGY_REAL: {
            "strategy": STRATEGY_REAL,
            "future_metrics": {"f1": 0.80, "accuracy": 0.81},
            "training_time_s": 0.2, "n_synthetic": 0, "n_real": 200, "intervention_size": 200,
        },
    }
    df = compare_recovery_strategies(fake_results, baseline_future_f1=0.70)
    real_row = df[df["strategy"] == STRATEGY_REAL].iloc[0]
    assert real_row["recovery_gain"] == pytest.approx(0.10, abs=0.001)
    assert real_row["decision"] == "ACCEPT"
