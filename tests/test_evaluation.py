"""
test_evaluation.py — Tests for independent future evaluation.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.evaluation import (
    build_evaluation_summary,
    calculate_recovery_gain,
    compare_baseline_candidate,
    evaluate_future_window,
)
from src.datadecay.preprocessing import (
    chronological_split,
    encode_target,
    fit_preprocessor,
    prepare_features,
)
from src.datadecay.models import create_model, train_model


@pytest.fixture
def pipeline_components():
    rng = np.random.default_rng(42)
    n = 600
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = (X1 + X2 + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"feat_a": X1, "feat_b": X2, "target": y})

    split = chronological_split(df, n_windows=8, test_window_count=2)
    ref = split["reference_df"]
    future = split["future_df"]

    X_ref, y_ref = prepare_features(ref, "target", ["feat_a", "feat_b"], [])
    preprocessor, X_trans = fit_preprocessor(X_ref, ["feat_a", "feat_b"], [])
    y_enc, le = encode_target(y_ref)

    clf = create_model("logistic_regression", random_seed=42)
    clf, _ = train_model(clf, X_trans.values, y_enc)

    return {
        "model": clf,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "future_df": future,
    }


# ---------------------------------------------------------------------------
# evaluate_future_window
# ---------------------------------------------------------------------------

def test_future_window_returns_metrics(pipeline_components):
    result = evaluate_future_window(
        model=pipeline_components["model"],
        future_df=pipeline_components["future_df"],
        target_column="target",
        numerical_cols=["feat_a", "feat_b"],
        categorical_cols=[],
        preprocessor=pipeline_components["preprocessor"],
        label_encoder=pipeline_components["label_encoder"],
    )
    assert "f1" in result
    assert "accuracy" in result
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0


def test_future_window_none_model(pipeline_components):
    result = evaluate_future_window(
        model=None,
        future_df=pipeline_components["future_df"],
        target_column="target",
        numerical_cols=["feat_a", "feat_b"],
        categorical_cols=[],
        preprocessor=pipeline_components["preprocessor"],
        label_encoder=pipeline_components["label_encoder"],
    )
    assert result["f1"] == 0.0
    assert "error" in result


# ---------------------------------------------------------------------------
# compare_baseline_candidate
# ---------------------------------------------------------------------------

def test_promote_when_better():
    baseline = {"f1": 0.70}
    candidate = {"f1": 0.80}
    result = compare_baseline_candidate(baseline, candidate, candidate_name="test")
    assert result["decision"] == "PROMOTE_CANDIDATE"
    assert result["recovery_gain"] == pytest.approx(0.10, abs=0.001)


def test_retain_when_worse():
    baseline = {"f1": 0.80}
    candidate = {"f1": 0.70}
    result = compare_baseline_candidate(baseline, candidate, candidate_name="test")
    assert result["decision"] == "RETAIN_BASELINE"
    assert result["recovery_gain"] < 0


def test_neutral_when_equal():
    baseline = {"f1": 0.75}
    candidate = {"f1": 0.75}
    result = compare_baseline_candidate(baseline, candidate, candidate_name="test")
    assert result["decision"] in ("NEUTRAL", "PROMOTE_CANDIDATE")


# ---------------------------------------------------------------------------
# calculate_recovery_gain
# ---------------------------------------------------------------------------

def test_recovery_gain_positive():
    result = calculate_recovery_gain(0.70, 0.80)
    assert result["recovery_gain"] == pytest.approx(0.10, abs=0.001)
    assert result["remaining_degradation"] == 0.0


def test_recovery_gain_negative():
    result = calculate_recovery_gain(0.80, 0.70)
    assert result["recovery_gain"] == pytest.approx(-0.10, abs=0.001)
    assert result["remaining_degradation"] == pytest.approx(0.10, abs=0.001)


# ---------------------------------------------------------------------------
# build_evaluation_summary
# ---------------------------------------------------------------------------

def test_build_evaluation_summary_recovery_successful():
    baseline_train = {"f1": 0.85}
    baseline_future = {"f1": 0.70}
    strategies = {
        "no_recovery": {
            "future_metrics": {"f1": 0.70, "accuracy": 0.71},
            "training_metrics": {"f1": 0.85},
            "training_time_s": 0,
            "n_synthetic": 0, "n_real": 0,
        },
        "real_only": {
            "future_metrics": {"f1": 0.80, "accuracy": 0.81},
            "training_metrics": {"f1": 0.84},
            "training_time_s": 0.5,
            "n_synthetic": 0, "n_real": 300,
        },
    }
    summary = build_evaluation_summary(baseline_train, baseline_future, strategies)
    assert summary["final_decision"] == "RECOVERY_SUCCESSFUL"
    assert summary["best_strategy"] == "real_only"
    assert summary["overall_recovery_gain"] > 0


def test_build_evaluation_summary_no_improvement():
    baseline_train = {"f1": 0.85}
    baseline_future = {"f1": 0.70}
    strategies = {
        "no_recovery": {
            "future_metrics": {"f1": 0.70, "accuracy": 0.71},
            "training_metrics": {"f1": 0.85},
            "training_time_s": 0,
            "n_synthetic": 0, "n_real": 0,
        },
        "real_only": {
            "future_metrics": {"f1": 0.695, "accuracy": 0.71},
            "training_metrics": {"f1": 0.84},
            "training_time_s": 0.5,
            "n_synthetic": 0, "n_real": 200,
        },
    }
    summary = build_evaluation_summary(baseline_train, baseline_future, strategies)
    assert summary["final_decision"] in ("NO_IMPROVEMENT", "RECOVERY_FAILED")


def test_build_evaluation_summary_negative_result():
    baseline_train = {"f1": 0.85}
    baseline_future = {"f1": 0.72}
    strategies = {
        "no_recovery": {
            "future_metrics": {"f1": 0.72, "accuracy": 0.73},
            "training_metrics": {"f1": 0.85},
            "training_time_s": 0,
            "n_synthetic": 0, "n_real": 0,
        },
        "synthetic_only": {
            "future_metrics": {"f1": 0.50, "accuracy": 0.52},
            "training_metrics": {"f1": 0.88},  # high training, low future
            "training_time_s": 0.3,
            "n_synthetic": 500, "n_real": 0,
        },
    }
    summary = build_evaluation_summary(baseline_train, baseline_future, strategies)
    # Negative results must be shown honestly
    syn_row = next(
        (s for s in summary["strategy_summaries"] if s["strategy"] == "synthetic_only"),
        None,
    )
    assert syn_row is not None
    assert syn_row["future_f1"] == pytest.approx(0.50, abs=0.001)
