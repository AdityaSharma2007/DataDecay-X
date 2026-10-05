"""
recovery.py — Recovery strategy engine for DataDecay-X.

Implements four recovery strategies:
  1. No Recovery         — use original baseline model unchanged
  2. Real Data Only      — retrain on recent real data
  3. Synthetic Data Only — retrain using generated synthetic data
  4. Hybrid              — retrain on combination of real + synthetic

IMPORTANT: Candidate models are evaluated on INDEPENDENT future data.
Training performance alone does NOT determine acceptance.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    create_model,
    evaluate_model,
    predict_model,
    train_model,
)
from .preprocessing import (
    fit_preprocessor,
    prepare_features,
    transform_with_preprocessor,
)

logger = logging.getLogger(__name__)

# Strategy name constants
STRATEGY_NONE = "no_recovery"
STRATEGY_REAL = "real_only"
STRATEGY_SYNTHETIC = "synthetic_only"
STRATEGY_HYBRID = "hybrid"

ALL_STRATEGIES = [STRATEGY_NONE, STRATEGY_REAL, STRATEGY_SYNTHETIC, STRATEGY_HYBRID]


# ---------------------------------------------------------------------------
# Individual recovery strategies
# ---------------------------------------------------------------------------

def run_no_recovery(
    baseline_model: Any,
    future_df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    preprocessor: Any,
    label_encoder: Any,
) -> Dict[str, Any]:
    """
    No Recovery: evaluate the existing baseline model on future data.

    Returns performance metrics of the unmodified baseline.
    """
    t0 = time.time()

    X_fut, y_fut = prepare_features(
        future_df, target_column, numerical_cols, categorical_cols
    )
    X_fut_trans = transform_with_preprocessor(X_fut, preprocessor, numerical_cols, categorical_cols)

    y_enc = _encode_target(y_fut, label_encoder)
    valid_mask = y_enc != -1
    if valid_mask.sum() == 0:
        return _empty_result(STRATEGY_NONE)

    y_pred = predict_model(baseline_model, X_fut_trans.values[valid_mask])
    metrics = evaluate_model(y_enc[valid_mask], y_pred)
    elapsed = time.time() - t0

    return {
        "strategy": STRATEGY_NONE,
        "model": baseline_model,
        "future_metrics": metrics,
        "training_time_s": 0.0,
        "intervention_size": 0,
        "n_synthetic": 0,
        "n_real": 0,
        "total_time_s": round(elapsed, 4),
        "note": "Baseline model — no retraining performed.",
    }


def run_real_only_recovery(
    baseline_model: Any,
    recent_real_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    model_name: str = "logistic_regression",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Real Data Only: retrain a new candidate model on recent real data.

    Evaluates candidate on future data to determine acceptance.
    """
    t0 = time.time()

    try:
        X_tr, y_tr = prepare_features(
            recent_real_df, target_column, numerical_cols, categorical_cols
        )

        if len(X_tr) == 0:
            return _empty_result(STRATEGY_REAL, error="No real data available for training.")

        # Fit preprocessor on this training data
        new_preprocessor, X_tr_trans = fit_preprocessor(X_tr, numerical_cols, categorical_cols)
        y_tr_enc, le = _fit_encode_target(y_tr)

        clf = create_model(model_name, random_seed=random_seed)
        clf, train_time = train_model(clf, X_tr_trans.values, y_tr_enc)

        # Training metrics (not used for acceptance decision)
        y_tr_pred = predict_model(clf, X_tr_trans.values)
        train_metrics = evaluate_model(y_tr_enc, y_tr_pred)

        # Future evaluation
        X_fut, y_fut = prepare_features(
            future_df, target_column, numerical_cols, categorical_cols
        )
        X_fut_trans = transform_with_preprocessor(X_fut, new_preprocessor, numerical_cols, categorical_cols)
        y_fut_enc = _encode_target(y_fut, le)
        valid_mask = y_fut_enc != -1

        if valid_mask.sum() == 0:
            return _empty_result(STRATEGY_REAL, error="No valid future data for evaluation.")

        y_fut_pred = predict_model(clf, X_fut_trans.values[valid_mask])
        future_metrics = evaluate_model(y_fut_enc[valid_mask], y_fut_pred)

        elapsed = time.time() - t0

        return {
            "strategy": STRATEGY_REAL,
            "model": clf,
            "preprocessor": new_preprocessor,
            "label_encoder": le,
            "training_metrics": train_metrics,
            "future_metrics": future_metrics,
            "training_time_s": round(train_time, 4),
            "intervention_size": len(recent_real_df),
            "n_real": len(recent_real_df),
            "n_synthetic": 0,
            "total_time_s": round(elapsed, 4),
        }

    except Exception as exc:
        logger.error("Real-only recovery failed: %s", exc)
        return _empty_result(STRATEGY_REAL, error=str(exc))


def run_synthetic_only_recovery(
    baseline_model: Any,
    synthetic_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    model_name: str = "logistic_regression",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Synthetic Data Only: retrain on generated synthetic data.

    IMPORTANT: Synthetic data must have passed the quality gate before this is called.
    Evaluates on independent future data.
    """
    t0 = time.time()

    try:
        X_syn, y_syn = prepare_features(
            synthetic_df, target_column, numerical_cols, categorical_cols
        )

        if len(X_syn) == 0:
            return _empty_result(STRATEGY_SYNTHETIC, error="Synthetic data is empty.")

        new_preprocessor, X_syn_trans = fit_preprocessor(X_syn, numerical_cols, categorical_cols)
        y_syn_enc, le = _fit_encode_target(y_syn)

        clf = create_model(model_name, random_seed=random_seed)
        clf, train_time = train_model(clf, X_syn_trans.values, y_syn_enc)

        # Training metrics
        y_syn_pred = predict_model(clf, X_syn_trans.values)
        train_metrics = evaluate_model(y_syn_enc, y_syn_pred)

        # Future evaluation
        X_fut, y_fut = prepare_features(
            future_df, target_column, numerical_cols, categorical_cols
        )
        X_fut_trans = transform_with_preprocessor(X_fut, new_preprocessor, numerical_cols, categorical_cols)
        y_fut_enc = _encode_target(y_fut, le)
        valid_mask = y_fut_enc != -1

        if valid_mask.sum() == 0:
            return _empty_result(STRATEGY_SYNTHETIC, error="No valid future data.")

        y_fut_pred = predict_model(clf, X_fut_trans.values[valid_mask])
        future_metrics = evaluate_model(y_fut_enc[valid_mask], y_fut_pred)

        elapsed = time.time() - t0

        return {
            "strategy": STRATEGY_SYNTHETIC,
            "model": clf,
            "preprocessor": new_preprocessor,
            "label_encoder": le,
            "training_metrics": train_metrics,
            "future_metrics": future_metrics,
            "training_time_s": round(train_time, 4),
            "intervention_size": len(synthetic_df),
            "n_real": 0,
            "n_synthetic": len(synthetic_df),
            "total_time_s": round(elapsed, 4),
        }

    except Exception as exc:
        logger.error("Synthetic-only recovery failed: %s", exc)
        return _empty_result(STRATEGY_SYNTHETIC, error=str(exc))


def run_hybrid_recovery(
    baseline_model: Any,
    recent_real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    model_name: str = "logistic_regression",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Hybrid: combine real + synthetic data for retraining.

    Evaluates on independent future data.
    """
    t0 = time.time()

    try:
        # Combine datasets
        combined_df = pd.concat(
            [recent_real_df, synthetic_df], ignore_index=True
        )

        if len(combined_df) == 0:
            return _empty_result(STRATEGY_HYBRID, error="Combined dataset is empty.")

        X_comb, y_comb = prepare_features(
            combined_df, target_column, numerical_cols, categorical_cols
        )

        new_preprocessor, X_comb_trans = fit_preprocessor(X_comb, numerical_cols, categorical_cols)
        y_comb_enc, le = _fit_encode_target(y_comb)

        clf = create_model(model_name, random_seed=random_seed)
        clf, train_time = train_model(clf, X_comb_trans.values, y_comb_enc)

        # Training metrics
        y_comb_pred = predict_model(clf, X_comb_trans.values)
        train_metrics = evaluate_model(y_comb_enc, y_comb_pred)

        # Future evaluation
        X_fut, y_fut = prepare_features(
            future_df, target_column, numerical_cols, categorical_cols
        )
        X_fut_trans = transform_with_preprocessor(X_fut, new_preprocessor, numerical_cols, categorical_cols)
        y_fut_enc = _encode_target(y_fut, le)
        valid_mask = y_fut_enc != -1

        if valid_mask.sum() == 0:
            return _empty_result(STRATEGY_HYBRID, error="No valid future data.")

        y_fut_pred = predict_model(clf, X_fut_trans.values[valid_mask])
        future_metrics = evaluate_model(y_fut_enc[valid_mask], y_fut_pred)

        elapsed = time.time() - t0

        return {
            "strategy": STRATEGY_HYBRID,
            "model": clf,
            "preprocessor": new_preprocessor,
            "label_encoder": le,
            "training_metrics": train_metrics,
            "future_metrics": future_metrics,
            "training_time_s": round(train_time, 4),
            "intervention_size": len(combined_df),
            "n_real": len(recent_real_df),
            "n_synthetic": len(synthetic_df),
            "total_time_s": round(elapsed, 4),
        }

    except Exception as exc:
        logger.error("Hybrid recovery failed: %s", exc)
        return _empty_result(STRATEGY_HYBRID, error=str(exc))


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def compare_recovery_strategies(
    strategy_results: Dict[str, Dict[str, Any]],
    baseline_future_f1: float,
) -> pd.DataFrame:
    """
    Build a comparison table across all recovery strategies.

    Acceptance is based on FUTURE evaluation vs. baseline.
    Training metrics alone do NOT determine acceptance.

    Parameters
    ----------
    strategy_results     : dict keyed by strategy name
    baseline_future_f1   : baseline F1 on future data (no recovery)

    Returns
    -------
    pd.DataFrame with comparison
    """
    rows = []

    for strategy_name, result in strategy_results.items():
        if result.get("error"):
            rows.append({
                "strategy": strategy_name,
                "future_accuracy": 0.0,
                "future_f1": 0.0,
                "recovery_gain": 0.0,
                "remaining_degradation": 0.0,
                "training_time_s": 0.0,
                "n_synthetic": 0,
                "n_real": 0,
                "intervention_size": 0,
                "decision": "ERROR",
                "error": result.get("error", ""),
            })
            continue

        future_m = result.get("future_metrics", {})
        future_f1 = future_m.get("f1", 0.0)
        future_acc = future_m.get("accuracy", 0.0)

        recovery_gain = future_f1 - baseline_future_f1
        remaining_deg = max(0.0, baseline_future_f1 - future_f1)

        # Decision: ACCEPT only if future F1 improves over no-recovery baseline
        if strategy_name == STRATEGY_NONE:
            decision = "BASELINE"
        elif recovery_gain > 0.005:  # meaningful improvement threshold
            decision = "ACCEPT"
        elif recovery_gain > -0.005:  # essentially the same
            decision = "NEUTRAL"
        else:
            decision = "REJECT"

        rows.append({
            "strategy": strategy_name,
            "future_accuracy": round(future_acc, 4),
            "future_f1": round(future_f1, 4),
            "recovery_gain": round(recovery_gain, 4),
            "remaining_degradation": round(remaining_deg, 4),
            "training_time_s": round(result.get("training_time_s", 0), 4),
            "n_synthetic": result.get("n_synthetic", 0),
            "n_real": result.get("n_real", 0),
            "intervention_size": result.get("intervention_size", 0),
            "decision": decision,
            "error": "",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("future_f1", ascending=False).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_target(y: pd.Series, le: Any) -> np.ndarray:
    """Encode target using a fitted LabelEncoder; unknown → -1."""
    return np.array([
        int(le.transform([str(v)])[0]) if str(v) in le.classes_ else -1
        for v in y
    ])


def _fit_encode_target(y: pd.Series):
    """Fit a new LabelEncoder on y and encode it."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def _empty_result(strategy: str, error: str = "") -> Dict[str, Any]:
    """Return an empty/failed result dict."""
    return {
        "strategy": strategy,
        "model": None,
        "future_metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0},
        "training_metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0},
        "training_time_s": 0.0,
        "intervention_size": 0,
        "n_real": 0,
        "n_synthetic": 0,
        "total_time_s": 0.0,
        "error": error,
    }
