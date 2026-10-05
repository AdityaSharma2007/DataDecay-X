"""
evaluation.py — Independent future evaluation for DataDecay-X.

CRITICAL: Final evaluation uses ONLY the held-out future window.
This data was never seen during training, preprocessing fitting,
synthetic generation, or quality gate validation.

Candidate acceptance is based solely on future performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .models import evaluate_model, predict_model
from .preprocessing import prepare_features, transform_with_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_future_window(
    model: Any,
    future_df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    preprocessor: Any,
    label_encoder: Any,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on the independent future window.

    Parameters
    ----------
    model          : fitted sklearn estimator (baseline or candidate)
    future_df      : held-out future data — never used during training
    target_column  : target column name
    numerical_cols : numerical feature columns
    categorical_cols : categorical feature columns
    preprocessor   : fitted ColumnTransformer
    label_encoder  : fitted LabelEncoder for target

    Returns
    -------
    dict with: accuracy, precision, recall, f1, n_samples, n_classes
    """
    if model is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_samples": 0, "error": "Model is None"}

    try:
        X_fut, y_fut = prepare_features(
            future_df, target_column, numerical_cols, categorical_cols
        )

        if len(X_fut) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_samples": 0, "error": "Empty future data"}

        X_fut_trans = transform_with_preprocessor(
            X_fut, preprocessor, numerical_cols, categorical_cols
        )

        # Encode target
        y_enc = np.array([
            int(label_encoder.transform([str(v)])[0])
            if str(v) in label_encoder.classes_ else -1
            for v in y_fut
        ])
        valid_mask = y_enc != -1

        if valid_mask.sum() < 2:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "n_samples": 0, "error": "Insufficient valid labels in future data"}

        y_pred = predict_model(model, X_fut_trans.values[valid_mask])
        metrics = evaluate_model(y_enc[valid_mask], y_pred)
        return metrics

    except Exception as exc:
        logger.error("Future window evaluation failed: %s", exc)
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_samples": 0, "error": str(exc)}


def compare_baseline_candidate(
    baseline_future_metrics: Dict[str, Any],
    candidate_future_metrics: Dict[str, Any],
    candidate_name: str = "candidate",
) -> Dict[str, Any]:
    """
    Compare baseline vs candidate model performance on future data.

    Returns
    -------
    dict with: recovery_gain, decision, explanation
    """
    baseline_f1 = float(baseline_future_metrics.get("f1", 0.0))
    candidate_f1 = float(candidate_future_metrics.get("f1", 0.0))

    recovery_gain = candidate_f1 - baseline_f1
    recovery_gain_pct = (recovery_gain / max(baseline_f1, 1e-6)) * 100

    # Decision thresholds (engineering values)
    if recovery_gain > 0.01:
        decision = "PROMOTE_CANDIDATE"
        explanation = (
            f"Candidate '{candidate_name}' achieves F1={candidate_f1:.4f} on future data, "
            f"a gain of {recovery_gain:.4f} ({recovery_gain_pct:.1f}%) over baseline "
            f"F1={baseline_f1:.4f}. Candidate is recommended."
        )
    elif recovery_gain >= -0.005:
        decision = "NEUTRAL"
        explanation = (
            f"Candidate '{candidate_name}' achieves F1={candidate_f1:.4f}, essentially "
            f"equivalent to baseline F1={baseline_f1:.4f} (difference={recovery_gain:.4f}). "
            f"No clear improvement. Recommend retaining baseline."
        )
    else:
        decision = "RETAIN_BASELINE"
        explanation = (
            f"Candidate '{candidate_name}' achieves F1={candidate_f1:.4f} on future data, "
            f"which is WORSE than baseline F1={baseline_f1:.4f} by {abs(recovery_gain):.4f}. "
            f"Baseline retained. Candidate rejected."
        )

    return {
        "baseline_f1": round(baseline_f1, 4),
        "candidate_f1": round(candidate_f1, 4),
        "recovery_gain": round(recovery_gain, 4),
        "recovery_gain_pct": round(recovery_gain_pct, 2),
        "decision": decision,
        "explanation": explanation,
        "candidate_name": candidate_name,
    }


def calculate_recovery_gain(
    baseline_future_f1: float,
    candidate_future_f1: float,
) -> Dict[str, float]:
    """
    Calculate recovery gain and remaining degradation.

    recovery_gain = candidate_future_f1 - baseline_future_f1
    remaining_degradation = max(0, original_baseline_f1 - candidate_future_f1)
    """
    gain = candidate_future_f1 - baseline_future_f1
    remaining = max(0.0, baseline_future_f1 - candidate_future_f1)

    return {
        "recovery_gain": round(gain, 4),
        "recovery_gain_pct": round(gain / max(baseline_future_f1, 1e-6) * 100, 2),
        "remaining_degradation": round(remaining, 4),
    }


def build_evaluation_summary(
    baseline_train_metrics: Dict[str, Any],
    baseline_future_metrics: Dict[str, Any],
    strategy_results: Dict[str, Dict[str, Any]],
    best_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete evaluation summary for reporting.

    Parameters
    ----------
    baseline_train_metrics   : baseline metrics on training/reference data
    baseline_future_metrics  : baseline metrics on held-out future data
    strategy_results         : dict of recovery strategy results
    best_strategy            : name of the best recovery strategy (if any)

    Returns
    -------
    Complete summary dict
    """
    baseline_train_f1 = float(baseline_train_metrics.get("f1", 0.0))
    baseline_future_f1 = float(baseline_future_metrics.get("f1", 0.0))

    # Degradation from training to future (natural drift effect)
    train_to_future_drop = baseline_train_f1 - baseline_future_f1
    train_to_future_drop_pct = (
        train_to_future_drop / max(baseline_train_f1, 1e-6) * 100
    )

    # Summarize each strategy
    strategy_summaries = []
    best_future_f1 = baseline_future_f1
    best_strat_name = "no_recovery"

    for name, result in strategy_results.items():
        future_m = result.get("future_metrics", {})
        fut_f1 = float(future_m.get("f1", 0.0))
        gain = fut_f1 - baseline_future_f1
        train_m = result.get("training_metrics", {})

        strategy_summaries.append({
            "strategy": name,
            "training_f1": round(float(train_m.get("f1", 0.0)), 4),
            "future_f1": round(fut_f1, 4),
            "future_accuracy": round(float(future_m.get("accuracy", 0.0)), 4),
            "future_precision": round(float(future_m.get("precision", 0.0)), 4),
            "future_recall": round(float(future_m.get("recall", 0.0)), 4),
            "recovery_gain": round(gain, 4),
            "n_synthetic": result.get("n_synthetic", 0),
            "n_real": result.get("n_real", 0),
            "training_time_s": round(result.get("training_time_s", 0.0), 4),
            "error": result.get("error", ""),
        })

        if fut_f1 > best_future_f1:
            best_future_f1 = fut_f1
            best_strat_name = name

    effective_best = best_strategy or best_strat_name
    overall_recovery_gain = best_future_f1 - baseline_future_f1

    # Final decision
    if overall_recovery_gain > 0.01:
        final_decision = "RECOVERY_SUCCESSFUL"
        final_explanation = (
            f"Best strategy '{effective_best}' improved future F1 by "
            f"{overall_recovery_gain:.4f} over baseline."
        )
    elif overall_recovery_gain >= -0.005:
        final_decision = "NO_IMPROVEMENT"
        final_explanation = (
            "No recovery strategy produced a meaningful improvement. "
            "Recommend retaining the baseline model and continuing monitoring."
        )
    else:
        final_decision = "RECOVERY_FAILED"
        final_explanation = (
            "All recovery strategies performed worse than baseline on future data. "
            "Baseline model retained. Investigate data quality and generation parameters."
        )

    return {
        "baseline_train_f1": round(baseline_train_f1, 4),
        "baseline_future_f1": round(baseline_future_f1, 4),
        "train_to_future_drop": round(train_to_future_drop, 4),
        "train_to_future_drop_pct": round(train_to_future_drop_pct, 2),
        "strategy_summaries": strategy_summaries,
        "best_strategy": effective_best,
        "best_future_f1": round(best_future_f1, 4),
        "overall_recovery_gain": round(overall_recovery_gain, 4),
        "final_decision": final_decision,
        "final_explanation": final_explanation,
    }
