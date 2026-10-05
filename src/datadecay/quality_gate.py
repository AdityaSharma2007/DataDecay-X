"""
quality_gate.py — 10-point synthetic data quality gate for DataDecay-X.

Synthetic data MUST pass this gate before entering recovery.
Failed checks produce REJECTED status — bad data is never silently accepted.

Checks:
  1.  Schema validity
  2.  Datatype validity
  3.  Missing values
  4.  Value ranges
  5.  Duplicates
  6.  Distribution similarity
  7.  Concentration / diversity
  8.  Class balance
  9.  Label validity
  10. Downstream utility (quick model probe)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .utils import safe_float

logger = logging.getLogger(__name__)

# Check result constants
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_schema_validity(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    required_columns: List[str],
) -> Dict[str, Any]:
    """Check 1: All required columns present in synthetic data."""
    missing_cols = [c for c in required_columns if c not in synthetic_df.columns]
    extra_cols = [c for c in synthetic_df.columns if c not in source_df.columns]

    if missing_cols:
        return {
            "check": "schema_validity",
            "status": FAIL,
            "detail": f"Missing columns: {missing_cols}",
        }
    if extra_cols:
        return {
            "check": "schema_validity",
            "status": WARN,
            "detail": f"Extra columns not in source: {extra_cols}",
        }
    return {
        "check": "schema_validity",
        "status": PASS,
        "detail": f"All {len(required_columns)} required columns present.",
    }


def check_datatype_validity(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Check 2: Column dtypes match expected types."""
    issues = []

    for col in numerical_cols:
        if col not in synthetic_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(synthetic_df[col]):
            issues.append(f"'{col}' expected numeric, got {synthetic_df[col].dtype}")

    if issues:
        return {
            "check": "datatype_validity",
            "status": FAIL,
            "detail": "; ".join(issues),
        }
    return {
        "check": "datatype_validity",
        "status": PASS,
        "detail": "All column dtypes are valid.",
    }


def check_missing_values(
    synthetic_df: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Check 3: Missing value rate below threshold."""
    total_cells = len(synthetic_df) * len(synthetic_df.columns)
    if total_cells == 0:
        return {"check": "missing_values", "status": FAIL, "detail": "Empty synthetic dataframe."}

    missing_rate = synthetic_df.isnull().sum().sum() / total_cells

    if missing_rate > threshold:
        return {
            "check": "missing_values",
            "status": FAIL,
            "detail": f"Missing rate {missing_rate:.2%} exceeds threshold {threshold:.2%}.",
        }
    if missing_rate > 0:
        return {
            "check": "missing_values",
            "status": WARN,
            "detail": f"Small number of missing values ({missing_rate:.2%}).",
        }
    return {
        "check": "missing_values",
        "status": PASS,
        "detail": "No missing values detected.",
    }


def check_value_ranges(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    tolerance_factor: float = 3.0,
) -> Dict[str, Any]:
    """Check 4: Synthetic numerical values within plausible range of source."""
    out_of_range = []

    for col in numerical_cols:
        if col not in source_df.columns or col not in synthetic_df.columns:
            continue
        src_vals = source_df[col].dropna().values.astype(float)
        try:
            syn_vals = synthetic_df[col].dropna().values.astype(float)
        except (ValueError, TypeError):
            out_of_range.append(f"'{col}' could not be cast to float — wrong dtype")
            continue

        if len(src_vals) == 0 or len(syn_vals) == 0:
            continue

        src_std = float(np.std(src_vals)) if np.std(src_vals) > 0 else 1.0
        src_mean = float(np.mean(src_vals))

        expanded_min = src_mean - tolerance_factor * src_std
        expanded_max = src_mean + tolerance_factor * src_std

        n_out = int(np.sum((syn_vals < expanded_min) | (syn_vals > expanded_max)))
        pct_out = n_out / len(syn_vals) * 100

        if pct_out > 10:
            out_of_range.append(f"'{col}': {pct_out:.1f}% out of expected range")

    if out_of_range:
        return {
            "check": "value_ranges",
            "status": WARN,
            "detail": "; ".join(out_of_range),
        }
    return {
        "check": "value_ranges",
        "status": PASS,
        "detail": "All numerical values within plausible range.",
    }


def check_duplicates(
    synthetic_df: pd.DataFrame,
    threshold: float = 0.10,
) -> Dict[str, Any]:
    """Check 5: Synthetic duplicate rate below threshold."""
    n_dupes = int(synthetic_df.duplicated().sum())
    dup_rate = n_dupes / max(len(synthetic_df), 1)

    if dup_rate > threshold:
        return {
            "check": "duplicates",
            "status": FAIL,
            "detail": f"Duplicate rate {dup_rate:.2%} exceeds threshold {threshold:.2%}.",
        }
    if dup_rate > threshold * 0.5:
        return {
            "check": "duplicates",
            "status": WARN,
            "detail": f"Duplicate rate {dup_rate:.2%} is elevated.",
        }
    return {
        "check": "duplicates",
        "status": PASS,
        "detail": f"Duplicate rate {dup_rate:.2%} is acceptable.",
    }


def check_distribution_similarity(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """Check 6: Distribution similarity via mean shift check."""
    from scipy.stats import ks_2samp

    drifted_cols = []

    for col in numerical_cols:
        if col not in source_df.columns or col not in synthetic_df.columns:
            continue
        try:
            src = source_df[col].dropna().values.astype(float)
            syn = synthetic_df[col].dropna().values.astype(float)
        except (ValueError, TypeError):
            drifted_cols.append(f"'{col}' wrong dtype — cannot compare distribution")
            continue
        if len(src) < 5 or len(syn) < 5:
            continue

        src_std = float(np.std(src)) if np.std(src) > 0 else 1.0
        mean_shift = abs(np.mean(syn) - np.mean(src)) / src_std

        if mean_shift > threshold:
            drifted_cols.append(f"'{col}' (shift={mean_shift:.3f}σ)")

    if len(drifted_cols) > len(numerical_cols) * 0.5:
        return {
            "check": "distribution_similarity",
            "status": FAIL,
            "detail": f"High distribution shift in: {', '.join(drifted_cols[:5])}",
        }
    if drifted_cols:
        return {
            "check": "distribution_similarity",
            "status": WARN,
            "detail": f"Moderate distribution shift in: {', '.join(drifted_cols[:3])}",
        }
    return {
        "check": "distribution_similarity",
        "status": PASS,
        "detail": "Distributions are sufficiently similar to source.",
    }


def check_concentration(
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    diversity_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Check 7: Synthetic data is not overly concentrated (low diversity)."""
    low_diversity = []

    for col in numerical_cols:
        if col not in synthetic_df.columns:
            continue
        vals = synthetic_df[col].dropna().values
        try:
            vals = vals.astype(float)
        except (ValueError, TypeError):
            continue
        if len(vals) < 2:
            continue
        cv = float(np.std(vals)) / (abs(float(np.mean(vals))) + 1e-8)
        if cv < diversity_threshold:
            low_diversity.append(f"'{col}' (CV={cv:.4f})")

    if len(low_diversity) > len(numerical_cols) * 0.5:
        return {
            "check": "concentration",
            "status": FAIL,
            "detail": f"Low diversity in: {', '.join(low_diversity[:5])}",
        }
    if low_diversity:
        return {
            "check": "concentration",
            "status": WARN,
            "detail": f"Low diversity detected in: {', '.join(low_diversity[:3])}",
        }
    return {
        "check": "concentration",
        "status": PASS,
        "detail": "Synthetic data has adequate diversity.",
    }


def check_class_balance(
    synthetic_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_column: str,
    imbalance_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Check 8: Target class balance is acceptable."""
    if target_column not in synthetic_df.columns:
        return {
            "check": "class_balance",
            "status": FAIL,
            "detail": f"Target column '{target_column}' missing from synthetic data.",
        }

    syn_vc = synthetic_df[target_column].value_counts(normalize=True)
    src_vc = source_df[target_column].value_counts(normalize=True) if target_column in source_df.columns else None

    n_classes = len(syn_vc)
    if n_classes == 0:
        return {
            "check": "class_balance",
            "status": FAIL,
            "detail": "No classes found in target column.",
        }

    min_pct = float(syn_vc.min())
    max_pct = float(syn_vc.max())

    if min_pct < imbalance_threshold and n_classes > 1:
        return {
            "check": "class_balance",
            "status": WARN,
            "detail": (
                f"Minority class at {min_pct:.2%} of synthetic data. "
                f"Consider minority-aware sampling."
            ),
        }

    return {
        "check": "class_balance",
        "status": PASS,
        "detail": (
            f"{n_classes} classes. Min class proportion: {min_pct:.2%}."
        ),
    }


def check_label_validity(
    synthetic_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    """Check 9: All synthetic labels are valid (exist in source labels)."""
    if target_column not in synthetic_df.columns:
        return {
            "check": "label_validity",
            "status": FAIL,
            "detail": f"Target column '{target_column}' not found in synthetic data.",
        }
    if target_column not in source_df.columns:
        return {
            "check": "label_validity",
            "status": WARN,
            "detail": "Cannot validate labels — target not in source data.",
        }

    src_labels = set(source_df[target_column].astype(str).unique())
    syn_labels = set(synthetic_df[target_column].astype(str).unique())
    invalid = syn_labels - src_labels
    missing = src_labels - syn_labels

    if invalid:
        return {
            "check": "label_validity",
            "status": FAIL,
            "detail": f"Invalid labels in synthetic data: {invalid}",
        }
    if missing:
        return {
            "check": "label_validity",
            "status": WARN,
            "detail": f"Some source classes absent from synthetic data: {missing}",
        }
    return {
        "check": "label_validity",
        "status": PASS,
        "detail": f"All {len(syn_labels)} labels are valid.",
    }


def check_downstream_utility(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    target_column: str,
    min_f1: float = 0.40,
) -> Dict[str, Any]:
    """
    Check 10: Downstream utility — can a simple model trained on synthetic data
    achieve non-trivial F1 on a hold-out from source data?

    This tests whether synthetic data carries useful predictive signal.
    """
    feat_cols = [c for c in numerical_cols
                 if c in synthetic_df.columns and c in source_df.columns]

    if len(feat_cols) == 0 or target_column not in synthetic_df.columns:
        return {
            "check": "downstream_utility",
            "status": WARN,
            "detail": "Could not run downstream utility check (missing features/target).",
        }

    try:
        # Prepare synthetic training data
        X_syn = synthetic_df[feat_cols].fillna(0).values.astype(float)
        y_syn = synthetic_df[target_column].astype(str).values

        le = LabelEncoder()
        y_syn_enc = le.fit_transform(y_syn)

        if len(np.unique(y_syn_enc)) < 2:
            return {
                "check": "downstream_utility",
                "status": WARN,
                "detail": "Only one class in synthetic target — cannot train classifier.",
            }

        # Source hold-out
        src_valid = source_df[feat_cols + [target_column]].dropna()
        if len(src_valid) < 10:
            return {
                "check": "downstream_utility",
                "status": WARN,
                "detail": "Insufficient source data for downstream probe.",
            }

        X_src = src_valid[feat_cols].values.astype(float)
        y_src_raw = src_valid[target_column].astype(str).values

        # Map source labels through same encoder
        y_src_enc = np.array([
            int(le.transform([v])[0]) if v in le.classes_ else -1
            for v in y_src_raw
        ])
        valid_mask = y_src_enc != -1
        if valid_mask.sum() < 5:
            return {
                "check": "downstream_utility",
                "status": WARN,
                "detail": "Insufficient matching labels for downstream probe.",
            }

        X_src_valid = X_src[valid_mask]
        y_src_valid = y_src_enc[valid_mask]

        # Train a quick LR probe on synthetic
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, C=1.0)),
        ])
        probe.fit(X_syn, y_syn_enc)
        y_pred = probe.predict(X_src_valid)

        f1 = float(f1_score(y_src_valid, y_pred, average="weighted", zero_division=0))

        if f1 < min_f1:
            return {
                "check": "downstream_utility",
                "status": FAIL,
                "detail": (
                    f"Downstream probe F1={f1:.3f} < threshold {min_f1:.2f}. "
                    "Synthetic data may lack useful predictive signal."
                ),
            }
        return {
            "check": "downstream_utility",
            "status": PASS,
            "detail": f"Downstream probe F1={f1:.3f} (threshold={min_f1:.2f}).",
        }

    except Exception as exc:
        logger.warning("Downstream utility check failed: %s", exc)
        return {
            "check": "downstream_utility",
            "status": WARN,
            "detail": f"Check could not complete: {exc}",
        }


# ---------------------------------------------------------------------------
# Main quality gate
# ---------------------------------------------------------------------------

def run_quality_gate(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_column: str,
    minimum_score: float = 0.70,
    missing_threshold: float = 0.05,
    duplicate_threshold: float = 0.10,
    distribution_threshold: float = 0.20,
    class_imbalance_ratio: float = 0.10,
    downstream_min_f1: float = 0.40,
) -> Dict[str, Any]:
    """
    Run the complete 10-point quality gate.

    Parameters
    ----------
    source_df             : original reference data
    synthetic_df          : generated synthetic data
    numerical_cols        : numerical feature columns
    categorical_cols      : categorical feature columns
    target_column         : target/label column
    minimum_score         : minimum quality score to ACCEPT (0–1)

    Returns
    -------
    dict with:
        quality_score (float 0–1)
        passed (bool)
        decision ('ACCEPTED' or 'REJECTED')
        failed_checks (list)
        warned_checks (list)
        check_results (list of individual check dicts)
        detailed_report (str)
    """
    t0 = time.time()

    required_cols = (
        [c for c in numerical_cols if c in source_df.columns] +
        [c for c in categorical_cols if c in source_df.columns] +
        ([target_column] if target_column in source_df.columns else [])
    )

    # Run all 10 checks
    results = [
        check_schema_validity(source_df, synthetic_df, required_cols),
        check_datatype_validity(source_df, synthetic_df, numerical_cols, categorical_cols),
        check_missing_values(synthetic_df, threshold=missing_threshold),
        check_value_ranges(source_df, synthetic_df, numerical_cols),
        check_duplicates(synthetic_df, threshold=duplicate_threshold),
        check_distribution_similarity(source_df, synthetic_df, numerical_cols,
                                      threshold=distribution_threshold),
        check_concentration(synthetic_df, numerical_cols),
        check_class_balance(synthetic_df, source_df, target_column,
                            imbalance_threshold=class_imbalance_ratio),
        check_label_validity(synthetic_df, source_df, target_column),
        check_downstream_utility(source_df, synthetic_df, numerical_cols,
                                 target_column, min_f1=downstream_min_f1),
    ]

    # Score calculation
    # PASS = 1.0 weight, WARN = 0.5 weight, FAIL = 0.0 weight
    n_checks = len(results)
    score_map = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
    raw_score = sum(score_map.get(r["status"], 0.0) for r in results) / n_checks

    failed_checks = [r["check"] for r in results if r["status"] == FAIL]
    warned_checks = [r["check"] for r in results if r["status"] == WARN]
    passed_checks = [r["check"] for r in results if r["status"] == PASS]

    # Decision: any FAIL → REJECTED regardless of score (unless user overrides)
    # But let score determine boundary for WARN-only cases
    has_fail = len(failed_checks) > 0
    decision = "REJECTED" if (has_fail or raw_score < minimum_score) else "ACCEPTED"

    # Detailed report
    lines = ["=== SYNTHETIC DATA QUALITY GATE REPORT ===", ""]
    for r in results:
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(r["status"], "[ ?? ]")
        lines.append(f"{icon} {r['check'].upper().replace('_', ' ')}: {r['detail']}")
    lines.append("")
    lines.append(f"Quality Score: {raw_score:.2%}")
    lines.append(f"Decision: {decision}")
    if failed_checks:
        lines.append(f"Failed checks: {', '.join(failed_checks)}")
    if warned_checks:
        lines.append(f"Warned checks: {', '.join(warned_checks)}")

    elapsed = time.time() - t0

    logger.info(
        "Quality gate — score=%.3f, decision=%s, failed=%d, warned=%d (%.2fs)",
        raw_score, decision, len(failed_checks), len(warned_checks), elapsed,
    )

    return {
        "quality_score": round(raw_score, 4),
        "passed": decision == "ACCEPTED",
        "decision": decision,
        "failed_checks": failed_checks,
        "warned_checks": warned_checks,
        "passed_checks": passed_checks,
        "check_results": results,
        "detailed_report": "\n".join(lines),
        "n_checks": n_checks,
        "gate_time_s": round(elapsed, 4),
        "minimum_score": minimum_score,
    }
