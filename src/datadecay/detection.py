"""
detection.py — Drift detection engine for DataDecay-X.

Implements PSI, Kolmogorov-Smirnov test, and Wasserstein distance
for feature-level and window-level drift quantification.

NOTE: Thresholds are engineering/project configuration values.
They are NOT universal scientific thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

# Default engineering thresholds (configurable)
PSI_MODERATE = 0.10
PSI_SEVERE = 0.20
KS_ALPHA = 0.05
WASSERSTEIN_THRESHOLD = 0.10

# PSI bin count
N_BINS = 10
EPS = 1e-6  # avoid log(0)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = sum((P_current - P_reference) * ln(P_current / P_reference))

    Interpretation (engineering thresholds — project configuration):
      PSI < 0.10  → stable
      PSI 0.10-0.20 → moderate shift
      PSI > 0.20  → significant shift

    Parameters
    ----------
    reference : array of reference (training) feature values
    current   : array of current window feature values
    n_bins    : number of buckets

    Returns
    -------
    PSI value (float)
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Use reference distribution to define bins
    min_val = min(ref.min(), cur.min())
    max_val = max(ref.max(), cur.max())

    if min_val == max_val:
        return 0.0  # constant column

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0] -= EPS
    bins[-1] += EPS

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = (ref_counts + EPS) / (len(ref) + EPS * n_bins)
    cur_pct = (cur_counts + EPS) / (len(cur) + EPS * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def calculate_psi_categorical(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    PSI for categorical features using category frequencies.
    """
    ref = pd.Series(reference).astype(str)
    cur = pd.Series(current).astype(str)

    all_cats = set(ref.unique()) | set(cur.unique())

    ref_pct = ref.value_counts(normalize=True).reindex(all_cats, fill_value=0) + EPS
    cur_pct = cur.value_counts(normalize=True).reindex(all_cats, fill_value=0) + EPS

    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = float(np.sum((cur_pct.values - ref_pct.values) * np.log(cur_pct.values / ref_pct.values)))
    return max(0.0, psi)


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov
# ---------------------------------------------------------------------------

def calculate_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for numerical features.

    Returns
    -------
    ks_statistic (float), p_value (float)
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) < 5 or len(cur) < 5:
        return 0.0, 1.0  # insufficient data

    ks_stat, p_val = stats.ks_2samp(ref, cur)
    return float(ks_stat), float(p_val)


# ---------------------------------------------------------------------------
# Wasserstein Distance
# ---------------------------------------------------------------------------

def calculate_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Wasserstein (Earth Mover's) distance between two distributions.

    Optionally normalised by the reference standard deviation
    to make it scale-independent.

    Returns
    -------
    Wasserstein distance (float)
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    dist = wasserstein_distance(ref, cur)

    if normalize:
        ref_std = float(np.std(ref))
        if ref_std > EPS:
            dist = dist / ref_std

    return float(dist)


# ---------------------------------------------------------------------------
# Feature-level drift detection
# ---------------------------------------------------------------------------

def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    psi_threshold_moderate: float = PSI_MODERATE,
    psi_threshold_severe: float = PSI_SEVERE,
    ks_alpha: float = KS_ALPHA,
    wasserstein_threshold: float = WASSERSTEIN_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute feature-level drift metrics comparing reference to current window.

    Returns
    -------
    pd.DataFrame with columns:
        feature, type, psi, ks_statistic, ks_p_value,
        ks_drift, wasserstein, severity
    """
    rows = []

    for col in numerical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].values.astype(float)
        cur_vals = current_df[col].values.astype(float)

        psi = calculate_psi(ref_vals, cur_vals)
        ks_stat, p_val = calculate_ks(ref_vals, cur_vals)
        wass = calculate_wasserstein(ref_vals, cur_vals)
        ks_drift = p_val < ks_alpha
        severity = _severity_label(psi, psi_threshold_moderate, psi_threshold_severe,
                                   ks_drift, wass, wasserstein_threshold)

        rows.append({
            "feature": col,
            "type": "numerical",
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(p_val, 4),
            "ks_drift": ks_drift,
            "wasserstein": round(wass, 4),
            "severity": severity,
        })

    for col in categorical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].astype(str).values
        cur_vals = current_df[col].astype(str).values

        psi = calculate_psi_categorical(ref_vals, cur_vals)
        # KS not applicable for categorical — use chi-square style as proxy
        ks_stat, p_val = _categorical_ks_proxy(ref_vals, cur_vals)
        ks_drift = p_val < ks_alpha
        wass = 0.0  # not meaningful for categorical

        severity = _severity_label(psi, psi_threshold_moderate, psi_threshold_severe,
                                   ks_drift, wass, wasserstein_threshold)

        rows.append({
            "feature": col,
            "type": "categorical",
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(p_val, 4),
            "ks_drift": ks_drift,
            "wasserstein": float("nan"),
            "severity": severity,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "feature", "type", "psi", "ks_statistic", "ks_p_value",
            "ks_drift", "wasserstein", "severity",
        ])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Window-level drift summary
# ---------------------------------------------------------------------------

def aggregate_drift_severity(feature_drift_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate feature-level drift into a single window-level severity score.

    Returns
    -------
    dict with: drift_severity_index, n_drifted_features, severity_label,
               max_psi, mean_psi, max_ks, max_wasserstein
    """
    if feature_drift_df.empty:
        return {
            "drift_severity_index": 0.0,
            "n_drifted_features": 0,
            "severity_label": "none",
            "max_psi": 0.0,
            "mean_psi": 0.0,
            "max_ks": 0.0,
            "max_wasserstein": 0.0,
        }

    n_drifted = int(
        (feature_drift_df["severity"].isin(["moderate", "severe"])).sum()
    )
    max_psi = float(feature_drift_df["psi"].max())
    mean_psi = float(feature_drift_df["psi"].mean())
    max_ks = float(feature_drift_df["ks_statistic"].max())

    wass_vals = feature_drift_df["wasserstein"].dropna()
    max_wass = float(wass_vals.max()) if len(wass_vals) > 0 else 0.0

    # Drift Severity Index: weighted combination (project engineering metric)
    dsi = 0.5 * min(max_psi / PSI_SEVERE, 1.0) + \
          0.3 * min(max_ks, 1.0) + \
          0.2 * min(max_wass / (WASSERSTEIN_THRESHOLD * 2), 1.0)
    dsi = float(np.clip(dsi, 0.0, 1.0))

    if dsi < 0.2:
        label = "none"
    elif dsi < 0.4:
        label = "low"
    elif dsi < 0.6:
        label = "moderate"
    elif dsi < 0.8:
        label = "high"
    else:
        label = "severe"

    return {
        "drift_severity_index": round(dsi, 4),
        "n_drifted_features": n_drifted,
        "severity_label": label,
        "max_psi": round(max_psi, 4),
        "mean_psi": round(mean_psi, 4),
        "max_ks": round(max_ks, 4),
        "max_wasserstein": round(max_wass, 4),
    }


def detect_drift_across_windows(
    reference_df: pd.DataFrame,
    windows: List[pd.DataFrame],
    numerical_cols: List[str],
    categorical_cols: List[str],
    psi_threshold_moderate: float = PSI_MODERATE,
    psi_threshold_severe: float = PSI_SEVERE,
    ks_alpha: float = KS_ALPHA,
    wasserstein_threshold: float = WASSERSTEIN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run drift detection for all temporal windows against the reference.

    Returns
    -------
    dict with:
        window_drift_table    : DataFrame (window vs. drift metrics)
        feature_drift_by_win  : dict of per-window feature drift DataFrames
        overall_severity      : aggregate drift severity over all windows
    """
    window_rows = []
    feature_drift_by_win: Dict[int, pd.DataFrame] = {}

    for win_idx, win_df in enumerate(windows):
        feature_drift = detect_feature_drift(
            reference_df=reference_df,
            current_df=win_df,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            psi_threshold_moderate=psi_threshold_moderate,
            psi_threshold_severe=psi_threshold_severe,
            ks_alpha=ks_alpha,
            wasserstein_threshold=wasserstein_threshold,
        )
        feature_drift_by_win[win_idx] = feature_drift

        agg = aggregate_drift_severity(feature_drift)
        row = {"window": win_idx}
        row.update(agg)
        row["n_rows"] = len(win_df)
        window_rows.append(row)

    window_drift_table = pd.DataFrame(window_rows)

    # Overall severity = max across all windows
    if not window_drift_table.empty:
        max_row = window_drift_table.loc[
            window_drift_table["drift_severity_index"].idxmax()
        ]
        overall_severity = aggregate_drift_severity(
            feature_drift_by_win.get(
                int(max_row["window"]),
                pd.DataFrame()
            )
        )
    else:
        overall_severity = aggregate_drift_severity(pd.DataFrame())

    logger.info(
        "Drift detection complete — %d windows, max DSI=%.3f (%s)",
        len(windows),
        overall_severity["drift_severity_index"],
        overall_severity["severity_label"],
    )

    return {
        "window_drift_table": window_drift_table,
        "feature_drift_by_win": feature_drift_by_win,
        "overall_severity": overall_severity,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _severity_label(
    psi: float,
    psi_moderate: float,
    psi_severe: float,
    ks_drift: bool,
    wass: float,
    wass_threshold: float,
) -> str:
    """Assign a severity label based on multiple drift signals."""
    if psi >= psi_severe or (ks_drift and wass >= wass_threshold * 1.5):
        return "severe"
    if psi >= psi_moderate or ks_drift or wass >= wass_threshold:
        return "moderate"
    if psi >= psi_moderate * 0.5:
        return "low"
    return "none"


def _categorical_ks_proxy(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit as a categorical drift proxy.

    Returns (statistic, p_value) in the same form as KS.
    """
    ref_series = pd.Series(reference)
    cur_series = pd.Series(current)
    all_cats = list(set(ref_series.unique()) | set(cur_series.unique()))

    ref_counts = ref_series.value_counts().reindex(all_cats, fill_value=0).values
    cur_counts = cur_series.value_counts().reindex(all_cats, fill_value=0).values

    # Chi-square requires at least some expected counts
    expected = ref_counts * (cur_counts.sum() / max(ref_counts.sum(), 1))
    expected = np.maximum(expected, EPS)

    try:
        chi2, p_val = stats.chisquare(f_obs=cur_counts, f_exp=expected)
        # Normalize chi2 to [0, 1] range like KS statistic
        norm_chi2 = min(float(chi2) / (len(all_cats) * 10), 1.0)
        return norm_chi2, float(p_val)
    except Exception:
        return 0.0, 1.0
