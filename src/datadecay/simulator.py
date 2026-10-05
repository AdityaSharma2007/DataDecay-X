"""
simulator.py — Temporal drift simulation engine for DataDecay-X.

Models covariate/feature distribution shift across temporal windows.
Supports: stable, gradual, abrupt, recurring drift types.

NOTE: This module simulates FEATURE drift (covariate shift).
It does NOT automatically simulate concept drift (change in P(y|X)).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import set_random_seed

logger = logging.getLogger(__name__)

# Drift type constants
DRIFT_STABLE = "stable"
DRIFT_GRADUAL = "gradual"
DRIFT_ABRUPT = "abrupt"
DRIFT_RECURRING = "recurring"

VALID_DRIFT_TYPES = [DRIFT_STABLE, DRIFT_GRADUAL, DRIFT_ABRUPT, DRIFT_RECURRING]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_drift(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    n_windows: int = 12,
    drift_type: str = DRIFT_GRADUAL,
    start_window: int = 4,
    duration: int = 4,
    magnitude: float = 2.0,
    affected_features: Optional[List[str]] = None,
    random_seed: int = 42,
    target_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulate temporal drift on a dataset by modifying feature distributions
    across temporal windows.

    Parameters
    ----------
    df               : chronologically ordered DataFrame (must contain __ddx_window)
    numerical_cols   : list of numerical feature column names
    categorical_cols : list of categorical feature column names
    n_windows        : total number of temporal windows
    drift_type       : 'stable', 'gradual', 'abrupt', 'recurring'
    start_window     : window index where drift begins (0-based)
    duration         : number of windows affected by drift
    magnitude        : drift strength (multiplier on feature std)
    affected_features: specific features to drift; None = auto-select half
    random_seed      : reproducibility seed
    target_column    : preserved unchanged
    timestamp_column : preserved unchanged

    Returns
    -------
    dict with:
        drifted_df        : DataFrame with drifted features
        ground_truth      : drift configuration metadata
        window_metadata   : per-window drift info
        drift_schedule    : list of (window_idx, drift_intensity) tuples
    """
    if drift_type not in VALID_DRIFT_TYPES:
        raise ValueError(f"drift_type must be one of {VALID_DRIFT_TYPES}")

    set_random_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    df = df.copy()

    # Ensure __ddx_window column exists
    if "__ddx_window" not in df.columns:
        n = len(df)
        window_size = max(n // n_windows, 1)
        df["__ddx_window"] = (np.arange(n) // window_size).clip(0, n_windows - 1)

    # Auto-select affected features if not specified
    all_features = [c for c in (numerical_cols + categorical_cols)
                    if c in df.columns]
    if not all_features:
        raise ValueError("No valid feature columns found for drift simulation.")

    if affected_features is None:
        n_affected = max(1, len(all_features) // 2)
        affected_features = [str(f) for f in
            rng.choice(all_features, size=n_affected, replace=False)
        ]
    else:
        affected_features = [f for f in affected_features if f in df.columns]
        if not affected_features:
            logger.warning("None of the specified affected_features exist; selecting automatically.")
            affected_features = all_features[:max(1, len(all_features) // 2)]

    # Build drift schedule
    drift_schedule = build_drift_schedule(
        n_windows=n_windows,
        drift_type=drift_type,
        start_window=start_window,
        duration=duration,
        magnitude=magnitude,
    )

    # Apply drift window by window
    drifted_df = df.copy()
    affected_num = [f for f in affected_features if f in numerical_cols]
    affected_cat = [f for f in affected_features if f in categorical_cols]

    # Compute reference statistics on all data (before drift)
    ref_stats = _compute_reference_stats(df, affected_num, affected_cat)

    window_metadata: List[Dict[str, Any]] = []

    for win_idx in range(n_windows):
        win_mask = drifted_df["__ddx_window"] == win_idx
        if win_mask.sum() == 0:
            continue

        intensity = drift_schedule[win_idx]

        if intensity > 0:
            # Apply numerical drift
            for col in affected_num:
                if col not in drifted_df.columns:
                    continue
                mean_shift = intensity * ref_stats[col]["std"]
                noise_std = 0.1 * intensity * ref_stats[col]["std"]
                n_rows = win_mask.sum()
                noise = rng.normal(0, noise_std, size=n_rows)
                drifted_df.loc[win_mask, col] = (
                    drifted_df.loc[win_mask, col] + mean_shift + noise
                )

            # Apply categorical drift
            for col in affected_cat:
                if col not in drifted_df.columns:
                    continue
                _apply_categorical_drift(
                    drifted_df, win_mask, col, ref_stats, intensity, rng
                )

        window_metadata.append({
            "window": win_idx,
            "drift_intensity": round(float(intensity), 4),
            "is_drifted": intensity > 0,
            "affected_numerical": affected_num if intensity > 0 else [],
            "affected_categorical": affected_cat if intensity > 0 else [],
            "n_rows": int(win_mask.sum()),
        })

    ground_truth = {
        "drift_type": drift_type,
        "start_window": start_window,
        "end_window": min(start_window + duration - 1, n_windows - 1),
        "duration": duration,
        "magnitude": magnitude,
        "affected_features": affected_features,
        "affected_numerical": affected_num,
        "affected_categorical": affected_cat,
        "n_windows": n_windows,
        "random_seed": random_seed,
        "drift_schedule": [
            {"window": i, "intensity": round(float(v), 4)}
            for i, v in enumerate(drift_schedule)
        ],
        "note": (
            "This simulation models covariate/feature drift only. "
            "It does not automatically model concept drift (P(y|X) changes)."
        ),
    }

    logger.info(
        "Drift simulation complete — type=%s, affected=%d features, windows=%d",
        drift_type,
        len(affected_features),
        n_windows,
    )

    return {
        "drifted_df": drifted_df,
        "ground_truth": ground_truth,
        "window_metadata": window_metadata,
        "drift_schedule": drift_schedule,
    }


def create_temporal_windows(
    df: pd.DataFrame,
    n_windows: int = 12,
) -> List[pd.DataFrame]:
    """
    Split a DataFrame into n_windows equal temporal windows.

    Returns list of DataFrames, one per window.
    """
    if "__ddx_window" not in df.columns:
        n = len(df)
        window_size = max(n // n_windows, 1)
        df = df.copy()
        df["__ddx_window"] = (np.arange(n) // window_size).clip(0, n_windows - 1)

    windows = []
    for w in sorted(df["__ddx_window"].unique()):
        windows.append(df[df["__ddx_window"] == w].copy())

    return windows


def build_drift_schedule(
    n_windows: int,
    drift_type: str,
    start_window: int,
    duration: int,
    magnitude: float,
) -> List[float]:
    """
    Build a list of per-window drift intensities [0.0, magnitude].

    Parameters
    ----------
    n_windows    : total window count
    drift_type   : 'stable', 'gradual', 'abrupt', 'recurring'
    start_window : first window with drift
    duration     : number of consecutive drifted windows
    magnitude    : maximum drift intensity

    Returns
    -------
    List of floats, length == n_windows
    """
    schedule = [0.0] * n_windows

    if drift_type == DRIFT_STABLE:
        return schedule

    end_window = min(start_window + duration, n_windows)

    if drift_type == DRIFT_ABRUPT:
        for w in range(start_window, end_window):
            schedule[w] = float(magnitude)

    elif drift_type == DRIFT_GRADUAL:
        # Linear ramp from 0 → magnitude over duration windows
        for i, w in enumerate(range(start_window, end_window)):
            ratio = (i + 1) / duration
            schedule[w] = float(magnitude * ratio)

    elif drift_type == DRIFT_RECURRING:
        # Drift ON for first half of duration, OFF, ON again
        half = max(1, duration // 2)
        # First occurrence
        for w in range(start_window, min(start_window + half, n_windows)):
            schedule[w] = float(magnitude)
        # Second occurrence (if windows remain)
        second_start = start_window + duration
        if second_start < n_windows:
            for w in range(second_start, min(second_start + half, n_windows)):
                schedule[w] = float(magnitude)

    return schedule


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_reference_stats(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute mean, std, and category distributions for reference data."""
    stats: Dict[str, Dict[str, Any]] = {}

    for col in numerical_cols:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()) if col_data.std() > 0 else 1.0,
            "min": float(col_data.min()),
            "max": float(col_data.max()),
        }

    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        stats[col] = {
            "categories": list(vc.index),
            "probs": list(vc.values),
        }

    return stats


def _apply_categorical_drift(
    df: pd.DataFrame,
    mask: pd.Series,
    col: str,
    ref_stats: Dict[str, Dict],
    intensity: float,
    rng: np.random.Generator,
) -> None:
    """
    Apply categorical drift by skewing sampling probabilities.

    Higher intensity → more concentration on minority categories.
    Modifies df in-place.
    """
    if col not in ref_stats:
        return

    cats = ref_stats[col]["categories"]
    probs = np.array(ref_stats[col]["probs"], dtype=float)

    if len(cats) < 2:
        return

    # Skew: concentrate probability on least-frequent category
    skew = np.zeros_like(probs)
    skew[-1] = 1.0  # put weight on last (least frequent)
    drift_probs = (1 - intensity / 3.0) * probs + (intensity / 3.0) * skew
    drift_probs = np.clip(drift_probs, 0, None)
    drift_probs = drift_probs / drift_probs.sum()

    n_rows = int(mask.sum())
    new_vals = rng.choice(cats, size=n_rows, p=drift_probs)
    df.loc[mask, col] = new_vals
