"""
synthetic.py — Controlled synthetic data generation for DataDecay-X.

Methods:
  1. resampling          — bootstrap resample from recent data
  2. perturbation        — add controlled Gaussian noise to numerical features
  3. categorical_sampling — resample categoricals from adjusted distributions
  4. hybrid              — combine perturbation + categorical_sampling

All methods are transparent, reproducible, and understandable.
No GAN/VAE/diffusion models are required for the default implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import set_random_seed

logger = logging.getLogger(__name__)

VALID_METHODS = ["resampling", "perturbation", "categorical_sampling", "hybrid"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    source_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_column: str,
    n_samples: int = 500,
    method: str = "perturbation",
    perturbation_scale: float = 0.05,
    target_regime: Optional[str] = None,
    random_seed: int = 42,
    minority_aware: bool = True,
) -> Dict[str, Any]:
    """
    Generate synthetic candidate recovery data.

    Parameters
    ----------
    source_df           : reference/recent data to generate from
    numerical_cols      : numerical feature columns
    categorical_cols    : categorical feature columns
    target_column       : target/label column
    n_samples           : number of synthetic rows to generate
    method              : generation method
    perturbation_scale  : noise magnitude (fraction of std) for perturbation
    target_regime       : optional description of target data regime
    random_seed         : reproducibility seed
    minority_aware      : if True, oversample minority class proportionally

    Returns
    -------
    dict with:
        synthetic_df        : generated DataFrame
        method              : method used
        generation_metadata : reproducibility and config info
        generation_time_s   : seconds taken
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}. Got: '{method}'")

    if len(source_df) == 0:
        raise ValueError("source_df is empty — cannot generate synthetic data.")

    t0 = time.time()
    set_random_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # Verify columns exist
    num_cols = [c for c in numerical_cols if c in source_df.columns]
    cat_cols = [c for c in categorical_cols if c in source_df.columns]
    has_target = target_column in source_df.columns

    if method == "resampling":
        synthetic_df = _generate_resampling(
            source_df, num_cols, cat_cols, target_column, n_samples, rng,
            minority_aware=minority_aware,
        )
    elif method == "perturbation":
        synthetic_df = _generate_perturbation(
            source_df, num_cols, cat_cols, target_column, n_samples,
            perturbation_scale, rng, minority_aware=minority_aware,
        )
    elif method == "categorical_sampling":
        synthetic_df = _generate_categorical_sampling(
            source_df, num_cols, cat_cols, target_column, n_samples, rng,
            minority_aware=minority_aware,
        )
    elif method == "hybrid":
        synthetic_df = _generate_hybrid(
            source_df, num_cols, cat_cols, target_column, n_samples,
            perturbation_scale, rng, minority_aware=minority_aware,
        )

    elapsed = time.time() - t0

    # Generation metadata for reproducibility
    metadata = {
        "method": method,
        "n_samples_requested": n_samples,
        "n_samples_generated": len(synthetic_df),
        "source_rows": len(source_df),
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "target_column": target_column,
        "perturbation_scale": perturbation_scale,
        "minority_aware": minority_aware,
        "random_seed": random_seed,
        "target_regime": target_regime or "not specified",
        "generation_time_s": round(elapsed, 4),
        "note": (
            "Synthetic data generated using transparent statistical methods. "
            "Quality must be validated before use in recovery."
        ),
    }

    # Source distribution statistics
    source_stats = _compute_distribution_stats(source_df, num_cols, cat_cols)
    synth_stats = _compute_distribution_stats(synthetic_df, num_cols, cat_cols)

    metadata["source_stats"] = source_stats
    metadata["synthetic_stats"] = synth_stats

    logger.info(
        "Synthetic data generated — method=%s, n=%d, time=%.2fs",
        method, len(synthetic_df), elapsed,
    )

    return {
        "synthetic_df": synthetic_df,
        "method": method,
        "generation_metadata": metadata,
        "generation_time_s": round(elapsed, 4),
        "source_stats": source_stats,
        "synthetic_stats": synth_stats,
    }


# ---------------------------------------------------------------------------
# Generation methods
# ---------------------------------------------------------------------------

def _generate_resampling(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    n_samples: int,
    rng: np.random.Generator,
    minority_aware: bool = True,
) -> pd.DataFrame:
    """
    Bootstrap resample from source data.

    If minority_aware, oversample the minority class to restore balance.
    """
    if minority_aware and target_col in df.columns:
        return _class_balanced_resample(df, target_col, n_samples, rng)

    idx = rng.integers(0, len(df), size=n_samples)
    sample = df.iloc[idx].copy().reset_index(drop=True)
    return sample


def _generate_perturbation(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    n_samples: int,
    perturbation_scale: float,
    rng: np.random.Generator,
    minority_aware: bool = True,
) -> pd.DataFrame:
    """
    Add controlled Gaussian noise to numerical features.
    Categorical features and target are resampled directly.
    """
    if minority_aware and target_col in df.columns:
        base = _class_balanced_resample(df, target_col, n_samples, rng)
    else:
        idx = rng.integers(0, len(df), size=n_samples)
        base = df.iloc[idx].copy().reset_index(drop=True)

    result = base.copy()

    for col in num_cols:
        if col not in result.columns:
            continue
        col_std = float(df[col].std()) if df[col].std() > 0 else 1.0
        noise_std = perturbation_scale * col_std
        noise = rng.normal(0, noise_std, size=len(result))
        result[col] = result[col].values.astype(float) + noise

    return result


def _generate_categorical_sampling(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    n_samples: int,
    rng: np.random.Generator,
    minority_aware: bool = True,
) -> pd.DataFrame:
    """
    Sample numerical features from empirical distribution;
    resample categoricals from adjusted marginal distributions.
    """
    if minority_aware and target_col in df.columns:
        base = _class_balanced_resample(df, target_col, n_samples, rng)
    else:
        idx = rng.integers(0, len(df), size=n_samples)
        base = df.iloc[idx].copy().reset_index(drop=True)

    result = base.copy()

    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        new_vals = rng.choice(vc.index.tolist(), size=len(result), p=vc.values)
        result[col] = new_vals

    return result


def _generate_hybrid(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    n_samples: int,
    perturbation_scale: float,
    rng: np.random.Generator,
    minority_aware: bool = True,
) -> pd.DataFrame:
    """
    Hybrid: numerical perturbation + categorical probability sampling.
    """
    # Start with perturbation
    result = _generate_perturbation(
        df, num_cols, cat_cols, target_col, n_samples,
        perturbation_scale, rng, minority_aware=minority_aware,
    )

    # Then resample categoricals from marginals
    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        new_vals = rng.choice(vc.index.tolist(), size=len(result), p=vc.values)
        result[col] = new_vals

    return result


# ---------------------------------------------------------------------------
# Class-balanced resampling helper
# ---------------------------------------------------------------------------

def _class_balanced_resample(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample with balanced class representation.
    Each class contributes proportionally to n_samples.
    """
    classes = df[target_col].unique()
    n_classes = len(classes)

    if n_classes == 0:
        idx = rng.integers(0, len(df), size=n_samples)
        return df.iloc[idx].copy().reset_index(drop=True)

    per_class = max(1, n_samples // n_classes)
    frames = []

    for cls in classes:
        cls_df = df[df[target_col] == cls]
        if len(cls_df) == 0:
            continue
        idx = rng.integers(0, len(cls_df), size=per_class)
        frames.append(cls_df.iloc[idx])

    if not frames:
        idx = rng.integers(0, len(df), size=n_samples)
        return df.iloc[idx].copy().reset_index(drop=True)

    combined = pd.concat(frames, ignore_index=True)

    # Trim or pad to exactly n_samples
    if len(combined) > n_samples:
        combined = combined.sample(n=n_samples, random_state=int(rng.integers(0, 10000))).reset_index(drop=True)
    elif len(combined) < n_samples:
        extra_idx = rng.integers(0, len(df), size=n_samples - len(combined))
        extra = df.iloc[extra_idx]
        combined = pd.concat([combined, extra], ignore_index=True)

    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution statistics for comparison
# ---------------------------------------------------------------------------

def _compute_distribution_stats(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, Any]:
    """Compute per-column distribution statistics."""
    stats: Dict[str, Any] = {}

    for col in num_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        stats[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "median": round(float(np.median(vals)), 4),
        }

    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        stats[col] = {
            "top_category": str(vc.index[0]) if len(vc) > 0 else "",
            "n_categories": int(len(vc)),
            "distribution": {str(k): round(float(v), 4) for k, v in vc.items()},
        }

    return stats


def distribution_comparison(
    source_stats: Dict[str, Any],
    synthetic_stats: Dict[str, Any],
    num_cols: List[str],
) -> pd.DataFrame:
    """
    Compare source vs synthetic distribution statistics for numerical cols.

    Returns
    -------
    DataFrame with columns: feature, source_mean, synth_mean, source_std,
                             synth_std, mean_diff_pct, std_diff_pct
    """
    rows = []
    for col in num_cols:
        if col not in source_stats or col not in synthetic_stats:
            continue
        src = source_stats[col]
        syn = synthetic_stats[col]

        src_mean = src.get("mean", 0)
        syn_mean = syn.get("mean", 0)
        src_std = src.get("std", 1)
        syn_std = syn.get("std", 1)

        mean_diff_pct = abs(syn_mean - src_mean) / (abs(src_mean) + 1e-8) * 100
        std_diff_pct = abs(syn_std - src_std) / (src_std + 1e-8) * 100

        rows.append({
            "feature": col,
            "source_mean": round(src_mean, 4),
            "synth_mean": round(syn_mean, 4),
            "source_std": round(src_std, 4),
            "synth_std": round(syn_std, 4),
            "mean_diff_pct": round(mean_diff_pct, 2),
            "std_diff_pct": round(std_diff_pct, 2),
        })

    return pd.DataFrame(rows)
