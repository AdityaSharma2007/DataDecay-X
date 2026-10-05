"""
demo_data.py — Reproducible demo dataset for DataDecay-X.

Generates a realistic tabular classification dataset (credit/banking risk)
with timestamps, numerical features, categorical features, and a binary target.

The dataset is large enough for 12+ temporal windows.
Uses deterministic random seed for full reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_csv

logger = logging.getLogger(__name__)

# Dataset configuration
DEMO_N_ROWS = 3000
DEMO_SEED = 42
DEMO_FILENAME = "demo_credit_risk.csv"

# Column definitions
NUMERICAL_COLS = [
    "age",
    "income",
    "tenure_months",
    "transaction_amount",
    "activity_score",
    "balance",
    "credit_utilization",
]

CATEGORICAL_COLS = [
    "region",
    "segment",
    "product_type",
]

TARGET_COL = "default_risk"
TIMESTAMP_COL = "timestamp"


def generate_demo_dataset(
    n_rows: int = DEMO_N_ROWS,
    random_seed: int = DEMO_SEED,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic credit/banking risk classification dataset.

    Features (conceptual — not real customer data):
        timestamp           : evenly spaced between start_date and end_date
        age                 : customer age (20–70)
        income              : annual income (15k–200k)
        tenure_months       : months as customer (1–120)
        transaction_amount  : average transaction amount
        activity_score      : engagement score (0–100)
        balance             : account balance
        credit_utilization  : credit utilization ratio (0–1)
        region              : North / South / East / West
        segment             : Retail / Business / Premium
        product_type        : Loan / Credit / Savings
        default_risk        : 0 = low risk, 1 = high risk (target)

    Parameters
    ----------
    n_rows      : number of rows
    random_seed : reproducibility seed
    start_date  : start of timestamp range
    end_date    : end of timestamp range

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_seed)

    # Timestamps (evenly spaced)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_rows)

    # Numerical features
    age = rng.integers(20, 71, size=n_rows).astype(float)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n_rows)  # ~50k median
    income = np.clip(income, 15_000, 250_000)

    tenure_months = rng.integers(1, 121, size=n_rows).astype(float)

    transaction_amount = rng.lognormal(mean=5.5, sigma=0.8, size=n_rows)
    transaction_amount = np.clip(transaction_amount, 10, 10_000)

    activity_score = rng.uniform(0, 100, size=n_rows)
    balance = rng.lognormal(mean=8.0, sigma=1.0, size=n_rows)
    balance = np.clip(balance, 0, 500_000)

    credit_utilization = rng.beta(2, 5, size=n_rows)  # skewed toward lower values

    # Categorical features
    regions = rng.choice(
        ["North", "South", "East", "West"],
        size=n_rows,
        p=[0.30, 0.25, 0.25, 0.20],
    )
    segments = rng.choice(
        ["Retail", "Business", "Premium"],
        size=n_rows,
        p=[0.60, 0.25, 0.15],
    )
    product_types = rng.choice(
        ["Loan", "Credit", "Savings"],
        size=n_rows,
        p=[0.40, 0.40, 0.20],
    )

    # Target — logistic-style probability from features
    # Higher credit utilization, lower income, lower score → more risk
    log_odds = (
        -2.5
        + 1.8 * credit_utilization
        - 0.6 * (income / income.max())
        + 0.4 * (credit_utilization > 0.7).astype(float)
        - 0.3 * (activity_score / 100)
        - 0.2 * (tenure_months / 120)
        + 0.3 * (balance < 5000).astype(float)
        + rng.normal(0, 0.5, size=n_rows)  # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    # Target: ~25% positive class rate
    default_risk = (prob_default > rng.uniform(0, 1, size=n_rows)).astype(int)

    # Assemble DataFrame
    df = pd.DataFrame({
        TIMESTAMP_COL: dates,
        "age": np.round(age, 1),
        "income": np.round(income, 2),
        "tenure_months": np.round(tenure_months, 0),
        "transaction_amount": np.round(transaction_amount, 2),
        "activity_score": np.round(activity_score, 2),
        "balance": np.round(balance, 2),
        "credit_utilization": np.round(credit_utilization, 4),
        "region": regions,
        "segment": segments,
        "product_type": product_types,
        TARGET_COL: default_risk,
    })

    # Sort chronologically (should already be, but ensure)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Add minor missing values (~1%) for realism
    rng2 = np.random.default_rng(random_seed + 1)
    for col in ["income", "activity_score", "balance"]:
        mask = rng2.random(n_rows) < 0.01
        df.loc[mask, col] = np.nan

    logger.info(
        "Demo dataset generated — %d rows, %d columns, %.1f%% positive class",
        len(df),
        len(df.columns),
        default_risk.mean() * 100,
    )

    return df


def save_demo_dataset(
    df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    random_seed: int = DEMO_SEED,
) -> Path:
    """
    Save the demo dataset to disk.

    Parameters
    ----------
    df         : pre-generated DataFrame (or None to generate fresh)
    output_dir : directory to save to (defaults to data/samples/)
    random_seed: seed for generation if df is None

    Returns
    -------
    Path to the saved CSV file
    """
    if df is None:
        df = generate_demo_dataset(random_seed=random_seed)

    if output_dir is None:
        # Default: data/samples/ relative to project root
        root = Path(__file__).parent.parent.parent
        output_dir = root / "data" / "samples"

    ensure_dir(output_dir)
    output_path = output_dir / DEMO_FILENAME

    df.to_csv(output_path, index=False)
    logger.info("Demo dataset saved to %s", output_path)
    return output_path


def get_demo_feature_config() -> dict:
    """
    Return the feature configuration for the demo dataset.
    Used to pre-fill Streamlit controls.
    """
    return {
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_column": TARGET_COL,
        "timestamp_column": TIMESTAMP_COL,
        "n_rows": DEMO_N_ROWS,
        "seed": DEMO_SEED,
        "description": (
            "Synthetic credit/banking risk dataset. "
            "Binary target: 0=low risk, 1=high risk. "
            "Not real customer data."
        ),
    }
