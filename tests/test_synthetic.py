"""
test_synthetic.py — Tests for synthetic data generation.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.synthetic import generate_synthetic_data, distribution_comparison


@pytest.fixture
def source_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "age": rng.integers(20, 70, n).astype(float),
        "income": rng.lognormal(10.5, 0.5, n),
        "region": rng.choice(["North", "South", "East"], n),
        "target": rng.integers(0, 2, n),
    })


NUM_COLS = ["age", "income"]
CAT_COLS = ["region"]
TARGET = "target"


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_generate_returns_correct_shape(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=200, method="perturbation", random_seed=42,
    )
    assert len(result["synthetic_df"]) == 200
    assert set(NUM_COLS + CAT_COLS + [TARGET]).issubset(result["synthetic_df"].columns)


def test_generate_resampling(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=100, method="resampling", random_seed=1,
    )
    assert len(result["synthetic_df"]) == 100
    assert result["method"] == "resampling"


def test_generate_perturbation(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=100, method="perturbation", perturbation_scale=0.05, random_seed=2,
    )
    syn = result["synthetic_df"]
    # Numerical values should differ from source due to perturbation
    assert not np.array_equal(
        syn["age"].values[:10],
        source_df["age"].values[:10],
    )


def test_generate_categorical_sampling(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=100, method="categorical_sampling", random_seed=3,
    )
    syn = result["synthetic_df"]
    # All categories should be from the original set
    assert set(syn["region"].unique()).issubset({"North", "South", "East"})


def test_generate_hybrid(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=150, method="hybrid", random_seed=4,
    )
    assert len(result["synthetic_df"]) == 150
    assert result["method"] == "hybrid"


# ---------------------------------------------------------------------------
# Schema preservation
# ---------------------------------------------------------------------------

def test_schema_preserved(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=100, method="perturbation", random_seed=5,
    )
    syn = result["synthetic_df"]
    # All expected columns present
    for col in NUM_COLS + CAT_COLS + [TARGET]:
        assert col in syn.columns


def test_target_labels_valid(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=100, method="resampling", random_seed=6,
    )
    syn_labels = set(result["synthetic_df"][TARGET].unique())
    src_labels = set(source_df[TARGET].unique())
    assert syn_labels.issubset(src_labels)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(source_df):
    kwargs = dict(
        source_df=source_df, numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS, target_column=TARGET,
        n_samples=100, method="perturbation", random_seed=99,
    )
    r1 = generate_synthetic_data(**kwargs)
    r2 = generate_synthetic_data(**kwargs)
    pd.testing.assert_frame_equal(
        r1["synthetic_df"].reset_index(drop=True),
        r2["synthetic_df"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_keys(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=50, method="perturbation", random_seed=7,
    )
    meta = result["generation_metadata"]
    assert "method" in meta
    assert "n_samples_generated" in meta
    assert "random_seed" in meta
    assert "generation_time_s" in meta


def test_empty_source_raises():
    with pytest.raises(ValueError):
        generate_synthetic_data(
            pd.DataFrame(), [], [], "target",
            n_samples=50, method="resampling",
        )


def test_invalid_method(source_df):
    with pytest.raises(ValueError):
        generate_synthetic_data(
            source_df, NUM_COLS, CAT_COLS, TARGET,
            n_samples=50, method="gan_magic",
        )


# ---------------------------------------------------------------------------
# distribution_comparison
# ---------------------------------------------------------------------------

def test_distribution_comparison(source_df):
    result = generate_synthetic_data(
        source_df, NUM_COLS, CAT_COLS, TARGET,
        n_samples=300, method="perturbation",
        perturbation_scale=0.05, random_seed=8,
    )
    cmp_df = distribution_comparison(
        result["source_stats"], result["synthetic_stats"], NUM_COLS
    )
    assert not cmp_df.empty
    assert "mean_diff_pct" in cmp_df.columns
    # Small perturbation → mean should be close
    assert cmp_df["mean_diff_pct"].max() < 20  # less than 20% mean diff
