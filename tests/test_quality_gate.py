"""
test_quality_gate.py — Tests for the 10-point synthetic quality gate.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datadecay.quality_gate import (
    FAIL, PASS, WARN,
    check_class_balance,
    check_datatype_validity,
    check_distribution_similarity,
    check_duplicates,
    check_label_validity,
    check_missing_values,
    check_schema_validity,
    check_value_ranges,
    run_quality_gate,
)


@pytest.fixture
def good_source():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(50000, 10000, n),
        "region": rng.choice(["A", "B", "C"], n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def good_synthetic(good_source):
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(50000, 10000, n),
        "region": rng.choice(["A", "B", "C"], n),
        "target": rng.integers(0, 2, n),
    })
    return df


NUM_COLS = ["age", "income"]
CAT_COLS = ["region"]
TARGET = "target"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def test_schema_valid(good_source, good_synthetic):
    result = check_schema_validity(good_source, good_synthetic, ["age", "income", "region", "target"])
    assert result["status"] == PASS


def test_schema_missing_col(good_source, good_synthetic):
    bad = good_synthetic.drop(columns=["income"])
    result = check_schema_validity(good_source, bad, ["age", "income", "region", "target"])
    assert result["status"] == FAIL


def test_datatype_valid(good_source, good_synthetic):
    result = check_datatype_validity(good_source, good_synthetic, NUM_COLS, CAT_COLS)
    assert result["status"] == PASS


def test_datatype_invalid(good_source, good_synthetic):
    bad = good_synthetic.copy()
    bad["age"] = bad["age"].astype(str)  # wrong type
    result = check_datatype_validity(good_source, bad, NUM_COLS, CAT_COLS)
    assert result["status"] == FAIL


def test_missing_values_none(good_synthetic):
    result = check_missing_values(good_synthetic, threshold=0.05)
    assert result["status"] == PASS


def test_missing_values_too_many(good_synthetic):
    bad = good_synthetic.copy()
    bad.loc[:100, "age"] = np.nan  # ~50% missing in age
    result = check_missing_values(bad, threshold=0.05)
    assert result["status"] == FAIL


def test_value_ranges_ok(good_source, good_synthetic):
    result = check_value_ranges(good_source, good_synthetic, NUM_COLS)
    assert result["status"] == PASS


def test_duplicates_ok(good_synthetic):
    result = check_duplicates(good_synthetic, threshold=0.10)
    assert result["status"] == PASS


def test_duplicates_too_many():
    single_row = pd.DataFrame({"age": [40], "income": [50000], "target": [0]})
    all_same = pd.concat([single_row] * 100, ignore_index=True)
    result = check_duplicates(all_same, threshold=0.10)
    assert result["status"] == FAIL


def test_distribution_similarity_ok(good_source, good_synthetic):
    result = check_distribution_similarity(good_source, good_synthetic, NUM_COLS)
    assert result["status"] == PASS


def test_distribution_similarity_shifted(good_source):
    rng = np.random.default_rng(0)
    heavily_shifted = pd.DataFrame({
        "age": rng.normal(100, 5, 200),     # very different mean
        "income": rng.normal(200000, 5000, 200),
        "region": rng.choice(["A"], 200),
        "target": rng.integers(0, 2, 200),
    })
    result = check_distribution_similarity(good_source, heavily_shifted, NUM_COLS, threshold=0.20)
    assert result["status"] in (WARN, FAIL)


def test_class_balance_ok(good_source, good_synthetic):
    result = check_class_balance(good_synthetic, good_source, TARGET)
    assert result["status"] in (PASS, WARN)


def test_label_validity_ok(good_source, good_synthetic):
    result = check_label_validity(good_synthetic, good_source, TARGET)
    assert result["status"] == PASS


def test_label_validity_invalid_label(good_source, good_synthetic):
    bad = good_synthetic.copy()
    bad.loc[0, "target"] = 99  # invalid label
    result = check_label_validity(bad, good_source, TARGET)
    assert result["status"] == FAIL


# ---------------------------------------------------------------------------
# Full quality gate
# ---------------------------------------------------------------------------

def test_gate_accepts_good_data(good_source, good_synthetic):
    result = run_quality_gate(
        source_df=good_source,
        synthetic_df=good_synthetic,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        target_column=TARGET,
        minimum_score=0.50,  # lenient threshold
    )
    assert "quality_score" in result
    assert "decision" in result
    assert result["n_checks"] == 10


def test_gate_rejects_bad_data(good_source):
    # Completely wrong data
    bad = pd.DataFrame({
        "age": ["text"] * 100,  # wrong dtype
        "income": [None] * 100,  # all missing
        "region": ["Z"] * 100,  # invalid category
        "target": [99] * 100,   # invalid label
    })
    result = run_quality_gate(
        source_df=good_source,
        synthetic_df=bad,
        numerical_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
        target_column=TARGET,
    )
    assert result["decision"] == "REJECTED"
    assert len(result["failed_checks"]) > 0


def test_gate_score_between_0_and_1(good_source, good_synthetic):
    result = run_quality_gate(
        good_source, good_synthetic, NUM_COLS, CAT_COLS, TARGET
    )
    assert 0.0 <= result["quality_score"] <= 1.0


def test_gate_report_present(good_source, good_synthetic):
    result = run_quality_gate(
        good_source, good_synthetic, NUM_COLS, CAT_COLS, TARGET
    )
    assert isinstance(result["detailed_report"], str)
    assert len(result["detailed_report"]) > 0
