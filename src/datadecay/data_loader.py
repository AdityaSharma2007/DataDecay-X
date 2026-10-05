"""
data_loader.py — Dataset loading and validation for DataDecay-X.

Handles CSV loading, feature type inference, target/timestamp detection,
dataset summary, and multi-layer validation.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    check_column_exists,
    check_dataframe_not_empty,
    check_no_duplicate_columns,
    dataframe_hash,
    infer_column_types,
    missing_value_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    source: Union[str, Path, StringIO],
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file or StringIO buffer into a DataFrame.

    Parameters
    ----------
    source : file path or StringIO
    encoding : file encoding (default utf-8)

    Returns
    -------
    pd.DataFrame
    """
    try:
        if isinstance(source, StringIO):
            df = pd.read_csv(source, **kwargs)
        else:
            df = pd.read_csv(str(source), encoding=encoding, **kwargs)
        logger.info("Loaded dataset: %d rows × %d columns", len(df), len(df.columns))
        return df
    except Exception as exc:
        raise ValueError(f"Failed to load CSV: {exc}") from exc


def validate_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    min_rows: int = 50,
) -> Dict[str, Any]:
    """
    Run multi-layer validation on a dataset.

    Returns
    -------
    dict with keys:
        valid (bool)
        errors (list of str)
        warnings (list of str)
        info (list of str)
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    # --- Basic shape ---
    if df is None or len(df) == 0:
        errors.append("Dataset is empty.")
        return {"valid": False, "errors": errors, "warnings": warnings, "info": info}

    if len(df) < min_rows:
        errors.append(
            f"Dataset has only {len(df)} rows; minimum required is {min_rows}."
        )

    # --- Duplicate columns ---
    dupes = check_no_duplicate_columns(df)
    if dupes:
        errors.append(f"Duplicate column names detected: {dupes}")

    # --- Target column ---
    if target_column:
        if target_column not in df.columns:
            errors.append(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        else:
            n_classes = df[target_column].nunique()
            if n_classes < 2:
                errors.append(
                    f"Target column '{target_column}' has only {n_classes} unique value(s). "
                    "Need at least 2 classes for classification."
                )
            elif n_classes > 50:
                warnings.append(
                    f"Target column '{target_column}' has {n_classes} unique values. "
                    "Verify this is a classification target."
                )
            null_target = df[target_column].isnull().sum()
            if null_target > 0:
                errors.append(
                    f"Target column '{target_column}' has {null_target} missing values."
                )

    # --- Timestamp column ---
    if timestamp_column:
        if timestamp_column not in df.columns:
            warnings.append(
                f"Timestamp column '{timestamp_column}' not found; "
                "row order will be used as temporal order."
            )
        else:
            try:
                parsed = pd.to_datetime(df[timestamp_column])
                n_nulls = parsed.isnull().sum()
                if n_nulls > 0:
                    warnings.append(
                        f"Timestamp column '{timestamp_column}' has {n_nulls} unparseable values."
                    )
            except Exception:
                warnings.append(
                    f"Could not parse '{timestamp_column}' as datetime. "
                    "Row order will be used."
                )

    # --- Missing values ---
    missing = df.isnull().sum()
    total_missing = missing.sum()
    pct_missing = total_missing / (len(df) * len(df.columns)) * 100
    if pct_missing > 30:
        errors.append(f"Dataset has {pct_missing:.1f}% missing values (>30%). Check data quality.")
    elif pct_missing > 10:
        warnings.append(f"Dataset has {pct_missing:.1f}% missing values.")
    elif pct_missing > 0:
        info.append(f"Dataset has {total_missing} missing cells ({pct_missing:.1f}%).")

    high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.5]
    if high_missing_cols:
        warnings.append(f"Columns with >50% missing: {high_missing_cols}")

    # --- All-constant columns ---
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        warnings.append(f"Constant (zero-variance) columns detected: {constant_cols}")

    # --- Data type support ---
    col_types = infer_column_types(df)
    unsupported = [c for c, t in col_types.items() if t == "other"]
    if unsupported:
        warnings.append(f"Columns with unsupported dtype (will be dropped): {unsupported}")

    # --- Duplicated rows ---
    n_dupe_rows = df.duplicated().sum()
    if n_dupe_rows > 0:
        pct_dupe = n_dupe_rows / len(df) * 100
        if pct_dupe > 20:
            warnings.append(f"{n_dupe_rows} duplicate rows ({pct_dupe:.1f}%).")
        else:
            info.append(f"{n_dupe_rows} duplicate rows ({pct_dupe:.1f}%).")

    # --- Minimum columns ---
    if len(df.columns) < 3:
        errors.append("Dataset has fewer than 3 columns. Need features + target at minimum.")

    valid = len(errors) == 0
    return {"valid": valid, "errors": errors, "warnings": warnings, "info": info}


def infer_feature_types(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Infer and return lists of numerical and categorical feature columns.

    Parameters
    ----------
    df : input DataFrame
    exclude_cols : columns to exclude (e.g. target, timestamp)

    Returns
    -------
    dict with keys 'numerical' and 'categorical'
    """
    exclude = set(exclude_cols or [])
    col_types = infer_column_types(df)

    numerical = [
        c for c, t in col_types.items()
        if t == "numerical" and c not in exclude
    ]
    categorical = [
        c for c, t in col_types.items()
        if t == "categorical" and c not in exclude
    ]

    logger.debug(
        "Feature types — numerical: %d, categorical: %d",
        len(numerical),
        len(categorical),
    )
    return {"numerical": numerical, "categorical": categorical}


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically detect the most likely target column.

    Checks common names first, then falls back to the last column
    if it looks like a binary classification target.
    """
    common_target_names = [
        "target", "label", "y", "class", "outcome", "churn",
        "default", "fraud", "output", "response",
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    for name in common_target_names:
        if name in cols_lower:
            return cols_lower[name]

    # Fallback: last column with binary-ish cardinality
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 10:
        return last_col

    return None


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically detect a timestamp column.
    """
    common_ts_names = [
        "timestamp", "date", "time", "datetime", "ts",
        "created_at", "event_time", "obs_date",
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    for name in common_ts_names:
        if name in cols_lower:
            col = cols_lower[name]
            # Verify it looks like a datetime
            try:
                pd.to_datetime(df[col].head(20))
                return col
            except Exception:
                continue

    # Fallback: scan for datetime-parseable columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample)
                if parsed.notnull().all():
                    return col
            except Exception:
                continue

    return None


def dataset_summary(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce a structured metadata summary of the dataset.

    Returns
    -------
    dict with dataset statistics and metadata
    """
    col_types = infer_column_types(df)
    exclude = {c for c in [target_column, timestamp_column] if c}
    feature_types = infer_feature_types(df, exclude_cols=list(exclude))

    # Missing summary
    missing_df = missing_value_summary(df)
    cols_with_missing = int((missing_df["missing_count"] > 0).sum())

    # Time range
    time_range: Dict[str, Any] = {}
    if timestamp_column and timestamp_column in df.columns:
        try:
            ts = pd.to_datetime(df[timestamp_column])
            time_range = {
                "min": str(ts.min()),
                "max": str(ts.max()),
                "range_days": (ts.max() - ts.min()).days,
            }
        except Exception:
            time_range = {"error": "Could not parse timestamp column"}

    # Target distribution
    target_dist: Dict[str, Any] = {}
    if target_column and target_column in df.columns:
        vc = df[target_column].value_counts()
        target_dist = {
            "n_classes": int(vc.shape[0]),
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "class_balance": {
                str(k): round(v / len(df), 4) for k, v in vc.items()
            },
        }

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_numerical_features": len(feature_types["numerical"]),
        "n_categorical_features": len(feature_types["categorical"]),
        "numerical_features": feature_types["numerical"],
        "categorical_features": feature_types["categorical"],
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "n_missing_cells": int(df.isnull().sum().sum()),
        "pct_missing": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "cols_with_missing": cols_with_missing,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "column_types": col_types,
        "time_range": time_range,
        "target_distribution": target_dist,
        "dataset_hash": dataframe_hash(df),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def prepare_upload(
    uploaded_content: Union[str, bytes, StringIO],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Parse uploaded file content (from Streamlit file_uploader) into DataFrame.

    Parameters
    ----------
    uploaded_content : raw bytes, string, or StringIO from st.file_uploader
    encoding : text encoding

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(uploaded_content, bytes):
        content_str = uploaded_content.decode(encoding, errors="replace")
        return pd.read_csv(StringIO(content_str))
    elif isinstance(uploaded_content, str):
        return pd.read_csv(StringIO(uploaded_content))
    elif isinstance(uploaded_content, StringIO):
        return pd.read_csv(uploaded_content)
    else:
        # Streamlit UploadedFile object
        return pd.read_csv(uploaded_content)
