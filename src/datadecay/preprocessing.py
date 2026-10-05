"""
preprocessing.py — Chronological preprocessing for DataDecay-X.

CRITICAL: All preprocessing is fitted on training/reference data only.
No information from future windows is allowed to leak into the fitted
transformers.  Random shuffling across temporal boundaries is FORBIDDEN.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .utils import check_column_exists, check_dataframe_not_empty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chronological ordering and splitting
# ---------------------------------------------------------------------------

def sort_by_time(
    df: pd.DataFrame,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sort DataFrame chronologically.

    If timestamp_column is provided and parseable, sort by it.
    Otherwise, preserve existing row order (which must be assumed chronological).

    Returns
    -------
    Sorted (or original-order) DataFrame with reset index.
    """
    if timestamp_column and timestamp_column in df.columns:
        try:
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(
                df[timestamp_column]
            )
            df = df.sort_values(timestamp_column).reset_index(drop=True)
            logger.info("Sorted dataset by '%s'.", timestamp_column)
        except Exception as exc:
            logger.warning(
                "Could not sort by '%s': %s. Using existing row order.",
                timestamp_column,
                exc,
            )
    else:
        logger.info(
            "No timestamp column specified. Assuming dataset is already chronological."
        )
    return df


def chronological_split(
    df: pd.DataFrame,
    n_windows: int = 12,
    test_window_count: int = 2,
    validation_window_count: int = 2,
) -> Dict[str, Any]:
    """
    Split dataset into reference (train), validation, and future-test windows.

    The split is STRICTLY chronological — no shuffling.

    Parameters
    ----------
    df : chronologically sorted DataFrame
    n_windows : total number of temporal windows
    test_window_count : windows reserved for independent future evaluation
    validation_window_count : windows for validation / forecast calibration

    Returns
    -------
    dict with keys:
        reference_df      — earliest portion for baseline training
        validation_df     — middle portion for monitoring / forecast
        future_df         — latest portion held out for final evaluation
        split_info        — split metadata
        window_size       — rows per window
    """
    check_dataframe_not_empty(df, "DataFrame for chronological split")

    n = len(df)
    window_size = max(n // n_windows, 1)

    # Assign window indices
    df = df.copy()
    df["__ddx_window"] = (np.arange(n) // window_size).clip(0, n_windows - 1)

    train_windows = n_windows - test_window_count - validation_window_count
    if train_windows < 2:
        # Fallback: at least 2 train windows
        train_windows = 2
        validation_window_count = max(1, n_windows - train_windows - test_window_count)

    train_mask = df["__ddx_window"] < train_windows
    val_mask = (df["__ddx_window"] >= train_windows) & (
        df["__ddx_window"] < train_windows + validation_window_count
    )
    test_mask = df["__ddx_window"] >= train_windows + validation_window_count

    reference_df = df[train_mask].copy()
    validation_df = df[val_mask].copy()
    future_df = df[test_mask].copy()

    split_info = {
        "total_rows": n,
        "total_windows": n_windows,
        "window_size": window_size,
        "train_windows": int(train_windows),
        "validation_windows": int(validation_window_count),
        "test_windows": int(test_window_count),
        "reference_rows": len(reference_df),
        "validation_rows": len(validation_df),
        "future_rows": len(future_df),
        "train_pct": round(len(reference_df) / n * 100, 1),
        "validation_pct": round(len(validation_df) / n * 100, 1),
        "future_pct": round(len(future_df) / n * 100, 1),
    }

    logger.info(
        "Chronological split — reference: %d, validation: %d, future: %d rows",
        len(reference_df),
        len(validation_df),
        len(future_df),
    )

    return {
        "reference_df": reference_df,
        "validation_df": validation_df,
        "future_df": future_df,
        "full_df": df,
        "split_info": split_info,
        "window_size": window_size,
        "n_train_windows": train_windows,
        "n_val_windows": validation_window_count,
        "n_test_windows": test_window_count,
    }


# ---------------------------------------------------------------------------
# Feature preparation helpers
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    numerical_cols: List[str],
    categorical_cols: List[str],
    timestamp_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target (y).

    Drops the timestamp column and any internal window columns from X.

    Returns
    -------
    X : DataFrame of features
    y : Series of labels
    """
    check_column_exists(df, target_column)
    drop_cols = [target_column]
    if timestamp_column and timestamp_column in df.columns:
        drop_cols.append(timestamp_column)
    # Drop internal metadata columns
    drop_cols += [c for c in df.columns if c.startswith("__ddx_")]

    feature_cols = numerical_cols + categorical_cols
    # Only keep columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_column].copy()
    return X, y


# ---------------------------------------------------------------------------
# Encoding and imputation
# ---------------------------------------------------------------------------

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    fitted_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.

    If fitted_encoders provided, use them (transform only).
    Otherwise fit new ones (fit+transform).

    Returns
    -------
    df_encoded : DataFrame with encoded categoricals
    encoders   : dict of fitted LabelEncoders keyed by column name
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = fitted_encoders or {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("__missing__")
        if col in encoders:
            # Transform only — unseen labels mapped to -1
            le = encoders[col]
            def _safe_transform(le: LabelEncoder, val: str) -> int:
                try:
                    return int(le.transform([val])[0])
                except ValueError:
                    return -1
            df[col] = df[col].apply(lambda v: _safe_transform(le, v))
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


def impute_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    num_strategy: str = "median",
    cat_strategy: str = "most_frequent",
    fitted_imputers: Optional[Dict[str, SimpleImputer]] = None,
) -> Tuple[pd.DataFrame, Dict[str, SimpleImputer]]:
    """
    Impute missing values separately for numerical and categorical columns.

    If fitted_imputers provided, transform only. Otherwise fit+transform.

    Returns
    -------
    df_imputed : DataFrame with imputed values
    imputers   : dict of fitted SimpleImputers
    """
    df = df.copy()
    imputers: Dict[str, SimpleImputer] = fitted_imputers or {}

    for col_group, strategy in [
        (numerical_cols, num_strategy),
        (categorical_cols, cat_strategy),
    ]:
        for col in col_group:
            if col not in df.columns:
                continue
            if df[col].isnull().any():
                key = f"{col}_{strategy}"
                if key in imputers:
                    imp = imputers[key]
                    df[[col]] = imp.transform(df[[col]])
                else:
                    imp = SimpleImputer(strategy=strategy)
                    df[[col]] = imp.fit_transform(df[[col]])
                    imputers[key] = imp

    return df, imputers


# ---------------------------------------------------------------------------
# Sklearn Pipeline-based preprocessor
# ---------------------------------------------------------------------------

def fit_preprocessor(
    X: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    Fit a sklearn ColumnTransformer on training data.

    Numerical: median impute → StandardScaler
    Categorical: most-frequent impute → OneHotEncoder (handle_unknown='ignore')

    CRITICAL: This must ONLY be called on training/reference data.

    Returns
    -------
    preprocessor : fitted ColumnTransformer
    X_transformed : transformed array as DataFrame
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    actual_num = [c for c in numerical_cols if c in X.columns]
    actual_cat = [c for c in categorical_cols if c in X.columns]

    if actual_num:
        transformers.append(("numerical", num_pipeline, actual_num))
    if actual_cat:
        transformers.append(("categorical", cat_pipeline, actual_cat))

    if not transformers:
        raise ValueError("No valid feature columns found to fit preprocessor.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_arr = preprocessor.fit_transform(X)
    feature_names = _get_feature_names(preprocessor, actual_num, actual_cat, X)
    X_transformed = pd.DataFrame(X_arr, columns=feature_names, index=X.index)

    logger.info(
        "Preprocessor fitted — %d numerical, %d categorical → %d output features",
        len(actual_num),
        len(actual_cat),
        X_transformed.shape[1],
    )
    return preprocessor, X_transformed


def transform_with_preprocessor(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a FITTED preprocessor to new data (validation / test windows).

    Never re-fits the preprocessor — avoids temporal leakage.

    Returns
    -------
    X_transformed : DataFrame
    """
    X_arr = preprocessor.transform(X)
    actual_num = [c for c in numerical_cols if c in X.columns]
    actual_cat = [c for c in categorical_cols if c in X.columns]
    feature_names = _get_feature_names(preprocessor, actual_num, actual_cat, X)
    X_transformed = pd.DataFrame(X_arr, columns=feature_names, index=X.index)
    return X_transformed


def _get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
    X: pd.DataFrame,
) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: build manually
        names: List[str] = list(numerical_cols)
        if categorical_cols:
            try:
                cat_step = preprocessor.named_transformers_.get("categorical")
                if cat_step:
                    encoder = cat_step.named_steps.get("encoder")
                    if encoder:
                        for i, col in enumerate(categorical_cols):
                            cats = encoder.categories_[i]
                            names.extend([f"{col}_{c}" for c in cats])
            except Exception:
                names.extend(categorical_cols)
        return names


# ---------------------------------------------------------------------------
# Label encoding for target
# ---------------------------------------------------------------------------

def encode_target(
    y: pd.Series,
    fitted_encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode target labels.

    Parameters
    ----------
    y : target Series
    fitted_encoder : use existing encoder (transform only), or None to fit new.

    Returns
    -------
    y_encoded : numpy array
    label_encoder : fitted LabelEncoder
    """
    if fitted_encoder is None:
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
    else:
        le = fitted_encoder
        y_enc = np.array([
            int(le.transform([str(v)])[0]) if str(v) in le.classes_ else -1
            for v in y
        ])
    return y_enc, le


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

def build_feature_metadata(
    numerical_cols: List[str],
    categorical_cols: List[str],
    preprocessor: Optional[ColumnTransformer] = None,
) -> Dict[str, Any]:
    """Return a structured dict of feature metadata."""
    output_features: List[str] = []
    if preprocessor is not None:
        try:
            output_features = list(preprocessor.get_feature_names_out())
        except Exception:
            output_features = numerical_cols + categorical_cols

    return {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "n_numerical": len(numerical_cols),
        "n_categorical": len(categorical_cols),
        "output_features": output_features,
        "n_output_features": len(output_features),
    }
