"""
utils.py — General reusable utilities for DataDecay-X.

Contains helpers for directory management, JSON/CSV I/O, random seeds,
timestamps, experiment IDs, and dataframe validation.
No business logic lives here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs(*paths: Union[str, Path]) -> List[Path]:
    """Create multiple directories. Return list of Paths."""
    return [ensure_dir(p) for p in paths]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and pandas Timestamps."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """Serialize data to JSON file. Returns the path written."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, cls=_NumpyEncoder, ensure_ascii=False)
    logger.debug("JSON saved to %s", p)
    return p


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from file. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> Path:
    """Save DataFrame to CSV. Returns path."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    logger.debug("CSV saved to %s (%d rows)", p, len(df))
    return p


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV into DataFrame. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    return pd.read_csv(p, **kwargs)


# ---------------------------------------------------------------------------
# Random seed helpers
# ---------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy (and optionally torch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Optional torch seed
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.debug("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def now_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current UTC time as formatted string."""
    return datetime.now(tz=timezone.utc).strftime(fmt)


def iso_now() -> str:
    """Return current UTC datetime as ISO 8601 string."""
    return now_utc().isoformat()


# ---------------------------------------------------------------------------
# Experiment ID helpers
# ---------------------------------------------------------------------------

def generate_experiment_id(prefix: str = "EXP") -> str:
    """Generate a short unique experiment ID."""
    ts = now_str("%Y%m%d%H%M%S")
    uid = uuid.uuid4().hex[:6].upper()
    return f"{prefix}_{ts}_{uid}"


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------

def dataframe_hash(df: pd.DataFrame, n_rows: int = 1000) -> str:
    """
    Compute a lightweight reproducible hash of a DataFrame.
    Uses shape, column names, and a sample of the data.
    """
    sample = df.head(n_rows)
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    cols_str = ",".join(df.columns.tolist())
    data_str = sample.to_csv(index=False)
    combined = f"{shape_str}|{cols_str}|{data_str}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Safe numeric conversion
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_round(value: Any, decimals: int = 4) -> float:
    """Round to decimals safely."""
    return round(safe_float(value), decimals)


# ---------------------------------------------------------------------------
# DataFrame validation helpers
# ---------------------------------------------------------------------------

def check_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Raise ValueError if DataFrame is empty."""
    if df is None or len(df) == 0:
        raise ValueError(f"{name} is empty.")


def check_column_exists(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if column does not exist."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")


def check_no_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Return list of duplicate column names (empty if none)."""
    seen: Dict[str, int] = {}
    dupes: List[str] = []
    for col in df.columns:
        seen[col] = seen.get(col, 0) + 1
    for col, count in seen.items():
        if count > 1:
            dupes.append(col)
    return dupes


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing value counts and percentages."""
    counts = df.isnull().sum()
    pct = counts / len(df) * 100
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct}).sort_values(
        "missing_pct", ascending=False
    )


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify each column as 'numerical', 'categorical', 'datetime', or 'other'.
    """
    result: Dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            result[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result[col] = "datetime"
        elif pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            result[col] = "categorical"
        else:
            result[col] = "other"
    return result


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def dict_to_flat_str(d: Dict[str, Any], sep: str = ", ") -> str:
    """Convert a flat dict to a readable key=value string."""
    return sep.join(f"{k}={v}" for k, v in d.items())


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    items: List[Tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure root logger with console handler and optional file handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        lp = Path(log_file)
        ensure_dir(lp.parent)
        file_handler = logging.FileHandler(lp, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, format=fmt, handlers=handlers, force=True)
    logger.debug("Logging initialised at level %s", level)
