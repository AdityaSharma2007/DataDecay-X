"""
config.py — Configuration management for DataDecay-X.

Loads config.yaml, provides defaults, validates configuration,
and exposes a singleton Config object to all modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (used when config.yaml is missing or incomplete)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "project": {
        "name": "DataDecay-X",
        "version": "1.0.0",
        "description": (
            "Predictive Model Reliability Forecasting and Proactive Synthetic "
            "Remediation under Temporal Distribution Shift"
        ),
    },
    "random_seed": 42,
    "temporal": {
        "n_windows": 12,
        "forecast_horizon": 3,
        "test_window_count": 2,
        "validation_window_count": 2,
    },
    "reliability": {
        "metric": "f1",
        "threshold": 0.70,
        "degradation_warning": 0.10,
        "degradation_critical": 0.20,
    },
    "drift": {
        "psi_threshold_moderate": 0.10,
        "psi_threshold_severe": 0.20,
        "ks_alpha": 0.05,
        "wasserstein_threshold": 0.10,
        "severity_weights": {
            "psi": 0.4,
            "ks": 0.3,
            "wasserstein": 0.3,
        },
    },
    "models": {
        "default": "logistic_regression",
        "available": ["logistic_regression", "svm", "naive_bayes", "random_forest"],
        "logistic_regression": {"max_iter": 1000, "C": 1.0},
        "svm": {"C": 1.0, "kernel": "rbf", "probability": True},
        "naive_bayes": {"var_smoothing": 1.0e-9},
        "random_forest": {"n_estimators": 100, "max_depth": 10},
    },
    "synthetic": {
        "default_samples": 500,
        "perturbation_scale": 0.05,
        "default_method": "perturbation",
        "methods": ["resampling", "perturbation", "categorical_sampling", "hybrid"],
    },
    "quality_gate": {
        "minimum_score": 0.70,
        "missing_threshold": 0.05,
        "duplicate_threshold": 0.10,
        "distribution_threshold": 0.20,
        "class_imbalance_ratio": 0.10,
    },
    "forecast": {
        "methods": ["persistence", "linear_trend"],
        "default_method": "linear_trend",
        "warning_lead_windows": 2,
    },
    "database": {
        "path": "data/datadecay.db",
    },
    "directories": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "data_synthetic": "data/synthetic",
        "models": "models",
        "reports": "reports",
        "logs": "logs",
        "experiments": "experiments",
    },
    "logging": {
        "level": "INFO",
        "file": "logs/datadecay.log",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """
    Singleton configuration object for DataDecay-X.

    Loads config.yaml from the project root and merges with defaults.
    Access values via attribute-style or dict-style access.
    """

    _instance: Optional["Config"] = None

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._data: Dict[str, Any] = _DEFAULTS.copy()
        self._config_path = config_path or self._find_config_file()
        self._load()
        self._validate()

    # ------------------------------------------------------------------
    # Class-level singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> "Config":
        """Return the singleton Config instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_config_file(self) -> Optional[Path]:
        """Search for config.yaml in known locations."""
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()
        return None

    def _load(self) -> None:
        """Load and merge YAML configuration."""
        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                self._data = _deep_merge(_DEFAULTS, loaded)
                logger.debug("Config loaded from %s", self._config_path)
            except yaml.YAMLError as exc:
                logger.warning("Could not parse config.yaml: %s — using defaults.", exc)
        else:
            logger.warning("config.yaml not found — using built-in defaults.")

    def _validate(self) -> None:
        """Validate critical configuration values."""
        seed = self._data.get("random_seed", 42)
        if not isinstance(seed, int) or seed < 0:
            logger.warning("random_seed should be a non-negative integer; resetting to 42.")
            self._data["random_seed"] = 42

        threshold = self._data.get("reliability", {}).get("threshold", 0.70)
        if not (0.0 < threshold < 1.0):
            logger.warning("reliability.threshold out of range [0,1]; resetting to 0.70.")
            self._data["reliability"]["threshold"] = 0.70

        n_windows = self._data.get("temporal", {}).get("n_windows", 12)
        if not isinstance(n_windows, int) or n_windows < 4:
            logger.warning("temporal.n_windows must be >= 4; resetting to 12.")
            self._data["temporal"]["n_windows"] = 12

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def random_seed(self) -> int:
        return int(self._data.get("random_seed", 42))

    @property
    def n_windows(self) -> int:
        return int(self._data["temporal"]["n_windows"])

    @property
    def forecast_horizon(self) -> int:
        return int(self._data["temporal"]["forecast_horizon"])

    @property
    def test_window_count(self) -> int:
        return int(self._data["temporal"].get("test_window_count", 2))

    @property
    def validation_window_count(self) -> int:
        return int(self._data["temporal"].get("validation_window_count", 2))

    @property
    def reliability_threshold(self) -> float:
        return float(self._data["reliability"]["threshold"])

    @property
    def reliability_metric(self) -> str:
        return str(self._data["reliability"].get("metric", "f1"))

    @property
    def psi_threshold_moderate(self) -> float:
        return float(self._data["drift"]["psi_threshold_moderate"])

    @property
    def psi_threshold_severe(self) -> float:
        return float(self._data["drift"]["psi_threshold_severe"])

    @property
    def ks_alpha(self) -> float:
        return float(self._data["drift"]["ks_alpha"])

    @property
    def wasserstein_threshold(self) -> float:
        return float(self._data["drift"]["wasserstein_threshold"])

    @property
    def default_synthetic_samples(self) -> int:
        return int(self._data["synthetic"]["default_samples"])

    @property
    def perturbation_scale(self) -> float:
        return float(self._data["synthetic"]["perturbation_scale"])

    @property
    def quality_gate_minimum_score(self) -> float:
        return float(self._data["quality_gate"]["minimum_score"])

    @property
    def database_path(self) -> Path:
        raw = self._data["database"]["path"]
        # Resolve relative to project root
        p = Path(raw)
        if not p.is_absolute():
            root = Path(__file__).parent.parent.parent
            p = root / p
        return p

    @property
    def log_level(self) -> str:
        return str(self._data["logging"].get("level", "INFO"))

    @property
    def log_file(self) -> Path:
        raw = self._data["logging"].get("file", "logs/datadecay.log")
        p = Path(raw)
        if not p.is_absolute():
            root = Path(__file__).parent.parent.parent
            p = root / p
        return p

    def dir(self, name: str) -> Path:
        """Return a directory path by logical name (e.g. 'models', 'reports')."""
        raw = self._data["directories"].get(name, name)
        p = Path(raw)
        if not p.is_absolute():
            root = Path(__file__).parent.parent.parent
            p = root / p
        return p

    def as_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a plain dict."""
        return self._data.copy()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

def get_config() -> Config:
    """Return the global Config singleton."""
    return Config.get_instance()
