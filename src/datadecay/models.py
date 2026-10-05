"""
models.py — Baseline ML models for DataDecay-X.

Provides create/train/predict/evaluate/save/load for:
  - Logistic Regression
  - Support Vector Machine
  - Naive Bayes
  - Random Forest
  - XGBoost (optional)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Model name constants
MODEL_LR = "logistic_regression"
MODEL_SVM = "svm"
MODEL_NB = "naive_bayes"
MODEL_RF = "random_forest"
MODEL_XGB = "xgboost"

AVAILABLE_MODELS = [MODEL_LR, MODEL_SVM, MODEL_NB, MODEL_RF]

# Try importing optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
    AVAILABLE_MODELS.append(MODEL_XGB)
except ImportError:
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    model_name: str,
    random_seed: int = 42,
    **kwargs,
) -> Any:
    """
    Create and return an unfitted sklearn-compatible classifier.

    Parameters
    ----------
    model_name  : one of MODEL_LR, MODEL_SVM, MODEL_NB, MODEL_RF, MODEL_XGB
    random_seed : reproducibility seed
    **kwargs    : override default hyperparameters

    Returns
    -------
    Unfitted sklearn estimator
    """
    name = model_name.lower().strip()

    if name == MODEL_LR:
        params = {"max_iter": 1000, "C": 1.0, "random_state": random_seed}
        params.update(kwargs)
        return LogisticRegression(**params)

    elif name == MODEL_SVM:
        params = {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": random_seed}
        params.update(kwargs)
        return SVC(**params)

    elif name == MODEL_NB:
        params = {"var_smoothing": 1.0e-9}
        params.update(kwargs)
        return GaussianNB(**params)

    elif name == MODEL_RF:
        if not _RF_AVAILABLE:
            logger.warning("RandomForest not available; falling back to LogisticRegression.")
            return LogisticRegression(max_iter=1000, random_state=random_seed)
        params = {"n_estimators": 100, "max_depth": 10, "random_state": random_seed, "n_jobs": -1}
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif name == MODEL_XGB:
        if not _XGB_AVAILABLE:
            logger.warning("XGBoost not available; falling back to LogisticRegression.")
            return LogisticRegression(max_iter=1000, random_state=random_seed)
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": random_seed,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return XGBClassifier(**params)

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {AVAILABLE_MODELS}"
        )


def train_model(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> Tuple[Any, float]:
    """
    Fit the model and return (fitted_model, training_time_seconds).
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(
        "Model %s trained in %.2f s on %d samples",
        type(model).__name__,
        elapsed,
        len(y_train),
    )
    return model, elapsed


def predict_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Return class predictions.
    """
    return model.predict(X)


def predict_proba_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> Optional[np.ndarray]:
    """
    Return class probabilities if available, else None.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def evaluate_model(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    average: str = "weighted",
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Returns
    -------
    dict with accuracy, precision, recall, f1, confusion_matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge case: all predictions same class
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)

    try:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception as exc:
        logger.warning("Metric computation error: %s", exc)
        acc = prec = rec = f1 = 0.0
        cm = []

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "n_classes": int(len(unique_true)),
    }


def save_model(
    model: Any,
    path: Union[str, Path],
) -> Path:
    """
    Persist model to disk using joblib.

    Returns
    -------
    path where model was saved
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("Model saved to %s", p)
    return p


def load_model(path: Union[str, Path]) -> Any:
    """
    Load a model previously saved with save_model.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = joblib.load(p)
    logger.info("Model loaded from %s", p)
    return model


# ---------------------------------------------------------------------------
# Multi-model training
# ---------------------------------------------------------------------------

def train_multiple_models(
    model_names: List[str],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    random_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple models on train + validation sets.

    Returns
    -------
    dict keyed by model_name with:
        model, metrics (on val), training_time
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        try:
            clf = create_model(name, random_seed=random_seed)
            clf, train_time = train_model(clf, X_train, y_train)
            y_pred = predict_model(clf, X_val)
            metrics = evaluate_model(y_val, y_pred)
            metrics["training_time"] = round(train_time, 4)

            results[name] = {
                "model": clf,
                "metrics": metrics,
                "training_time": train_time,
                "model_name": name,
            }
            logger.info(
                "Model %s — val F1=%.4f, acc=%.4f (%.2fs)",
                name, metrics["f1"], metrics["accuracy"], train_time,
            )
        except Exception as exc:
            logger.error("Failed to train model '%s': %s", name, exc)
            results[name] = {
                "model": None,
                "metrics": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0},
                "training_time": 0,
                "model_name": name,
                "error": str(exc),
            }

    return results


def select_best_model(
    model_results: Dict[str, Dict[str, Any]],
    metric: str = "f1",
) -> Tuple[str, Any, Dict]:
    """
    Select the best model by a given metric.

    Returns
    -------
    (model_name, model, metrics)
    """
    best_name = None
    best_score = -1.0
    best_model = None
    best_metrics: Dict = {}

    for name, result in model_results.items():
        if result.get("model") is None:
            continue
        score = result["metrics"].get(metric, 0.0)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = result["model"]
            best_metrics = result["metrics"]

    if best_name is None:
        raise ValueError("No valid model found in results.")

    logger.info("Best model: %s (%s=%.4f)", best_name, metric, best_score)
    return best_name, best_model, best_metrics


def model_summary_table(
    model_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a summary DataFrame from multi-model results.
    """
    rows = []
    for name, result in model_results.items():
        m = result.get("metrics", {})
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "training_time_s": round(result.get("training_time", 0), 3),
            "error": result.get("error", ""),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
