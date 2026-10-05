"""
database.py — SQLite experiment history for DataDecay-X.

Stores experiment metadata and results for comparison and reproducibility.
Large DataFrames are never stored directly — only references/paths.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, _NumpyEncoder

logger = logging.getLogger(__name__)

# Schema version
SCHEMA_VERSION = 1


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path) -> None:
    """
    Create all tables if they do not exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = _get_connection(db_path)
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dataset_name TEXT,
                dataset_hash TEXT,
                dataset_rows INTEGER,
                dataset_cols INTEGER,
                model_name TEXT,
                drift_type TEXT,
                affected_features TEXT,
                n_windows INTEGER,
                random_seed INTEGER,
                status TEXT DEFAULT 'pending',
                config_json TEXT,
                notes TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS drift_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                window INTEGER,
                drift_severity_index REAL,
                severity_label TEXT,
                max_psi REAL,
                mean_psi REAL,
                max_ks REAL,
                max_wasserstein REAL,
                n_drifted_features INTEGER,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS reliability_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                window INTEGER,
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1 REAL,
                health_score REAL,
                below_threshold INTEGER,
                degradation_pct REAL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                forecast_method TEXT,
                forecast_horizon INTEGER,
                current_reliability REAL,
                baseline_reliability REAL,
                degradation_pct REAL,
                drift_severity_index REAL,
                forecast_trend TEXT,
                crossing_detected INTEGER,
                crossing_window INTEGER,
                warning_lead_windows INTEGER,
                ttd_estimate TEXT,
                forecast_state TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS recovery_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                strategy TEXT,
                future_accuracy REAL,
                future_f1 REAL,
                future_precision REAL,
                future_recall REAL,
                recovery_gain REAL,
                remaining_degradation REAL,
                training_time_s REAL,
                n_synthetic INTEGER,
                n_real INTEGER,
                intervention_size INTEGER,
                decision TEXT,
                synthetic_quality_score REAL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiment_summary (
                experiment_id TEXT PRIMARY KEY,
                baseline_train_f1 REAL,
                baseline_future_f1 REAL,
                best_strategy TEXT,
                best_future_f1 REAL,
                overall_recovery_gain REAL,
                final_decision TEXT,
                final_explanation TEXT,
                quality_score REAL,
                quality_decision TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        conn.commit()
        logger.debug("Database schema initialized at %s", db_path)

    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)
        raise
    finally:
        conn.close()


def save_experiment(
    db_path: Path,
    experiment_id: str,
    experiment_name: str,
    config: Dict[str, Any],
) -> None:
    """Insert or replace experiment metadata row."""
    conn = _get_connection(db_path)
    try:
        now = datetime.now(tz=timezone.utc).isoformat()
        config_json = json.dumps(config, cls=_NumpyEncoder)

        conn.execute("""
            INSERT OR REPLACE INTO experiments
            (experiment_id, experiment_name, created_at,
             dataset_name, dataset_hash, dataset_rows, dataset_cols,
             model_name, drift_type, affected_features,
             n_windows, random_seed, status, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            experiment_name,
            now,
            config.get("dataset_name", ""),
            config.get("dataset_hash", ""),
            config.get("dataset_rows", 0),
            config.get("dataset_cols", 0),
            config.get("model_name", ""),
            config.get("drift_type", ""),
            json.dumps(config.get("affected_features", [])),
            config.get("n_windows", 0),
            config.get("random_seed", 42),
            "running",
            config_json,
        ))
        conn.commit()
        logger.info("Experiment %s saved to database.", experiment_id)
    finally:
        conn.close()


def save_drift_results(
    db_path: Path,
    experiment_id: str,
    window_drift_table: Any,  # pd.DataFrame
) -> None:
    """Save per-window drift metrics."""
    conn = _get_connection(db_path)
    try:
        for _, row in window_drift_table.iterrows():
            conn.execute("""
                INSERT INTO drift_results
                (experiment_id, window, drift_severity_index, severity_label,
                 max_psi, mean_psi, max_ks, max_wasserstein, n_drifted_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                int(row.get("window", 0)),
                float(row.get("drift_severity_index", 0.0)),
                str(row.get("severity_label", "")),
                float(row.get("max_psi", 0.0)),
                float(row.get("mean_psi", 0.0)),
                float(row.get("max_ks", 0.0)),
                float(row.get("max_wasserstein", 0.0)),
                int(row.get("n_drifted_features", 0)),
            ))
        conn.commit()
    finally:
        conn.close()


def save_reliability_results(
    db_path: Path,
    experiment_id: str,
    reliability_timeline: Any,  # pd.DataFrame
    model_name: str = "",
) -> None:
    """Save per-window reliability metrics."""
    conn = _get_connection(db_path)
    try:
        for _, row in reliability_timeline.iterrows():
            conn.execute("""
                INSERT INTO reliability_results
                (experiment_id, window, model_name, accuracy,
                 precision_score, recall, f1, health_score,
                 below_threshold, degradation_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                int(row.get("window", 0)),
                model_name,
                float(row.get("accuracy", 0.0)),
                float(row.get("precision", 0.0)),
                float(row.get("recall", 0.0)),
                float(row.get("f1", 0.0)),
                float(row.get("health_score", 0.0)),
                int(bool(row.get("below_threshold", False))),
                float(row.get("degradation_pct", 0.0)),
            ))
        conn.commit()
    finally:
        conn.close()


def save_forecast_results(
    db_path: Path,
    experiment_id: str,
    forecast_result: Dict[str, Any],
    ttd_result: Dict[str, Any],
    intervention_result: Dict[str, Any],
) -> None:
    """Save forecast and intervention state."""
    conn = _get_connection(db_path)
    try:
        evidence = intervention_result.get("evidence", {})
        conn.execute("""
            INSERT OR REPLACE INTO forecast_results
            (experiment_id, forecast_method, forecast_horizon,
             current_reliability, baseline_reliability, degradation_pct,
             drift_severity_index, forecast_trend, crossing_detected,
             crossing_window, warning_lead_windows, ttd_estimate, forecast_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            str(forecast_result.get("method", "")),
            int(forecast_result.get("horizon", 0)),
            float(ttd_result.get("current_reliability", 0.0)),
            float(evidence.get("baseline_reliability", 0.0)),
            float(evidence.get("degradation_pct", 0.0)),
            float(evidence.get("drift_severity_index", 0.0)),
            str(evidence.get("forecast_trend", "")),
            int(bool(ttd_result.get("crossing_detected", False))),
            ttd_result.get("crossing_window"),
            int(ttd_result.get("warning_lead_windows", 0)),
            str(ttd_result.get("ttd_estimate", "")),
            str(intervention_result.get("state", "")),
        ))
        conn.commit()
    finally:
        conn.close()


def save_recovery_results(
    db_path: Path,
    experiment_id: str,
    comparison_df: Any,  # pd.DataFrame
    quality_score: float = 0.0,
) -> None:
    """Save recovery strategy comparison results."""
    conn = _get_connection(db_path)
    try:
        for _, row in comparison_df.iterrows():
            conn.execute("""
                INSERT INTO recovery_results
                (experiment_id, strategy, future_accuracy, future_f1,
                 future_precision, future_recall, recovery_gain,
                 remaining_degradation, training_time_s, n_synthetic,
                 n_real, intervention_size, decision, synthetic_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                str(row.get("strategy", "")),
                float(row.get("future_accuracy", 0.0)),
                float(row.get("future_f1", 0.0)),
                float(row.get("future_f1", 0.0)),  # precision placeholder
                float(row.get("future_f1", 0.0)),  # recall placeholder
                float(row.get("recovery_gain", 0.0)),
                float(row.get("remaining_degradation", 0.0)),
                float(row.get("training_time_s", 0.0)),
                int(row.get("n_synthetic", 0)),
                int(row.get("n_real", 0)),
                int(row.get("intervention_size", 0)),
                str(row.get("decision", "")),
                float(quality_score),
            ))
        conn.commit()
    finally:
        conn.close()


def save_experiment_summary(
    db_path: Path,
    experiment_id: str,
    eval_summary: Dict[str, Any],
    quality_score: float = 0.0,
    quality_decision: str = "",
) -> None:
    """Save final experiment summary and update experiment status."""
    conn = _get_connection(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO experiment_summary
            (experiment_id, baseline_train_f1, baseline_future_f1,
             best_strategy, best_future_f1, overall_recovery_gain,
             final_decision, final_explanation, quality_score, quality_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            float(eval_summary.get("baseline_train_f1", 0.0)),
            float(eval_summary.get("baseline_future_f1", 0.0)),
            str(eval_summary.get("best_strategy", "")),
            float(eval_summary.get("best_future_f1", 0.0)),
            float(eval_summary.get("overall_recovery_gain", 0.0)),
            str(eval_summary.get("final_decision", "")),
            str(eval_summary.get("final_explanation", "")),
            float(quality_score),
            str(quality_decision),
        ))

        # Update experiment status
        conn.execute("""
            UPDATE experiments SET status = ? WHERE experiment_id = ?
        """, ("completed", experiment_id))

        conn.commit()
        logger.info("Experiment %s summary saved.", experiment_id)
    finally:
        conn.close()


def get_experiment_history(db_path: Path, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve a summary list of all experiments for the history table.

    Returns list of dicts with key experiment metadata.
    """
    if not db_path.exists():
        return []

    conn = _get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT
                e.experiment_id,
                e.experiment_name,
                e.created_at,
                e.dataset_name,
                e.model_name,
                e.drift_type,
                e.n_windows,
                e.random_seed,
                e.status,
                s.baseline_train_f1,
                s.baseline_future_f1,
                s.best_strategy,
                s.best_future_f1,
                s.overall_recovery_gain,
                s.final_decision,
                s.quality_score,
                f.forecast_state,
                f.ttd_estimate,
                f.drift_severity_index
            FROM experiments e
            LEFT JOIN experiment_summary s ON e.experiment_id = s.experiment_id
            LEFT JOIN forecast_results f ON e.experiment_id = f.experiment_id
            ORDER BY e.created_at DESC
            LIMIT ?
        """, (limit,))

        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as exc:
        logger.error("get_experiment_history failed: %s", exc)
        return []
    finally:
        conn.close()


def get_experiment(db_path: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full experiment record including all sub-tables."""
    if not db_path.exists():
        return None

    conn = _get_connection(db_path)
    try:
        exp_row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()

        if not exp_row:
            return None

        result = dict(exp_row)

        # Parse config_json
        try:
            result["config"] = json.loads(result.get("config_json") or "{}")
        except Exception:
            result["config"] = {}

        # Load sub-tables
        drift = conn.execute(
            "SELECT * FROM drift_results WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
        result["drift_results"] = [dict(r) for r in drift]

        reliability = conn.execute(
            "SELECT * FROM reliability_results WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
        result["reliability_results"] = [dict(r) for r in reliability]

        forecast = conn.execute(
            "SELECT * FROM forecast_results WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        result["forecast_results"] = dict(forecast) if forecast else {}

        recovery = conn.execute(
            "SELECT * FROM recovery_results WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
        result["recovery_results"] = [dict(r) for r in recovery]

        summary = conn.execute(
            "SELECT * FROM experiment_summary WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        result["summary"] = dict(summary) if summary else {}

        return result

    except Exception as exc:
        logger.error("get_experiment failed for %s: %s", experiment_id, exc)
        return None
    finally:
        conn.close()


def delete_experiment(db_path: Path, experiment_id: str) -> bool:
    """Delete an experiment and all associated records."""
    if not db_path.exists():
        return False

    conn = _get_connection(db_path)
    try:
        tables = [
            "experiment_summary",
            "recovery_results",
            "forecast_results",
            "reliability_results",
            "drift_results",
            "experiments",
        ]
        for table in tables:
            conn.execute(
                f"DELETE FROM {table} WHERE experiment_id = ?", (experiment_id,)
            )
        conn.commit()
        logger.info("Experiment %s deleted from database.", experiment_id)
        return True
    except Exception as exc:
        logger.error("delete_experiment failed for %s: %s", experiment_id, exc)
        return False
    finally:
        conn.close()
