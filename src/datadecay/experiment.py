"""
experiment.py — Pipeline orchestration for DataDecay-X.

run_experiment() is the single entry point that executes the complete pipeline:
  dataset → preprocessing → baseline → windows → drift → detection →
  monitoring → forecasting → TTD → intervention → synthetic → quality gate →
  recovery → future evaluation → comparison → save → report

Business logic only. No Streamlit code here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_config
from .data_loader import dataset_summary, infer_feature_types, validate_dataset
from .database import (
    initialize_database,
    save_drift_results,
    save_experiment,
    save_experiment_summary,
    save_forecast_results,
    save_recovery_results,
    save_reliability_results,
)
from .detection import detect_drift_across_windows
from .evaluation import build_evaluation_summary, evaluate_future_window
from .forecasting import (
    estimate_threshold_crossing,
    forecast_reliability,
    compare_forecasters,
)
from .intervention import determine_intervention
from .models import (
    create_model,
    evaluate_model,
    predict_model,
    train_model,
    train_multiple_models,
    model_summary_table,
)
from .monitoring import (
    build_reliability_timeline,
    calculate_health_score,
    calculate_reliability_degradation,
    calculate_time_below_threshold,
    evaluate_model_by_window,
)
from .preprocessing import (
    chronological_split,
    encode_target,
    fit_preprocessor,
    prepare_features,
    sort_by_time,
    transform_with_preprocessor,
)
from .quality_gate import run_quality_gate
from .recovery import (
    compare_recovery_strategies,
    run_hybrid_recovery,
    run_no_recovery,
    run_real_only_recovery,
    run_synthetic_only_recovery,
)
from .reporting import generate_experiment_report, save_report_csv, save_report_json
from .simulator import create_temporal_windows, simulate_drift
from .synthetic import generate_synthetic_data
from .utils import (
    dataframe_hash,
    ensure_dir,
    generate_experiment_id,
    iso_now,
    set_random_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    target_column: str,
    timestamp_column: Optional[str],
    numerical_cols: List[str],
    categorical_cols: List[str],
    model_names: Optional[List[str]] = None,
    drift_type: str = "gradual",
    drift_start_window: int = 4,
    drift_duration: int = 4,
    drift_magnitude: float = 2.0,
    drift_affected_features: Optional[List[str]] = None,
    n_windows: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    forecast_method: str = "linear_trend",
    synthetic_n_samples: Optional[int] = None,
    synthetic_method: str = "perturbation",
    perturbation_scale: Optional[float] = None,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    run_recovery: bool = True,
    skip_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete DataDecay-X pipeline.

    Parameters
    ----------
    df                       : raw dataset (chronological)
    experiment_name          : human-readable name
    target_column            : name of target/label column
    timestamp_column         : name of timestamp column (or None)
    numerical_cols           : list of numerical feature columns
    categorical_cols         : list of categorical feature columns
    model_names              : list of model names to train (default: LR, SVM, NB)
    drift_type               : 'stable', 'gradual', 'abrupt', 'recurring'
    drift_start_window       : window where drift begins
    drift_duration           : number of windows with drift
    drift_magnitude          : drift intensity
    drift_affected_features  : features to drift (None = auto)
    n_windows                : number of temporal windows
    forecast_horizon         : windows to forecast ahead
    forecast_method          : 'persistence' or 'linear_trend'
    synthetic_n_samples      : number of synthetic rows to generate
    synthetic_method         : generation method
    perturbation_scale       : noise scale for perturbation method
    random_seed              : reproducibility seed
    progress_callback        : optional fn(step_name: str, pct: int) for UI progress
    run_recovery             : whether to run recovery strategies
    skip_synthetic           : if True, skip synthetic generation and quality gate

    Returns
    -------
    Complete experiment state dict (used directly by Streamlit UI)
    """
    cfg = get_config()

    # Apply defaults from config
    seed = random_seed if random_seed is not None else cfg.random_seed
    n_win = n_windows if n_windows is not None else cfg.n_windows
    horizon = forecast_horizon if forecast_horizon is not None else cfg.forecast_horizon
    n_synth = synthetic_n_samples if synthetic_n_samples is not None else cfg.default_synthetic_samples
    p_scale = perturbation_scale if perturbation_scale is not None else cfg.perturbation_scale
    models = model_names or ["logistic_regression", "svm", "naive_bayes"]
    threshold = cfg.reliability_threshold

    set_random_seed(seed)

    def _progress(step: str, pct: int) -> None:
        if progress_callback:
            progress_callback(step, pct)
        logger.info("[%d%%] %s", pct, step)

    experiment_id = generate_experiment_id("EXP")
    t_start = time.time()

    state: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "started_at": iso_now(),
        "config": {
            "target_column": target_column,
            "timestamp_column": timestamp_column,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "model_names": models,
            "drift_type": drift_type,
            "drift_start_window": drift_start_window,
            "drift_duration": drift_duration,
            "drift_magnitude": drift_magnitude,
            "drift_affected_features": drift_affected_features or [],
            "n_windows": n_win,
            "forecast_horizon": horizon,
            "forecast_method": forecast_method,
            "synthetic_n_samples": n_synth,
            "synthetic_method": synthetic_method,
            "perturbation_scale": p_scale,
            "random_seed": seed,
            "threshold": threshold,
        },
        "errors": [],
    }

    # -----------------------------------------------------------------------
    # STEP 1: Validate and summarize dataset
    # -----------------------------------------------------------------------
    _progress("Validating dataset", 5)
    validation = validate_dataset(
        df,
        target_column=target_column,
        timestamp_column=timestamp_column,
    )
    state["validation"] = validation

    if not validation["valid"]:
        state["errors"].append(f"Dataset validation failed: {validation['errors']}")
        return state

    ds_summary = dataset_summary(df, target_column, timestamp_column)
    ds_summary["dataset_name"] = experiment_name
    state["dataset_summary"] = ds_summary

    # -----------------------------------------------------------------------
    # STEP 2: Chronological preprocessing
    # -----------------------------------------------------------------------
    _progress("Chronological preprocessing", 10)

    df_sorted = sort_by_time(df, timestamp_column)

    split_result = chronological_split(
        df_sorted,
        n_windows=n_win,
        test_window_count=cfg.test_window_count,
        validation_window_count=cfg.validation_window_count,
    )
    state["split_info"] = split_result["split_info"]
    state["split_result"] = split_result

    ref_df = split_result["reference_df"]
    val_df = split_result["validation_df"]
    future_df = split_result["future_df"]
    full_df = split_result["full_df"]

    # Fit preprocessor on reference data only (no leakage)
    X_ref, y_ref = prepare_features(
        ref_df, target_column, numerical_cols, categorical_cols
    )
    preprocessor, X_ref_trans = fit_preprocessor(X_ref, numerical_cols, categorical_cols)
    y_ref_enc, label_encoder = encode_target(y_ref)

    state["preprocessor"] = preprocessor
    state["label_encoder"] = label_encoder

    # Prepare validation set
    X_val, y_val = prepare_features(val_df, target_column, numerical_cols, categorical_cols)
    X_val_trans = transform_with_preprocessor(X_val, preprocessor, numerical_cols, categorical_cols)
    y_val_enc, _ = encode_target(y_val, label_encoder)

    # -----------------------------------------------------------------------
    # STEP 3: Train baseline models
    # -----------------------------------------------------------------------
    _progress("Training baseline models", 20)

    model_results = train_multiple_models(
        model_names=models,
        X_train=X_ref_trans.values,
        y_train=y_ref_enc,
        X_val=X_val_trans.values,
        y_val=y_val_enc,
        random_seed=seed,
    )
    state["model_results"] = model_results
    state["model_summary_table"] = model_summary_table(model_results)

    # Select primary model (best by F1 on validation)
    best_model_name = max(
        model_results,
        key=lambda k: model_results[k]["metrics"].get("f1", 0.0)
        if model_results[k].get("model") else 0.0,
    )
    best_model = model_results[best_model_name]["model"]
    baseline_metrics = model_results[best_model_name]["metrics"]
    baseline_f1 = float(baseline_metrics.get("f1", 0.0))

    state["best_model_name"] = best_model_name
    state["best_model"] = best_model
    state["baseline_metrics"] = baseline_metrics
    state["baseline_f1"] = baseline_f1

    # -----------------------------------------------------------------------
    # STEP 4: Create temporal windows and simulate drift
    # -----------------------------------------------------------------------
    _progress("Simulating temporal drift", 30)

    drift_result = simulate_drift(
        df=full_df,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        n_windows=n_win,
        drift_type=drift_type,
        start_window=drift_start_window,
        duration=drift_duration,
        magnitude=drift_magnitude,
        affected_features=drift_affected_features,
        random_seed=seed,
        target_column=target_column,
        timestamp_column=timestamp_column,
    )

    drifted_df = drift_result["drifted_df"]
    state["drift_ground_truth"] = drift_result["ground_truth"]
    state["drift_window_metadata"] = drift_result["window_metadata"]
    state["drift_schedule"] = drift_result["drift_schedule"]

    # Create temporal windows on drifted data
    windows = create_temporal_windows(drifted_df, n_windows=n_win)
    state["windows"] = windows
    state["n_actual_windows"] = len(windows)

    # Reference window = first window (pre-drift)
    reference_window = windows[0] if windows else ref_df

    # -----------------------------------------------------------------------
    # STEP 5: Drift detection (PSI + KS + Wasserstein)
    # -----------------------------------------------------------------------
    _progress("Running drift detection", 40)

    detection_results = detect_drift_across_windows(
        reference_df=reference_window,
        windows=windows,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        psi_threshold_moderate=cfg.psi_threshold_moderate,
        psi_threshold_severe=cfg.psi_threshold_severe,
        ks_alpha=cfg.ks_alpha,
        wasserstein_threshold=cfg.wasserstein_threshold,
    )
    state["detection_results"] = detection_results
    state["window_drift_table"] = detection_results["window_drift_table"]
    state["feature_drift_by_win"] = detection_results["feature_drift_by_win"]
    state["overall_drift_severity"] = detection_results["overall_severity"]

    # -----------------------------------------------------------------------
    # STEP 6: Model reliability evaluation by window
    # -----------------------------------------------------------------------
    _progress("Evaluating model reliability by window", 50)

    window_metrics = evaluate_model_by_window(
        model=best_model,
        windows=windows,
        target_column=target_column,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
    )
    state["window_metrics"] = window_metrics

    degradation = calculate_reliability_degradation(
        window_metrics, baseline_f1
    )
    current_f1 = degradation["current"]
    health_score = calculate_health_score(current_f1, baseline_f1, threshold)
    time_below = calculate_time_below_threshold(window_metrics, threshold)

    state["degradation"] = degradation
    state["health_score"] = round(health_score, 4)
    state["time_below_threshold"] = time_below

    reliability_timeline = build_reliability_timeline(
        window_metrics,
        baseline_f1,
        threshold,
        drift_severity_table=detection_results["window_drift_table"],
    )
    state["reliability_timeline"] = reliability_timeline

    # -----------------------------------------------------------------------
    # STEP 7: Reliability forecasting and TTD
    # -----------------------------------------------------------------------
    _progress("Forecasting reliability", 60)

    if not window_metrics.empty:
        history = window_metrics["f1"].values
    else:
        history = np.array([baseline_f1])

    forecast_result = forecast_reliability(
        history=history,
        horizon=horizon,
        method=forecast_method,
        random_seed=seed,
    )

    ttd_result = estimate_threshold_crossing(
        history=history,
        forecast_values=forecast_result["forecast_values"],
        threshold=threshold,
        start_window_offset=len(history),
    )

    state["forecast_result"] = forecast_result
    state["ttd_result"] = ttd_result

    # -----------------------------------------------------------------------
    # STEP 8: Intervention decision
    # -----------------------------------------------------------------------
    _progress("Determining intervention state", 65)

    dsi = float(detection_results["overall_severity"].get("drift_severity_index", 0.0))
    trend = degradation.get("trend", "stable")

    intervention_result = determine_intervention(
        current_reliability=current_f1,
        baseline_reliability=baseline_f1,
        degradation_pct=degradation.get("degradation_pct", 0.0),
        drift_severity_index=dsi,
        forecast_trend=trend,
        threshold_crossing_estimate=ttd_result.get("ttd_estimate", ""),
        threshold=threshold,
        crossing_detected=ttd_result.get("crossing_detected", False),
        already_below_threshold=ttd_result.get("already_below", False),
        warning_lead_windows=ttd_result.get("warning_lead_windows", 999),
    )
    state["intervention_result"] = intervention_result

    # -----------------------------------------------------------------------
    # STEP 9: Synthetic data generation and quality gate
    # -----------------------------------------------------------------------
    if not skip_synthetic and run_recovery:
        _progress("Generating synthetic data", 70)

        # Use recent windows for synthetic generation
        n_recent = max(1, len(windows) // 4)
        recent_windows = windows[-n_recent:] if len(windows) >= n_recent else windows
        recent_df = pd.concat(recent_windows, ignore_index=True)

        try:
            synthetic_result = generate_synthetic_data(
                source_df=recent_df,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_column=target_column,
                n_samples=n_synth,
                method=synthetic_method,
                perturbation_scale=p_scale,
                random_seed=seed,
            )
            synthetic_df = synthetic_result["synthetic_df"]
            state["synthetic_result"] = synthetic_result
            state["synthetic_df"] = synthetic_df

        except Exception as exc:
            logger.error("Synthetic generation failed: %s", exc)
            state["errors"].append(f"Synthetic generation failed: {exc}")
            synthetic_df = pd.DataFrame()
            state["synthetic_result"] = None
            state["synthetic_df"] = synthetic_df

        # Quality gate
        _progress("Running quality gate", 75)

        if len(synthetic_df) > 0:
            gate_result = run_quality_gate(
                source_df=recent_df,
                synthetic_df=synthetic_df,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_column=target_column,
                minimum_score=cfg.quality_gate_minimum_score,
            )
            state["quality_gate_result"] = gate_result
        else:
            state["quality_gate_result"] = {
                "quality_score": 0.0,
                "passed": False,
                "decision": "REJECTED",
                "failed_checks": ["no_data"],
                "warned_checks": [],
                "passed_checks": [],
                "check_results": [],
                "detailed_report": "Synthetic data generation failed — no data to evaluate.",
            }

    else:
        synthetic_df = pd.DataFrame()
        state["synthetic_result"] = None
        state["synthetic_df"] = synthetic_df
        state["quality_gate_result"] = None

    # -----------------------------------------------------------------------
    # STEP 10: Recovery strategies
    # -----------------------------------------------------------------------
    if run_recovery:
        _progress("Running recovery strategies", 80)

        gate_passed = (
            state.get("quality_gate_result", {}) or {}
        ).get("passed", False)

        # Recent real data for recovery
        n_recent_real = max(1, len(windows) // 3)
        recent_real = pd.concat(
            windows[-n_recent_real:] if len(windows) >= n_recent_real else windows,
            ignore_index=True,
        )

        # No recovery baseline (evaluate original model on future)
        no_rec_result = run_no_recovery(
            baseline_model=best_model,
            future_df=future_df,
            target_column=target_column,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
        )

        # Real only
        real_result = run_real_only_recovery(
            baseline_model=best_model,
            recent_real_df=recent_real,
            future_df=future_df,
            target_column=target_column,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            model_name=best_model_name,
            random_seed=seed,
        )

        # Synthetic only (only if gate passed)
        if gate_passed and len(synthetic_df) > 0:
            syn_result = run_synthetic_only_recovery(
                baseline_model=best_model,
                synthetic_df=synthetic_df,
                future_df=future_df,
                target_column=target_column,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                model_name=best_model_name,
                random_seed=seed,
            )
            hybrid_result = run_hybrid_recovery(
                baseline_model=best_model,
                recent_real_df=recent_real,
                synthetic_df=synthetic_df,
                future_df=future_df,
                target_column=target_column,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                model_name=best_model_name,
                random_seed=seed,
            )
        else:
            # Gate failed: run with empty synthetic
            from .recovery import _empty_result, STRATEGY_SYNTHETIC, STRATEGY_HYBRID
            syn_result = _empty_result(
                STRATEGY_SYNTHETIC,
                error="Synthetic data rejected by quality gate." if not gate_passed else "No synthetic data.",
            )
            hybrid_result = _empty_result(
                STRATEGY_HYBRID,
                error="Synthetic data rejected by quality gate." if not gate_passed else "No synthetic data.",
            )

        strategy_results = {
            "no_recovery": no_rec_result,
            "real_only": real_result,
            "synthetic_only": syn_result,
            "hybrid": hybrid_result,
        }
        state["strategy_results"] = strategy_results

        # Baseline future F1 (no recovery)
        baseline_future_f1 = float(
            no_rec_result.get("future_metrics", {}).get("f1", baseline_f1)
        )
        state["baseline_future_f1"] = baseline_future_f1

        # Compare strategies
        comparison_df = compare_recovery_strategies(strategy_results, baseline_future_f1)
        state["recovery_comparison"] = comparison_df

        # -----------------------------------------------------------------------
        # STEP 11: Build evaluation summary
        # -----------------------------------------------------------------------
        _progress("Building evaluation summary", 90)

        eval_summary = build_evaluation_summary(
            baseline_train_metrics=baseline_metrics,
            baseline_future_metrics=no_rec_result.get("future_metrics", {}),
            strategy_results=strategy_results,
        )
        state["evaluation_summary"] = eval_summary

    else:
        state["strategy_results"] = {}
        state["recovery_comparison"] = pd.DataFrame()
        state["evaluation_summary"] = {}

    # -----------------------------------------------------------------------
    # STEP 12: Persist to database
    # -----------------------------------------------------------------------
    _progress("Saving to database", 92)

    try:
        db_path = cfg.database_path
        initialize_database(db_path)

        db_config = {
            **state["config"],
            "dataset_name": experiment_name,
            "dataset_hash": dataframe_hash(df),
            "dataset_rows": len(df),
            "dataset_cols": len(df.columns),
        }
        save_experiment(db_path, experiment_id, experiment_name, db_config)

        if not detection_results["window_drift_table"].empty:
            save_drift_results(
                db_path, experiment_id, detection_results["window_drift_table"]
            )

        if not reliability_timeline.empty:
            save_reliability_results(
                db_path, experiment_id, reliability_timeline, best_model_name
            )

        save_forecast_results(
            db_path, experiment_id, forecast_result, ttd_result, intervention_result
        )

        if run_recovery and not state["recovery_comparison"].empty:
            quality_score = (
                state.get("quality_gate_result", {}) or {}
            ).get("quality_score", 0.0)
            save_recovery_results(
                db_path, experiment_id, state["recovery_comparison"], quality_score
            )

            if state.get("evaluation_summary"):
                save_experiment_summary(
                    db_path,
                    experiment_id,
                    state["evaluation_summary"],
                    quality_score=quality_score,
                    quality_decision=(
                        state.get("quality_gate_result", {}) or {}
                    ).get("decision", ""),
                )

    except Exception as exc:
        logger.error("Database save failed: %s", exc)
        state["errors"].append(f"Database save failed: {exc}")

    # -----------------------------------------------------------------------
    # STEP 13: Generate report
    # -----------------------------------------------------------------------
    _progress("Generating report", 96)

    try:
        report = generate_experiment_report(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            config=state["config"],
            dataset_summary=state["dataset_summary"],
            split_info=state["split_info"],
            baseline_metrics=baseline_metrics,
            drift_results=detection_results,
            reliability_results=reliability_timeline,
            forecast_result=forecast_result,
            ttd_result=ttd_result,
            intervention_result=intervention_result,
            synthetic_metadata=state.get("synthetic_result", {}) and state["synthetic_result"].get("generation_metadata"),
            quality_gate_result=state.get("quality_gate_result"),
            recovery_comparison=state.get("recovery_comparison"),
            evaluation_summary=state.get("evaluation_summary"),
        )
        state["report"] = report

        # Save report files
        reports_dir = cfg.dir("reports")
        ensure_dir(reports_dir)
        save_report_json(report, reports_dir / f"{experiment_id}_report.json")
        save_report_csv(report, reports_dir, experiment_id)

    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        state["errors"].append(f"Report generation failed: {exc}")
        state["report"] = {}

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    state["completed_at"] = iso_now()
    state["total_time_s"] = round(elapsed, 2)
    state["status"] = "completed" if not state["errors"] else "completed_with_errors"

    _progress("Experiment complete", 100)

    logger.info(
        "Experiment '%s' completed in %.1fs — status=%s",
        experiment_name, elapsed, state["status"],
    )

    return state
