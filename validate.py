"""
validate.py — Quick validation script for DataDecay-X.
Run: python validate.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("DataDecay-X Validation")
print("=" * 60)
errors = []

# 1. Core imports
print("\n[1] Testing core imports...")
try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import joblib
    import yaml
    print(f"  numpy {np.__version__}  pandas {pd.__version__}  "
          f"sklearn {sklearn.__version__}  OK")
except Exception as e:
    errors.append(f"Core import failed: {e}")
    print(f"  FAIL: {e}")

# 2. DataDecay module imports
print("\n[2] Testing datadecay module imports...")
modules = [
    "datadecay.config", "datadecay.utils", "datadecay.data_loader",
    "datadecay.preprocessing", "datadecay.simulator", "datadecay.detection",
    "datadecay.models", "datadecay.monitoring", "datadecay.forecasting",
    "datadecay.intervention", "datadecay.synthetic", "datadecay.quality_gate",
    "datadecay.recovery", "datadecay.evaluation", "datadecay.experiment",
    "datadecay.database", "datadecay.reporting", "datadecay.demo_data",
]
for mod in modules:
    try:
        __import__(mod)
        print(f"  {mod}  OK")
    except Exception as e:
        errors.append(f"Import {mod}: {e}")
        print(f"  {mod}  FAIL: {e}")

# 3. Config loading
print("\n[3] Testing configuration...")
try:
    from datadecay.config import get_config, Config
    Config.reset()
    cfg = get_config()
    assert cfg.random_seed == 42
    assert 0 < cfg.reliability_threshold < 1
    assert cfg.n_windows >= 6
    print(f"  seed={cfg.random_seed}, threshold={cfg.reliability_threshold}, "
          f"n_windows={cfg.n_windows}  OK")
except Exception as e:
    errors.append(f"Config: {e}")
    print(f"  FAIL: {e}")

# 4. Demo data generation
print("\n[4] Testing demo data generation...")
try:
    from datadecay.demo_data import generate_demo_dataset, get_demo_feature_config
    df = generate_demo_dataset(n_rows=600, random_seed=42)
    assert len(df) == 600
    assert "default_risk" in df.columns  # demo target column name
    assert "timestamp" in df.columns
    cfg2 = get_demo_feature_config()
    assert len(cfg2["numerical_cols"]) > 0
    print(f"  Generated {len(df)} rows x {len(df.columns)} cols  OK")
except Exception as e:
    errors.append(f"Demo data: {repr(e)}")
    print(f"  FAIL: {repr(e)}")

# 5. Preprocessing pipeline
print("\n[5] Testing chronological preprocessing...")
try:
    from datadecay.preprocessing import (
        sort_by_time, chronological_split, prepare_features,
        fit_preprocessor, encode_target
    )
    from datadecay.demo_data import generate_demo_dataset, get_demo_feature_config
    df = generate_demo_dataset(n_rows=600)
    dcfg = get_demo_feature_config()
    df_sorted = sort_by_time(df, dcfg["timestamp_column"])
    split = chronological_split(df_sorted, n_windows=8, test_window_count=2)
    ref = split["reference_df"]
    X_ref, y_ref = prepare_features(
        ref, dcfg["target_column"], dcfg["numerical_cols"], dcfg["categorical_cols"]
    )
    prep, X_trans = fit_preprocessor(X_ref, dcfg["numerical_cols"], dcfg["categorical_cols"])
    y_enc, le = encode_target(y_ref)
    assert len(X_trans) == len(y_enc)
    print(f"  ref={len(ref)}, X={X_trans.shape}, classes={list(le.classes_)}  OK")
except Exception as e:
    errors.append(f"Preprocessing: {e}")
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# 6. Drift simulation
print("\n[6] Testing drift simulation...")
try:
    from datadecay.simulator import simulate_drift, create_temporal_windows
    from datadecay.demo_data import generate_demo_dataset, get_demo_feature_config
    from datadecay.preprocessing import sort_by_time, chronological_split
    df = generate_demo_dataset(n_rows=600)
    dcfg = get_demo_feature_config()
    df_sorted = sort_by_time(df, dcfg["timestamp_column"])
    split = chronological_split(df_sorted, n_windows=8, test_window_count=2)
    full_df = split["full_df"]
    result = simulate_drift(
        full_df, dcfg["numerical_cols"], dcfg["categorical_cols"],
        n_windows=8, drift_type="gradual", start_window=3,
        duration=3, magnitude=2.0, random_seed=42
    )
    windows = create_temporal_windows(result["drifted_df"], n_windows=8)
    assert len(windows) == 8
    assert result["ground_truth"]["drift_type"] == "gradual"
    print(f"  windows={len(windows)}, affected={result['ground_truth']['affected_features']}  OK")
except Exception as e:
    errors.append(f"Simulator: {e}")
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# 7. Drift detection
print("\n[7] Testing drift detection...")
try:
    from datadecay.detection import detect_drift_across_windows
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0,1,200), "b": rng.choice(["X","Y"],200)})
    shifted = pd.DataFrame({"a": rng.normal(3,1,100), "b": rng.choice(["X","Y"],100)})
    det = detect_drift_across_windows(
        ref, [ref.copy(), shifted], ["a"], ["b"]
    )
    dsi = det["window_drift_table"]["drift_severity_index"].iloc[-1]
    print(f"  DSI for shifted window={dsi:.3f}  OK")
except Exception as e:
    errors.append(f"Detection: {e}")
    print(f"  FAIL: {e}")

# 8. Model training
print("\n[8] Testing model training...")
try:
    from datadecay.models import create_model, train_model, evaluate_model, predict_model
    import numpy as np
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 4))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    for name in ["logistic_regression", "naive_bayes"]:
        clf = create_model(name, random_seed=42)
        clf, t = train_model(clf, X, y)
        preds = predict_model(clf, X)
        metrics = evaluate_model(y, preds)
        print(f"  {name}: F1={metrics['f1']:.3f} ({t:.2f}s)  OK")
except Exception as e:
    errors.append(f"Models: {e}")
    print(f"  FAIL: {e}")

# 9. Forecasting
print("\n[9] Testing forecasting...")
try:
    from datadecay.forecasting import forecast_reliability, estimate_threshold_crossing
    import numpy as np
    history = np.array([0.85, 0.82, 0.78, 0.74, 0.70])
    result = forecast_reliability(history, horizon=3, method="linear_trend")
    ttd = estimate_threshold_crossing(history, result["forecast_values"], threshold=0.70)
    print(f"  forecast={result['forecast_values'].round(3)}, "
          f"crossing={ttd['crossing_detected']}  OK")
except Exception as e:
    errors.append(f"Forecasting: {e}")
    print(f"  FAIL: {e}")

# 10. Synthetic generation + quality gate
print("\n[10] Testing synthetic generation + quality gate...")
try:
    from datadecay.synthetic import generate_synthetic_data
    from datadecay.quality_gate import run_quality_gate
    from datadecay.demo_data import generate_demo_dataset, get_demo_feature_config
    df = generate_demo_dataset(n_rows=400)
    dcfg = get_demo_feature_config()
    syn_result = generate_synthetic_data(
        df, dcfg["numerical_cols"], dcfg["categorical_cols"],
        dcfg["target_column"], n_samples=200, method="perturbation", random_seed=42
    )
    gate = run_quality_gate(
        df, syn_result["synthetic_df"],
        dcfg["numerical_cols"], dcfg["categorical_cols"],
        dcfg["target_column"], minimum_score=0.5
    )
    print(f"  synthetic={len(syn_result['synthetic_df'])}, "
          f"quality={gate['quality_score']:.2f}, "
          f"decision={gate['decision']}  OK")
except Exception as e:
    errors.append(f"Synthetic/Gate: {e}")
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# 11. Database
print("\n[11] Testing SQLite database...")
try:
    from datadecay.database import initialize_database, save_experiment, get_experiment_history
    from pathlib import Path
    test_db = Path("data/test_validate.db")
    initialize_database(test_db)
    save_experiment(test_db, "TESTEXP001", "Validate Test", {
        "dataset_name": "test", "dataset_hash": "abc123",
        "dataset_rows": 100, "dataset_cols": 5,
        "model_name": "lr", "drift_type": "gradual",
        "n_windows": 8, "random_seed": 42,
    })
    history = get_experiment_history(test_db)
    assert any(r["experiment_id"] == "TESTEXP001" for r in history)
    test_db.unlink(missing_ok=True)
    print(f"  DB init, save, load  OK")
except Exception as e:
    errors.append(f"Database: {e}")
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# 12. Intervention engine
print("\n[12] Testing intervention engine...")
try:
    from datadecay.intervention import determine_intervention, STATE_STABLE, STATE_RECOVERY_RECOMMENDED
    # Should be STABLE
    r1 = determine_intervention(0.82, 0.85, 3.5, 0.05, "stable", None, 0.70)
    # Should be RECOVERY_RECOMMENDED
    r2 = determine_intervention(0.60, 0.85, 30.0, 0.80, "declining",
                                "Crossing at window 2", 0.70,
                                crossing_detected=True, already_below_threshold=True,
                                warning_lead_windows=1)
    assert r1["state"] == STATE_STABLE
    assert r2["state"] == STATE_RECOVERY_RECOMMENDED
    print(f"  stable={r1['state']}, crisis={r2['state']}  OK")
except Exception as e:
    errors.append(f"Intervention: {e}")
    print(f"  FAIL: {e}")

# 13. Mini end-to-end pipeline
print("\n[13] Testing mini end-to-end pipeline...")
try:
    from datadecay.experiment import run_experiment
    from datadecay.demo_data import generate_demo_dataset, get_demo_feature_config
    df = generate_demo_dataset(n_rows=800)
    dcfg = get_demo_feature_config()
    t0 = time.time()
    state = run_experiment(
        df=df,
        experiment_name="Validation Mini Run",
        target_column=dcfg["target_column"],
        timestamp_column=dcfg["timestamp_column"],
        numerical_cols=dcfg["numerical_cols"],
        categorical_cols=dcfg["categorical_cols"],
        model_names=["logistic_regression", "naive_bayes"],
        drift_type="gradual",
        drift_start_window=3,
        drift_duration=3,
        drift_magnitude=2.0,
        n_windows=8,
        forecast_horizon=2,
        random_seed=42,
        synthetic_n_samples=200,
    )
    elapsed = time.time() - t0
    assert state["status"] in ("completed", "completed_with_errors")
    baseline_f1 = state.get("baseline_f1", 0)
    intervention = state.get("intervention_result", {}).get("state", "UNKNOWN")
    final_dec = state.get("evaluation_summary", {}).get("final_decision", "UNKNOWN")
    errors_list = state.get("errors", [])
    print(f"  status={state['status']}, baseline_F1={baseline_f1:.3f}, "
          f"intervention={intervention}, final={final_dec} ({elapsed:.1f}s)")
    if errors_list:
        print(f"  warnings: {errors_list}")
    print("  PIPELINE OK")
except Exception as e:
    errors.append(f"Pipeline: {e}")
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s):")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    sys.exit(1)
else:
    print("ALL VALIDATION CHECKS PASSED")
    print("Run command: python -m streamlit run app.py")
print("=" * 60)
