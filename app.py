"""
app.py — DataDecay-X Streamlit Application

Predictive Model Reliability Forecasting and Proactive Synthetic Remediation
under Temporal Distribution Shift

B.Tech 5th Semester Project
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DataDecay-X",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------
try:
    from src.datadecay.config import get_config
    from src.datadecay.data_loader import (
        dataset_summary,
        detect_target_column,
        detect_time_column,
        infer_feature_types,
        prepare_upload,
        validate_dataset,
    )
    from src.datadecay.database import (
        delete_experiment,
        get_experiment,
        get_experiment_history,
        initialize_database,
    )
    from src.datadecay.demo_data import (
        generate_demo_dataset,
        get_demo_feature_config,
    )
    from src.datadecay.experiment import run_experiment
    from src.datadecay.intervention import STATE_RECOVERY_RECOMMENDED, STATE_STABLE, STATE_WATCH
    from src.datadecay.reporting import generate_html_report_snippet, save_report_json
    from src.datadecay.utils import setup_logging, ensure_dir
    _BACKEND_OK = True
except Exception as _import_err:
    _BACKEND_OK = False
    _import_err_msg = str(_import_err)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if _BACKEND_OK:
    try:
        cfg = get_config()
        ensure_dir(cfg.log_file.parent)
        setup_logging(level=cfg.log_level, log_file=cfg.log_file)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #f1f5f9; margin: 4px 0; }
    .metric-sub   { font-size: 12px; color: #64748b; }

    /* Status badges */
    .badge-stable   { background:#16a34a; color:#fff; padding:4px 12px; border-radius:99px; font-size:13px; font-weight:600; }
    .badge-watch    { background:#d97706; color:#fff; padding:4px 12px; border-radius:99px; font-size:13px; font-weight:600; }
    .badge-recovery { background:#dc2626; color:#fff; padding:4px 12px; border-radius:99px; font-size:13px; font-weight:600; }
    .badge-neutral  { background:#475569; color:#fff; padding:4px 12px; border-radius:99px; font-size:13px; font-weight:600; }

    /* Section headers */
    .section-header {
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin: 16px 0 8px 0;
        font-weight: 600;
    }

    /* Info box */
    .info-box {
        background: #1e3a5f;
        border: 1px solid #2563eb;
        border-radius: 6px;
        padding: 12px;
        font-size: 13px;
        color: #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "page": "Overview",
        "experiment_state": None,
        "dataset": None,
        "dataset_metadata": None,
        "target_column": None,
        "timestamp_column": None,
        "numerical_cols": [],
        "categorical_cols": [],
        "experiment_name": "",
        "experiment_id": None,
        "settings": {
            "random_seed": 42,
            "n_windows": 12,
            "reliability_threshold": 0.70,
            "forecast_horizon": 3,
            "synthetic_samples": 500,
            "quality_gate_min": 0.70,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _pct(v):
    return f"{_safe_float(v)*100:.1f}%"


def _f4(v):
    return f"{_safe_float(v):.4f}"


def _badge(state: str) -> str:
    cls = {
        STATE_STABLE: "badge-stable",
        STATE_WATCH: "badge-watch",
        STATE_RECOVERY_RECOMMENDED: "badge-recovery",
    }.get(state, "badge-neutral")
    return f'<span class="{cls}">{state}</span>'


def _metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


def _no_experiment_warning():
    st.warning("No experiment loaded. Run the demo or create a new experiment first.")
    if st.button("Go to Overview"):
        st.session_state["page"] = "Overview"
        st.rerun()


def _get_exp() -> dict | None:
    return st.session_state.get("experiment_state")


def _plotly_defaults() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=30, t=50, b=40),
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## DATADECAY-X")
        st.markdown("---")

        pages = [
            ("Overview", "🏠"),
            ("New Experiment", "🧪"),
            ("Dataset", "📊"),
            ("Baseline Model", "🤖"),
            ("Drift Lab", "🌊"),
            ("Drift Monitor", "📡"),
            ("Model Reliability", "📉"),
            ("Reliability Forecast", "🔮"),
            ("Synthetic Lab", "🧬"),
            ("Quality Gate", "🛡️"),
            ("Recovery Center", "🔄"),
            ("Recovery Comparison", "⚖️"),
            ("Future Evaluation", "🎯"),
            ("Experiment History", "🗂️"),
            ("Reports", "📄"),
            ("Settings", "⚙️"),
        ]

        for name, icon in pages:
            if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state["page"] = name
                st.rerun()

        st.markdown("---")
        exp = _get_exp()
        if exp:
            st.markdown(f"**Active:** {exp.get('experiment_name', '—')[:20]}")
            st.markdown(f"**ID:** `{exp.get('experiment_id', '—')[:12]}...`")
            status = exp.get("status", "")
            if status == "completed":
                st.success("✓ Complete")
            elif "error" in status:
                st.warning("⚠ Errors")
            else:
                st.info("⏳ Running")
        else:
            st.markdown("*No active experiment*")


# ---------------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("DataDecay-X")
    st.markdown(
        "**Predictive Model Reliability Forecasting and Proactive Synthetic "
        "Remediation under Temporal Distribution Shift**"
    )
    st.markdown("---")

    exp = _get_exp()

    # Top action buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Run Demo Experiment", type="primary", use_container_width=True):
            _run_demo_experiment()
    with c2:
        if st.button("🧪 Create New Experiment", use_container_width=True):
            st.session_state["page"] = "New Experiment"
            st.rerun()
    with c3:
        if st.button("🗂️ View History", use_container_width=True):
            st.session_state["page"] = "Experiment History"
            st.rerun()

    st.markdown("---")

    if exp is None:
        st.markdown(
            '<div class="info-box">No experiment loaded. Click <b>Run Demo Experiment</b> '
            'to execute the full pipeline immediately, or create a new experiment.</div>',
            unsafe_allow_html=True,
        )
        return

    # Metric cards row
    irec = exp.get("intervention_result", {}) or {}
    state = irec.get("state", "STABLE")
    health = exp.get("health_score", 0.0)
    baseline_f1 = _safe_float(exp.get("baseline_f1", 0.0))
    rel_tl = exp.get("reliability_timeline")
    current_f1 = _safe_float(rel_tl["f1"].iloc[-1]) if rel_tl is not None and not rel_tl.empty else baseline_f1
    overall_drift = exp.get("overall_drift_severity", {}) or {}
    dsi = _safe_float(overall_drift.get("drift_severity_index", 0.0))
    ttd = exp.get("ttd_result", {}) or {}
    ttd_text = ttd.get("ttd_estimate", "N/A")
    eval_sum = exp.get("evaluation_summary", {}) or {}
    best_strat = eval_sum.get("best_strategy", "—")
    rec_comp = exp.get("recovery_comparison")

    # Health score color
    health_color = "#22c55e" if health >= 0.7 else "#f59e0b" if health >= 0.4 else "#ef4444"

    cols = st.columns(6)
    cards = [
        ("Model Health", f"{health*100:.0f}%", f"Threshold: {_safe_float(cfg.reliability_threshold if _BACKEND_OK else 0.7)*100:.0f}%"),
        ("Current F1", f"{current_f1:.3f}", f"Baseline: {baseline_f1:.3f}"),
        ("Drift Severity", f"{dsi:.3f}", overall_drift.get("severity_label", "—").upper()),
        ("Forecast State", state.replace("_", " "), irec.get("confidence", "").upper()),
        ("TTD Estimate", "Detected" if ttd.get("crossing_detected") else "None", "within horizon [ESTIMATE]"),
        ("Recovery Status", best_strat.replace("_", " ").title(), eval_sum.get("final_decision", "—")),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(_metric_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("---")

    # Reliability timeline
    if rel_tl is not None and not rel_tl.empty:
        st.markdown('<div class="section-header">Reliability Timeline</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rel_tl["window"], y=rel_tl["f1"],
            mode="lines+markers", name="F1 Score",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=6),
        ))
        fig.add_hline(
            y=_safe_float(exp.get("settings", {}).get("threshold", 0.70)),
            line_dash="dash", line_color="#ef4444",
            annotation_text="Threshold",
        )
        if "health_score" in rel_tl.columns:
            fig.add_trace(go.Scatter(
                x=rel_tl["window"], y=rel_tl["health_score"],
                mode="lines", name="Health Score",
                line=dict(color="#22c55e", width=1, dash="dot"),
            ))
        fig.update_layout(
            title="Model Reliability Over Time", xaxis_title="Window", yaxis_title="Score",
            **_plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

    # Drift severity
    wdt = exp.get("window_drift_table")
    if wdt is not None and not wdt.empty and "drift_severity_index" in wdt.columns:
        st.markdown('<div class="section-header">Drift Severity Timeline</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            wdt, x="window", y="drift_severity_index",
            color="severity_label" if "severity_label" in wdt.columns else "drift_severity_index",
            color_discrete_map={"none":"#22c55e","low":"#84cc16","moderate":"#f59e0b","high":"#f97316","severe":"#ef4444"},
            title="Drift Severity Index per Window",
        )
        fig2.update_layout(**_plotly_defaults())
        st.plotly_chart(fig2, use_container_width=True)

    # Intervention state
    st.markdown('<div class="section-header">Current System State</div>', unsafe_allow_html=True)
    st.markdown(_badge(state), unsafe_allow_html=True)
    st.markdown("")
    for r in irec.get("reasons", []):
        st.markdown(f"- {r}")
    if irec.get("recommended_action"):
        st.info(f"**Recommended Action:** {irec['recommended_action']}")

    # Experiment info
    with st.expander("Experiment Details"):
        c1, c2 = st.columns(2)
        c1.write(f"**ID:** {exp.get('experiment_id','')}")
        c1.write(f"**Name:** {exp.get('experiment_name','')}")
        c1.write(f"**Status:** {exp.get('status','')}")
        c1.write(f"**Total time:** {exp.get('total_time_s',0):.1f}s")
        c2.write(f"**Drift type:** {exp.get('config',{}).get('drift_type','')}")
        c2.write(f"**Model:** {exp.get('best_model_name','')}")
        c2.write(f"**Windows:** {exp.get('n_actual_windows','')}")
        c2.write(f"**Seed:** {exp.get('config',{}).get('random_seed','')}")
        if exp.get("errors"):
            st.warning(f"Errors: {exp['errors']}")


def _run_demo_experiment():
    """Execute the full pipeline with the demo dataset."""
    if not _BACKEND_OK:
        st.error(f"Backend import failed: {_import_err_msg}")
        return

    with st.spinner("Running demo experiment — this takes 20–60 seconds..."):
        progress = st.progress(0)
        status_text = st.empty()

        def _cb(step, pct):
            progress.progress(pct)
            status_text.text(f"[{pct}%] {step}")

        try:
            df = generate_demo_dataset()
            demo_cfg = get_demo_feature_config()

            state = run_experiment(
                df=df,
                experiment_name="Demo — Gradual Drift (Credit Risk)",
                target_column=demo_cfg["target_column"],
                timestamp_column=demo_cfg["timestamp_column"],
                numerical_cols=demo_cfg["numerical_cols"],
                categorical_cols=demo_cfg["categorical_cols"],
                model_names=["logistic_regression", "svm", "naive_bayes"],
                drift_type="gradual",
                drift_start_window=4,
                drift_duration=4,
                drift_magnitude=2.0,
                n_windows=st.session_state["settings"]["n_windows"],
                forecast_horizon=st.session_state["settings"]["forecast_horizon"],
                random_seed=st.session_state["settings"]["random_seed"],
                synthetic_n_samples=st.session_state["settings"]["synthetic_samples"],
                synthetic_method="perturbation",
                progress_callback=_cb,
            )

            st.session_state["experiment_state"] = state
            st.session_state["dataset"] = df
            st.session_state["dataset_metadata"] = state.get("dataset_summary", {})
            st.session_state["target_column"] = demo_cfg["target_column"]
            st.session_state["timestamp_column"] = demo_cfg["timestamp_column"]
            st.session_state["numerical_cols"] = demo_cfg["numerical_cols"]
            st.session_state["categorical_cols"] = demo_cfg["categorical_cols"]

            progress.progress(100)
            status_text.text("Complete!")
            st.success("Demo experiment complete! Explore the results using the sidebar.")

        except Exception as exc:
            st.error(f"Demo experiment failed: {exc}")
            with st.expander("Error traceback"):
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# PAGE 2: New Experiment
# ---------------------------------------------------------------------------

def page_new_experiment():
    st.title("🧪 New Experiment")
    st.markdown("Configure and launch a new experiment.")
    st.markdown("---")

    if not _BACKEND_OK:
        st.error(f"Backend unavailable: {_import_err_msg}")
        return

    # Step 1: Name
    st.subheader("Step 1 — Experiment Name")
    name = st.text_input("Experiment name", value="My Experiment", key="new_exp_name")

    st.markdown("---")

    # Step 2: Dataset
    st.subheader("Step 2 — Dataset")
    data_source = st.radio("Data source", ["Upload CSV", "Use Demo Dataset"], horizontal=True)

    df = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            try:
                df = prepare_upload(uploaded)
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    else:
        if st.button("Generate Demo Dataset"):
            df = generate_demo_dataset()
            st.session_state["_new_exp_df"] = df
            st.success(f"Demo dataset ready: {len(df):,} rows")
        df = st.session_state.get("_new_exp_df")
        if df is not None:
            st.info(f"Demo dataset loaded: {len(df):,} rows × {len(df.columns)} columns")

    if df is None:
        st.info("Load a dataset to continue.")
        return

    st.markdown("---")

    # Step 3: Column config
    st.subheader("Step 3 — Column Configuration")
    cols = list(df.columns)
    inferred = infer_feature_types(df)

    target_guess = detect_target_column(df)
    ts_guess = detect_time_column(df)

    c1, c2 = st.columns(2)
    target_col = c1.selectbox("Target column", cols,
                               index=cols.index(target_guess) if target_guess in cols else 0,
                               key="ne_target")
    ts_col = c2.selectbox("Timestamp column (optional)",
                           ["(none)"] + cols,
                           index=(cols.index(ts_guess) + 1) if ts_guess in cols else 0,
                           key="ne_ts")
    ts_col = None if ts_col == "(none)" else ts_col

    exclude = {target_col} | ({ts_col} if ts_col else set())
    ft = infer_feature_types(df, exclude_cols=list(exclude))

    num_cols = st.multiselect("Numerical features", options=ft["numerical"],
                               default=ft["numerical"], key="ne_num")
    cat_cols = st.multiselect("Categorical features", options=ft["categorical"],
                               default=ft["categorical"], key="ne_cat")

    st.markdown("---")

    # Step 4: Drift and model config
    st.subheader("Step 4 — Experiment Configuration")
    c1, c2, c3 = st.columns(3)
    drift_type = c1.selectbox("Drift type", ["gradual", "abrupt", "stable", "recurring"])
    n_windows = c2.slider("Temporal windows", 6, 20, st.session_state["settings"]["n_windows"])
    seed = c3.number_input("Random seed", value=st.session_state["settings"]["random_seed"], step=1)

    c1, c2, c3 = st.columns(3)
    drift_start = c1.slider("Drift start window", 1, n_windows - 2, min(4, n_windows - 2))
    drift_duration = c2.slider("Drift duration (windows)", 1, n_windows - drift_start,
                                min(4, n_windows - drift_start))
    magnitude = c3.slider("Drift magnitude", 0.5, 5.0, 2.0, step=0.5)

    model_choices = st.multiselect(
        "Models to train",
        options=["logistic_regression", "svm", "naive_bayes", "random_forest"],
        default=["logistic_regression", "svm", "naive_bayes"],
        key="ne_models",
    )

    st.markdown("---")

    # Validate
    if st.button("Validate Dataset"):
        val = validate_dataset(df, target_column=target_col, timestamp_column=ts_col)
        if val["valid"]:
            st.success("Dataset validation passed.")
            for w in val.get("warnings", []):
                st.warning(w)
            for i in val.get("info", []):
                st.info(i)
        else:
            for e in val["errors"]:
                st.error(e)

    st.markdown("---")
    if st.button("🚀 Launch Experiment", type="primary"):
        if not model_choices:
            st.error("Select at least one model.")
            return
        if not num_cols and not cat_cols:
            st.error("Select at least one feature column.")
            return

        with st.spinner("Running experiment..."):
            progress = st.progress(0)
            status = st.empty()

            def _cb(step, pct):
                progress.progress(pct)
                status.text(f"[{pct}%] {step}")

            try:
                state = run_experiment(
                    df=df,
                    experiment_name=name,
                    target_column=target_col,
                    timestamp_column=ts_col,
                    numerical_cols=num_cols,
                    categorical_cols=cat_cols,
                    model_names=model_choices,
                    drift_type=drift_type,
                    drift_start_window=drift_start,
                    drift_duration=drift_duration,
                    drift_magnitude=magnitude,
                    n_windows=n_windows,
                    forecast_horizon=st.session_state["settings"]["forecast_horizon"],
                    random_seed=int(seed),
                    synthetic_n_samples=st.session_state["settings"]["synthetic_samples"],
                    progress_callback=_cb,
                )
                st.session_state["experiment_state"] = state
                st.session_state["dataset"] = df
                st.session_state["target_column"] = target_col
                st.session_state["timestamp_column"] = ts_col
                st.session_state["numerical_cols"] = num_cols
                st.session_state["categorical_cols"] = cat_cols
                st.session_state["experiment_name"] = name

                progress.progress(100)
                status.text("Done!")
                st.success(f"Experiment '{name}' complete! Explore results in the sidebar.")

                if state.get("errors"):
                    st.warning(f"Completed with warnings: {state['errors']}")

            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# PAGE 3: Dataset
# ---------------------------------------------------------------------------

def page_dataset():
    st.title("📊 Dataset")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    ds = exp.get("dataset_summary", {}) or {}
    df: pd.DataFrame = st.session_state.get("dataset")

    if df is None:
        st.error("Dataset not in session. Please reload the experiment.")
        return

    tab1, tab2, tab3 = st.tabs(["Overview", "Preview", "Distributions"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{ds.get('n_rows', len(df)):,}")
        c2.metric("Columns", ds.get("n_columns", len(df.columns)))
        c3.metric("Numerical Features", ds.get("n_numerical_features", 0))
        c4.metric("Categorical Features", ds.get("n_categorical_features", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing Cells", f"{ds.get('n_missing_cells',0):,}")
        c2.metric("Missing %", f"{ds.get('pct_missing',0):.1f}%")
        c3.metric("Duplicate Rows", ds.get("n_duplicate_rows", 0))
        c4.metric("Dataset Hash", ds.get("dataset_hash", "—")[:8])

        st.markdown("---")
        st.markdown("**Feature columns**")
        col_type_df = pd.DataFrame([
            {"column": k, "type": v}
            for k, v in (ds.get("column_types") or {}).items()
        ])
        if not col_type_df.empty:
            st.dataframe(col_type_df, use_container_width=True, height=250)

        tr = ds.get("time_range", {})
        if tr:
            st.markdown(f"**Time range:** {tr.get('min','—')} → {tr.get('max','—')} ({tr.get('range_days',0)} days)")

        td = ds.get("target_distribution", {})
        if td:
            st.markdown(f"**Target classes:** {td.get('n_classes')} | Distribution: {td.get('class_balance',{})}")

    with tab2:
        st.markdown("**Dataset Preview (first 100 rows)**")
        st.dataframe(df.head(100), use_container_width=True)

    with tab3:
        num_cols = st.session_state.get("numerical_cols", [])
        if num_cols:
            sel = st.selectbox("Select feature to visualise", num_cols)
            fig = px.histogram(
                df, x=sel, nbins=40,
                title=f"Distribution of {sel}",
                color_discrete_sequence=["#3b82f6"],
            )
            fig.update_layout(**_plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)

        split_info = exp.get("split_info", {})
        if split_info:
            st.markdown("**Chronological Split**")
            st.json(split_info)


# ---------------------------------------------------------------------------
# PAGE 4: Baseline Model
# ---------------------------------------------------------------------------

def page_baseline_model():
    st.title("🤖 Baseline Model")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    # Model comparison table
    tbl = exp.get("model_summary_table")
    if tbl is not None and not tbl.empty:
        st.markdown('<div class="section-header">Model Comparison (Validation Set)</div>',
                    unsafe_allow_html=True)
        best_name = exp.get("best_model_name", "")

        styled = tbl.copy()
        styled.columns = [c.upper() for c in styled.columns]
        st.dataframe(
            tbl.style.highlight_max(subset=["f1", "accuracy"], color="#1e3a5f"),
            use_container_width=True,
        )
        st.success(f"Best baseline model: **{best_name}** (highest F1)")

    # Metrics detail for best model
    bm = exp.get("baseline_metrics", {}) or {}
    if bm:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{_safe_float(bm.get('accuracy')):.4f}")
        c2.metric("Precision", f"{_safe_float(bm.get('precision')):.4f}")
        c3.metric("Recall", f"{_safe_float(bm.get('recall')):.4f}")
        c4.metric("F1 Score", f"{_safe_float(bm.get('f1')):.4f}")

        # Confusion matrix
        cm = bm.get("confusion_matrix", [])
        if cm:
            st.markdown("**Confusion Matrix (Validation)**")
            cm_df = pd.DataFrame(cm)
            fig = px.imshow(
                cm_df, text_auto=True,
                title="Confusion Matrix",
                color_continuous_scale="Blues",
            )
            fig.update_layout(**_plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)

    # Split info
    split_info = exp.get("split_info", {})
    if split_info:
        st.markdown("---")
        st.markdown('<div class="section-header">Chronological Split</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reference Rows", f"{split_info.get('reference_rows',0):,}",
                  f"{split_info.get('train_pct',0):.1f}% of data")
        c2.metric("Validation Rows", f"{split_info.get('validation_rows',0):,}",
                  f"{split_info.get('validation_pct',0):.1f}%")
        c3.metric("Future Test Rows", f"{split_info.get('future_rows',0):,}",
                  f"{split_info.get('future_pct',0):.1f}%")

    st.info(
        "**Note:** Preprocessing was fitted exclusively on the reference (training) data. "
        "No information from validation or future windows was used."
    )


# ---------------------------------------------------------------------------
# PAGE 5: Drift Lab
# ---------------------------------------------------------------------------

def page_drift_lab():
    st.title("🌊 Temporal Drift Lab")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    gt = exp.get("drift_ground_truth", {}) or {}
    schedule = exp.get("drift_schedule", [])
    window_meta = exp.get("drift_window_metadata", []) or []

    st.markdown('<div class="section-header">Ground Truth Drift Configuration</div>',
                unsafe_allow_html=True)

    if gt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drift Type", gt.get("drift_type", "—").upper())
        c2.metric("Start Window", gt.get("start_window", "—"))
        c3.metric("End Window", gt.get("end_window", "—"))
        c4.metric("Magnitude", gt.get("magnitude", "—"))

        st.markdown(f"**Affected features:** {', '.join(gt.get('affected_features', []))}")
        st.info(gt.get("note", ""))

    # Drift schedule chart
    if schedule:
        st.markdown('<div class="section-header">Drift Intensity Schedule</div>',
                    unsafe_allow_html=True)
        sched_df = pd.DataFrame({
            "window": list(range(len(schedule))),
            "intensity": schedule,
        })
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in schedule]
        fig = go.Figure(go.Bar(
            x=sched_df["window"], y=sched_df["intensity"],
            marker_color=colors, name="Drift Intensity",
        ))
        if gt.get("start_window") is not None:
            fig.add_vrect(
                x0=gt["start_window"] - 0.5,
                x1=gt.get("end_window", gt["start_window"]) + 0.5,
                fillcolor="rgba(239,68,68,0.1)",
                annotation_text="Drift Zone",
                line_width=0,
            )
        fig.update_layout(
            title="Ground Truth Drift Schedule",
            xaxis_title="Window", yaxis_title="Drift Intensity",
            **_plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature distribution comparison
    df = st.session_state.get("dataset")
    if df is not None and "window_metadata" not in str(exp.keys()):
        pass

    windows = exp.get("windows")
    if windows and df is not None:
        num_cols = st.session_state.get("numerical_cols", [])
        affected = gt.get("affected_numerical", [])
        if affected and num_cols:
            st.markdown('<div class="section-header">Feature Distribution: Before vs During Drift</div>',
                        unsafe_allow_html=True)
            feat = st.selectbox("Select affected feature", affected)
            pre_win_idx = max(0, gt.get("start_window", 4) - 2)
            drift_win_idx = min(gt.get("start_window", 4) + 1, len(windows) - 1)

            pre_data = windows[pre_win_idx][feat].dropna() if feat in windows[pre_win_idx].columns else pd.Series()
            drift_data = windows[drift_win_idx][feat].dropna() if feat in windows[drift_win_idx].columns else pd.Series()

            if len(pre_data) > 0 and len(drift_data) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pre_data, name=f"Pre-drift (W{pre_win_idx})",
                                           opacity=0.7, marker_color="#3b82f6"))
                fig.add_trace(go.Histogram(x=drift_data, name=f"During drift (W{drift_win_idx})",
                                           opacity=0.7, marker_color="#ef4444"))
                fig.update_layout(barmode="overlay", title=f"{feat} Distribution",
                                  **_plotly_defaults())
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 6: Drift Monitor
# ---------------------------------------------------------------------------

def page_drift_monitor():
    st.title("📡 Drift Monitor")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    st.info(
        "**Note:** Drift detection and model reliability are separate signals. "
        "Drift does not automatically mean model failure."
    )

    overall = exp.get("overall_drift_severity", {}) or {}
    if overall:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drift Severity Index", f"{_safe_float(overall.get('drift_severity_index')):.3f}")
        c2.metric("Severity Label", overall.get("severity_label", "—").upper())
        c3.metric("Max PSI", f"{_safe_float(overall.get('max_psi')):.4f}")
        c4.metric("Drifted Features", overall.get("n_drifted_features", 0))

    tab1, tab2, tab3 = st.tabs(["Window Drift", "Feature Drift Heatmap", "Feature Detail"])

    with tab1:
        wdt = exp.get("window_drift_table")
        if wdt is not None and not wdt.empty:
            # Drift severity over windows
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wdt["window"], y=wdt["drift_severity_index"],
                mode="lines+markers", name="DSI",
                line=dict(color="#f97316", width=2),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",
            ))
            fig.add_hline(y=0.20, line_dash="dash", line_color="#ef4444",
                          annotation_text="Severe threshold")
            fig.add_hline(y=0.10, line_dash="dot", line_color="#f59e0b",
                          annotation_text="Moderate threshold")
            fig.update_layout(
                title="Drift Severity Index Over Time",
                xaxis_title="Window", yaxis_title="DSI",
                **_plotly_defaults()
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Window-level drift table**")
            display_cols = [c for c in ["window", "drift_severity_index", "severity_label",
                                         "max_psi", "mean_psi", "max_ks", "n_drifted_features"]
                            if c in wdt.columns]
            st.dataframe(wdt[display_cols], use_container_width=True)

    with tab2:
        fdbw = exp.get("feature_drift_by_win", {}) or {}
        if fdbw:
            # Build heatmap
            windows_idx = sorted(fdbw.keys())
            all_features = None
            for w in windows_idx:
                fd = fdbw[w]
                if fd is not None and not fd.empty:
                    all_features = list(fd["feature"].values)
                    break

            if all_features:
                psi_matrix = []
                for w in windows_idx:
                    fd = fdbw[w]
                    if fd is not None and not fd.empty:
                        row = fd.set_index("feature")["psi"].reindex(all_features, fill_value=0).tolist()
                    else:
                        row = [0.0] * len(all_features)
                    psi_matrix.append(row)

                fig = px.imshow(
                    psi_matrix,
                    x=all_features,
                    y=[f"W{w}" for w in windows_idx],
                    color_continuous_scale="RdYlGn_r",
                    title="PSI Heatmap (Features × Windows)",
                    zmin=0, zmax=0.30,
                )
                fig.update_layout(**_plotly_defaults())
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fdbw = exp.get("feature_drift_by_win", {}) or {}
        if fdbw:
            last_win = max(fdbw.keys()) if fdbw else None
            if last_win is not None:
                fd = fdbw[last_win]
                if fd is not None and not fd.empty:
                    st.markdown(f"**Feature drift at last window (W{last_win})**")
                    display = [c for c in ["feature", "type", "psi", "ks_statistic",
                                            "ks_p_value", "ks_drift", "wasserstein", "severity"]
                               if c in fd.columns]
                    st.dataframe(fd[display].sort_values("psi", ascending=False),
                                 use_container_width=True)

                    # Top drifting features bar
                    top_psi = fd.nlargest(min(8, len(fd)), "psi")
                    fig = px.bar(
                        top_psi, x="feature", y="psi",
                        color="severity",
                        color_discrete_map={"none":"#22c55e","low":"#84cc16",
                                            "moderate":"#f59e0b","severe":"#ef4444"},
                        title="Top Drifting Features (PSI)",
                    )
                    fig.update_layout(**_plotly_defaults())
                    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 7: Model Reliability
# ---------------------------------------------------------------------------

def page_model_reliability():
    st.title("📉 Model Reliability")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    st.info(
        "Drift and model failure are separate signals. "
        "This panel shows the actual measured predictive impact on each temporal window."
    )

    baseline_f1 = _safe_float(exp.get("baseline_f1", 0.0))
    deg = exp.get("degradation", {}) or {}
    health = _safe_float(exp.get("health_score", 0.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline F1", f"{baseline_f1:.4f}")
    c2.metric("Current F1", f"{_safe_float(deg.get('current')):.4f}",
              delta=f"{_safe_float(deg.get('current'))-baseline_f1:+.4f}")
    c3.metric("Minimum F1", f"{_safe_float(deg.get('minimum')):.4f}")
    c4.metric("Degradation", f"{_safe_float(deg.get('degradation_pct')):.1f}%")
    c5.metric("Health Score", f"{health*100:.0f}%")

    rel_tl = exp.get("reliability_timeline")
    if rel_tl is not None and not rel_tl.empty:
        # F1 timeline
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rel_tl["window"], y=rel_tl["f1"],
            mode="lines+markers", name="F1",
            line=dict(color="#3b82f6", width=2),
        ))
        if "accuracy" in rel_tl.columns:
            fig.add_trace(go.Scatter(
                x=rel_tl["window"], y=rel_tl["accuracy"],
                mode="lines", name="Accuracy",
                line=dict(color="#a78bfa", width=1, dash="dot"),
            ))
        if "precision" in rel_tl.columns:
            fig.add_trace(go.Scatter(
                x=rel_tl["window"], y=rel_tl["precision"],
                mode="lines", name="Precision",
                line=dict(color="#34d399", width=1, dash="dash"),
            ))
        threshold = _safe_float(exp.get("config", {}).get("threshold", 0.70))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Threshold ({threshold})")
        fig.add_hline(y=baseline_f1, line_dash="dot", line_color="#94a3b8",
                      annotation_text="Baseline F1")
        fig.update_layout(
            title="Model Metrics Over Time",
            xaxis_title="Window", yaxis_title="Score",
            **_plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Below threshold indicator
        tbt = exp.get("time_below_threshold", {}) or {}
        if tbt:
            c1, c2, c3 = st.columns(3)
            c1.metric("Windows Below Threshold", tbt.get("n_windows_below", 0))
            c2.metric("Max Consecutive Below", tbt.get("consecutive_below", 0))
            c3.metric("First Crossing Window", tbt.get("first_crossing_window", "None"))


# ---------------------------------------------------------------------------
# PAGE 8: Reliability Forecast
# ---------------------------------------------------------------------------

def page_reliability_forecast():
    st.title("🔮 Reliability Forecast")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    forecast = exp.get("forecast_result", {}) or {}
    ttd = exp.get("ttd_result", {}) or {}
    irec = exp.get("intervention_result", {}) or {}

    if not forecast:
        st.warning("No forecast results available.")
        return

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Method", forecast.get("method", "—").replace("_", " ").title())
    c2.metric("Forecast Horizon", f"{forecast.get('horizon', 0)} windows")
    c3.metric("Trend", (exp.get("degradation", {}) or {}).get("trend", "—").upper())
    c4.metric("Current Reliability", f"{_safe_float(ttd.get('current_reliability')):.4f}")

    # TTD estimate
    st.markdown("---")
    st.markdown("**Time-to-Decay Estimate (ESTIMATE ONLY — not a guaranteed future event)**")
    if ttd.get("already_below"):
        st.error(f"Already below threshold: {ttd.get('ttd_estimate', '')}")
    elif ttd.get("crossing_detected"):
        st.warning(f"{ttd.get('ttd_estimate', '')}")
    else:
        st.success(f"{ttd.get('ttd_estimate', '')}")

    st.markdown("---")

    # Forecast chart
    history_vals = forecast.get("history_values", np.array([]))
    history_wins = forecast.get("history_windows", np.array([]))
    forecast_vals = forecast.get("forecast_values", np.array([]))
    forecast_wins = forecast.get("forecast_windows", np.array([]))

    fig = go.Figure()

    if len(history_vals) > 0:
        fig.add_trace(go.Scatter(
            x=history_wins, y=history_vals,
            mode="lines+markers", name="Observed Reliability",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=6),
        ))

    if len(forecast_vals) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_wins, y=forecast_vals,
            mode="lines+markers", name="Forecast (ESTIMATE)",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        ))

    threshold = _safe_float(exp.get("config", {}).get("threshold", 0.70))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Threshold ({threshold})")

    if ttd.get("crossing_window") is not None:
        fig.add_vline(
            x=ttd["crossing_window"],
            line_dash="dot", line_color="#ef4444",
            annotation_text="Est. Crossing [ESTIMATE]",
        )

    fig.update_layout(
        title="Reliability Forecast",
        xaxis_title="Window", yaxis_title="F1 Score",
        **_plotly_defaults()
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast metadata
    trend_meta = forecast.get("trend_metadata", {})
    if trend_meta:
        with st.expander("Forecast Details"):
            st.json(trend_meta)

    # Intervention state
    st.markdown("---")
    st.markdown("**Intervention State**")
    state = irec.get("state", "STABLE")
    st.markdown(_badge(state), unsafe_allow_html=True)
    for r in irec.get("reasons", []):
        st.markdown(f"- {r}")


# ---------------------------------------------------------------------------
# PAGE 9: Synthetic Lab
# ---------------------------------------------------------------------------

def page_synthetic_lab():
    st.title("🧬 Synthetic Lab")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    syn_result = exp.get("synthetic_result")

    if syn_result is None:
        st.warning("No synthetic data in current experiment.")
        st.info("Run an experiment with recovery enabled to generate synthetic data.")
        return

    meta = syn_result.get("generation_metadata", {}) or {}
    syn_df = exp.get("synthetic_df")

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method", meta.get("method", "—").replace("_", " ").title())
    c2.metric("Samples Generated", f"{meta.get('n_samples_generated', 0):,}")
    c3.metric("Source Rows", f"{meta.get('source_rows', 0):,}")
    c4.metric("Generation Time", f"{meta.get('generation_time_s', 0):.2f}s")

    st.markdown(f"**Note:** {meta.get('note', '')}")

    # Distribution comparison
    src_stats = syn_result.get("source_stats", {})
    syn_stats = syn_result.get("synthetic_stats", {})
    num_cols = meta.get("numerical_cols", [])

    if num_cols and src_stats and syn_stats:
        st.markdown("---")
        st.markdown('<div class="section-header">Source vs Synthetic Distribution</div>',
                    unsafe_allow_html=True)

        sel_feat = st.selectbox("Select numerical feature", num_cols, key="syn_feat")

        if sel_feat in src_stats and sel_feat in syn_stats:
            src_s = src_stats[sel_feat]
            syn_s = syn_stats[sel_feat]

            cmp_data = pd.DataFrame({
                "Metric": ["Mean", "Std", "Min", "Max", "Median"],
                "Source": [src_s.get("mean"), src_s.get("std"), src_s.get("min"),
                           src_s.get("max"), src_s.get("median")],
                "Synthetic": [syn_s.get("mean"), syn_s.get("std"), syn_s.get("min"),
                              syn_s.get("max"), syn_s.get("median")],
            })
            st.dataframe(cmp_data, use_container_width=True)

    # Class balance
    target_col = meta.get("target_column", "")
    if syn_df is not None and target_col and target_col in syn_df.columns:
        st.markdown("---")
        st.markdown('<div class="section-header">Target Class Balance</div>',
                    unsafe_allow_html=True)
        vc = syn_df[target_col].value_counts().reset_index()
        vc.columns = ["class", "count"]
        fig = px.bar(vc, x="class", y="count",
                     title="Synthetic Target Distribution",
                     color="class", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**_plotly_defaults())
        st.plotly_chart(fig, use_container_width=True)

    # Preview
    if syn_df is not None:
        with st.expander("Synthetic Data Preview (first 50 rows)"):
            st.dataframe(syn_df.head(50), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 10: Quality Gate
# ---------------------------------------------------------------------------

def page_quality_gate():
    st.title("🛡️ Quality Gate")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    qg = exp.get("quality_gate_result")
    if qg is None:
        st.warning("No quality gate results found.")
        return

    # Decision banner
    decision = qg.get("decision", "UNKNOWN")
    score = _safe_float(qg.get("quality_score", 0.0))

    if decision == "ACCEPTED":
        st.success(f"✅ ACCEPTED — Quality Score: {score*100:.0f}%")
    else:
        st.error(f"❌ REJECTED — Quality Score: {score*100:.0f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality Score", f"{score*100:.0f}%")
    c2.metric("Passed Checks", len(qg.get("passed_checks", [])))
    c3.metric("Warned Checks", len(qg.get("warned_checks", [])))
    c4.metric("Failed Checks", len(qg.get("failed_checks", [])))

    st.markdown("---")
    st.markdown('<div class="section-header">Check Results</div>', unsafe_allow_html=True)

    check_results = qg.get("check_results", [])
    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}

    for r in check_results:
        icon = status_icon.get(r["status"], "❓")
        check_name = r["check"].replace("_", " ").title()
        detail = r["detail"]
        if r["status"] == "PASS":
            st.markdown(f"{icon} **{check_name}**: {detail}")
        elif r["status"] == "WARN":
            st.warning(f"{icon} **{check_name}**: {detail}")
        else:
            st.error(f"{icon} **{check_name}**: {detail}")

    # Score gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": "Quality Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3b82f6"},
            "steps": [
                {"range": [0, 50], "color": "#ef4444"},
                {"range": [50, 70], "color": "#f59e0b"},
                {"range": [70, 100], "color": "#22c55e"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": _safe_float(qg.get("minimum_score", 0.70)) * 100,
            },
        },
    ))
    fig.update_layout(height=300, **_plotly_defaults())
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full Quality Gate Report"):
        st.text(qg.get("detailed_report", "No report available."))


# ---------------------------------------------------------------------------
# PAGE 11: Recovery Center
# ---------------------------------------------------------------------------

def page_recovery_center():
    st.title("🔄 Recovery Center")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    irec = exp.get("intervention_result", {}) or {}
    state = irec.get("state", STATE_STABLE)
    qg = exp.get("quality_gate_result", {}) or {}

    # Current status
    st.markdown("**Current System Status**")
    st.markdown(_badge(state), unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline F1", f"{_safe_float(exp.get('baseline_f1')):.4f}")
    c2.metric("Current F1", f"{_safe_float((exp.get('degradation') or {}).get('current')):.4f}")
    c3.metric("Drift Severity", f"{_safe_float((exp.get('overall_drift_severity') or {}).get('drift_severity_index')):.3f}")

    st.markdown("---")
    st.markdown("**Recommended Action**")
    st.info(irec.get("recommended_action", "—"))

    # Quality gate status
    st.markdown("---")
    st.markdown('<div class="section-header">Synthetic Data Quality Gate Status</div>',
                unsafe_allow_html=True)

    if not qg:
        st.warning("No quality gate results.")
    elif qg.get("decision") == "ACCEPTED":
        st.success(f"✅ Synthetic data ACCEPTED — Quality Score: {_safe_float(qg.get('quality_score'))*100:.0f}%")
    else:
        st.error(f"❌ Synthetic data REJECTED — Score: {_safe_float(qg.get('quality_score'))*100:.0f}%")
        if qg.get("failed_checks"):
            st.error(f"Failed: {', '.join(qg['failed_checks'])}")
        st.warning("Rejected synthetic data will not be used for recovery strategies.")

    # Strategy summary
    st.markdown("---")
    st.markdown('<div class="section-header">Recovery Strategy Results</div>',
                unsafe_allow_html=True)

    rec_comp = exp.get("recovery_comparison")
    if rec_comp is not None and not rec_comp.empty:
        display_cols = [c for c in ["strategy", "future_f1", "recovery_gain",
                                     "remaining_degradation", "n_real", "n_synthetic",
                                     "training_time_s", "decision"]
                        if c in rec_comp.columns]
        st.dataframe(rec_comp[display_cols], use_container_width=True)
    else:
        st.info("Run an experiment with recovery enabled to see strategy results.")


# ---------------------------------------------------------------------------
# PAGE 12: Recovery Comparison
# ---------------------------------------------------------------------------

def page_recovery_comparison():
    st.title("⚖️ Recovery Comparison")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    rec_comp = exp.get("recovery_comparison")
    if rec_comp is None or rec_comp.empty:
        st.warning("No recovery comparison available.")
        return

    baseline_future = _safe_float(exp.get("baseline_future_f1", 0.0))

    # Summary table
    st.markdown('<div class="section-header">Strategy Comparison (Independent Future Data)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**All decisions are based on performance on the held-out future data — "
        "not training performance.**"
    )

    decision_colors = {
        "ACCEPT": "✅", "NEUTRAL": "➡️", "REJECT": "❌",
        "BASELINE": "📌", "ERROR": "⚠️",
    }
    display_df = rec_comp.copy()
    if "decision" in display_df.columns:
        display_df[""] = display_df["decision"].map(decision_colors)
    st.dataframe(display_df, use_container_width=True)

    # F1 comparison bar chart
    if "future_f1" in rec_comp.columns:
        fig = go.Figure()
        colors = []
        for i, row in rec_comp.iterrows():
            d = row.get("decision", "")
            colors.append(
                "#22c55e" if d == "ACCEPT" else
                "#94a3b8" if d == "BASELINE" else
                "#f59e0b" if d == "NEUTRAL" else
                "#ef4444"
            )
        fig.add_trace(go.Bar(
            x=rec_comp["strategy"],
            y=rec_comp["future_f1"],
            marker_color=colors,
            name="Future F1",
        ))
        fig.add_hline(y=baseline_future, line_dash="dash", line_color="#94a3b8",
                      annotation_text="Baseline Future F1")
        fig.update_layout(
            title="Future F1 by Recovery Strategy",
            xaxis_title="Strategy", yaxis_title="Future F1",
            **_plotly_defaults()
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recovery gain chart
    if "recovery_gain" in rec_comp.columns:
        non_base = rec_comp[rec_comp["strategy"] != "no_recovery"]
        if not non_base.empty:
            colors_gain = ["#22c55e" if v > 0.005 else "#f59e0b" if v > -0.005 else "#ef4444"
                           for v in non_base["recovery_gain"]]
            fig2 = go.Figure(go.Bar(
                x=non_base["strategy"],
                y=non_base["recovery_gain"],
                marker_color=colors_gain,
                name="Recovery Gain",
            ))
            fig2.add_hline(y=0, line_color="#94a3b8", line_width=1)
            fig2.update_layout(
                title="Recovery Gain vs No-Recovery Baseline",
                xaxis_title="Strategy", yaxis_title="F1 Gain",
                **_plotly_defaults()
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Best strategy highlight
    eval_sum = exp.get("evaluation_summary", {}) or {}
    best = eval_sum.get("best_strategy", "")
    final_dec = eval_sum.get("final_decision", "")
    final_expl = eval_sum.get("final_explanation", "")
    if final_dec:
        st.markdown("---")
        if final_dec == "RECOVERY_SUCCESSFUL":
            st.success(f"**{final_dec}** — {final_expl}")
        elif final_dec == "NO_IMPROVEMENT":
            st.warning(f"**{final_dec}** — {final_expl}")
        else:
            st.error(f"**{final_dec}** — {final_expl}")


# ---------------------------------------------------------------------------
# PAGE 13: Future Evaluation
# ---------------------------------------------------------------------------

def page_future_evaluation():
    st.title("🎯 Future Evaluation")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    st.markdown(
        "**Critical:** All acceptance decisions are based on performance on the "
        "**independent held-out future window** — data never seen during training, "
        "synthetic generation, or quality gate evaluation."
    )

    eval_sum = exp.get("evaluation_summary", {}) or {}
    if not eval_sum:
        st.warning("No evaluation summary found.")
        return

    # Final decision banner
    final_dec = eval_sum.get("final_decision", "UNKNOWN")
    final_expl = eval_sum.get("final_explanation", "")

    if final_dec == "RECOVERY_SUCCESSFUL":
        st.success(f"**{final_dec}**")
    elif final_dec == "NO_IMPROVEMENT":
        st.warning(f"**{final_dec}**")
    else:
        st.error(f"**{final_dec}**")
    st.markdown(final_expl)

    st.markdown("---")

    # Baseline comparison
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Train F1", f"{_safe_float(eval_sum.get('baseline_train_f1')):.4f}")
    c2.metric("Baseline Future F1", f"{_safe_float(eval_sum.get('baseline_future_f1')):.4f}",
              delta=f"{_safe_float(eval_sum.get('train_to_future_drop_pct')):+.1f}% drift effect")
    c3.metric("Best Future F1", f"{_safe_float(eval_sum.get('best_future_f1')):.4f}")
    c4.metric("Recovery Gain", f"{_safe_float(eval_sum.get('overall_recovery_gain')):+.4f}")

    st.markdown("---")

    # Per-strategy detail
    st.markdown('<div class="section-header">Strategy Results on Future Data</div>',
                unsafe_allow_html=True)

    strategy_summaries = eval_sum.get("strategy_summaries", [])
    if strategy_summaries:
        for s in strategy_summaries:
            st.markdown(f"**{s['strategy'].replace('_',' ').title()}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Training F1", f"{_safe_float(s.get('training_f1')):.4f}",
                      help="Not used for acceptance decision")
            c2.metric("Future F1", f"{_safe_float(s.get('future_f1')):.4f}")
            c3.metric("Recovery Gain", f"{_safe_float(s.get('recovery_gain')):+.4f}")
            c4.metric("Synthetic Samples", s.get("n_synthetic", 0))
            c5.metric("Train Time (s)", f"{_safe_float(s.get('training_time_s')):.3f}")
            if s.get("error"):
                st.error(f"Error: {s['error']}")
            st.markdown("---")

    # Scatter: training F1 vs future F1 (overfitting check)
    if strategy_summaries:
        strat_df = pd.DataFrame(strategy_summaries)
        if "training_f1" in strat_df.columns and "future_f1" in strat_df.columns:
            fig = px.scatter(
                strat_df, x="training_f1", y="future_f1",
                text="strategy", size_max=15,
                title="Training vs Future F1 (Overfitting Check)",
                labels={"training_f1": "Training F1", "future_f1": "Future F1"},
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="#94a3b8", dash="dash"),
            )
            fig.update_traces(marker=dict(size=12, color="#3b82f6"),
                              textposition="top center")
            fig.update_layout(**_plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Points above the diagonal have better future performance than training (good). "
                       "Points below indicate overfitting to training data.")


# ---------------------------------------------------------------------------
# PAGE 14: Experiment History
# ---------------------------------------------------------------------------

def page_experiment_history():
    st.title("🗂️ Experiment History")

    if not _BACKEND_OK:
        st.error("Backend unavailable.")
        return

    try:
        db_path = get_config().database_path
        initialize_database(db_path)
        history = get_experiment_history(db_path, limit=50)
    except Exception as e:
        st.error(f"Could not load history: {e}")
        return

    if not history:
        st.info("No experiments recorded yet. Run an experiment to populate the history.")
        return

    hist_df = pd.DataFrame(history)

    # Display columns
    display_cols = [c for c in [
        "experiment_id", "experiment_name", "created_at", "dataset_name",
        "model_name", "drift_type", "baseline_train_f1", "drift_severity_index",
        "forecast_state", "best_future_f1", "final_decision", "status",
    ] if c in hist_df.columns]

    st.dataframe(hist_df[display_cols], use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("**Experiment Actions**")

    exp_ids = [row.get("experiment_id", "") for row in history]
    exp_names = [f"{row.get('experiment_name','—')[:30]} ({row.get('experiment_id','')[:8]})"
                 for row in history]

    col1, col2 = st.columns(2)
    with col1:
        sel_idx = st.selectbox("Select experiment", range(len(exp_ids)),
                               format_func=lambda i: exp_names[i],
                               key="hist_sel")
    with col2:
        action = st.radio("Action", ["View Summary", "Delete"], horizontal=True, key="hist_action")

    if st.button("Execute", key="hist_exec"):
        sel_id = exp_ids[sel_idx]
        if action == "View Summary":
            rec = get_experiment(db_path, sel_id)
            if rec:
                with st.expander(f"Experiment {sel_id[:12]}... — Summary", expanded=True):
                    st.json({k: v for k, v in rec.items()
                             if k not in ("config_json", "drift_results",
                                          "reliability_results", "recovery_results")})
            else:
                st.warning("Could not load experiment record.")
        elif action == "Delete":
            if delete_experiment(db_path, sel_id):
                st.success(f"Experiment {sel_id[:12]} deleted.")
                st.rerun()
            else:
                st.error("Delete failed.")

    # Comparison chart
    if len(hist_df) >= 2 and "baseline_train_f1" in hist_df.columns:
        st.markdown("---")
        st.markdown("**Baseline F1 Across Experiments**")
        plot_df = hist_df.dropna(subset=["baseline_train_f1"]).head(10)
        if not plot_df.empty:
            fig = px.bar(
                plot_df,
                x="experiment_name",
                y="baseline_train_f1",
                title="Baseline F1 Comparison",
                color="drift_type" if "drift_type" in plot_df.columns else "baseline_train_f1",
            )
            fig.update_layout(**_plotly_defaults())
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 15: Reports
# ---------------------------------------------------------------------------

def page_reports():
    st.title("📄 Reports")
    exp = _get_exp()
    if exp is None:
        _no_experiment_warning()
        return

    report = exp.get("report", {}) or {}

    if not report:
        st.warning("No report generated for this experiment.")
        return

    # Summary
    meta = report.get("report_metadata", {})
    st.markdown(f"**Experiment:** {meta.get('experiment_name','—')}")
    st.markdown(f"**ID:** {meta.get('experiment_id','—')}")
    st.markdown(f"**Generated:** {meta.get('generated_at','—')}")

    final_dec = report.get("final_decision", "UNKNOWN")
    if final_dec == "RECOVERY_SUCCESSFUL":
        st.success(f"Final Decision: **{final_dec}**")
    elif final_dec == "NO_IMPROVEMENT":
        st.warning(f"Final Decision: **{final_dec}**")
    else:
        st.error(f"Final Decision: **{final_dec}**")

    st.markdown(report.get("final_explanation", ""))

    # Section summaries
    st.markdown("---")
    for section_key, section_data in report.get("sections", {}).items():
        title = section_data.get("title", section_key)
        with st.expander(title):
            if "data" in section_data:
                d = section_data["data"]
                if isinstance(d, list):
                    st.dataframe(pd.DataFrame(d), use_container_width=True)
                elif isinstance(d, dict):
                    st.json(d)
            elif "check_results" in section_data:
                for cr in section_data["check_results"]:
                    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(cr["status"], "❓")
                    st.markdown(f"{icon} **{cr['check']}**: {cr['detail']}")
            else:
                st.json({k: v for k, v in section_data.items() if k != "title"})

    # Download buttons
    st.markdown("---")
    st.markdown("**Download Report**")

    import json
    from src.datadecay.utils import _NumpyEncoder

    c1, c2 = st.columns(2)
    with c1:
        report_json = json.dumps(report, indent=2, cls=_NumpyEncoder)
        st.download_button(
            "⬇️ Download JSON Report",
            data=report_json,
            file_name=f"{exp.get('experiment_id','report')}_report.json",
            mime="application/json",
        )
    with c2:
        # CSV: evaluation summary
        eval_sum = exp.get("evaluation_summary", {}) or {}
        summaries = eval_sum.get("strategy_summaries", [])
        if summaries:
            csv_data = pd.DataFrame(summaries).to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV (Strategy Comparison)",
                data=csv_data,
                file_name=f"{exp.get('experiment_id','report')}_strategies.csv",
                mime="text/csv",
            )

    # HTML snippet
    with st.expander("HTML Report Snippet"):
        html_snippet = generate_html_report_snippet(report)
        st.markdown(html_snippet, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 16: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.title("⚙️ Settings")
    st.markdown("Adjust default parameters for new experiments.")
    st.markdown("---")

    s = st.session_state["settings"]

    c1, c2 = st.columns(2)
    with c1:
        s["random_seed"] = st.number_input("Random seed", value=s["random_seed"], step=1)
        s["n_windows"] = st.slider("Default temporal windows", 6, 20, s["n_windows"])
        s["forecast_horizon"] = st.slider("Forecast horizon (windows)", 1, 6, s["forecast_horizon"])
    with c2:
        s["reliability_threshold"] = st.slider(
            "Reliability threshold", 0.40, 0.95, s["reliability_threshold"], step=0.01
        )
        s["synthetic_samples"] = st.number_input(
            "Synthetic sample count", min_value=100, max_value=5000,
            value=s["synthetic_samples"], step=100,
        )
        s["quality_gate_min"] = st.slider(
            "Quality gate minimum score", 0.40, 0.95, s["quality_gate_min"], step=0.05
        )

    st.session_state["settings"] = s

    st.markdown("---")
    st.markdown("**System Information**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Python:** {sys.version.split()[0]}")
        st.markdown(f"**DataDecay-X:** v1.0.0")
        try:
            import sklearn
            st.markdown(f"**scikit-learn:** {sklearn.__version__}")
        except Exception:
            st.markdown("**scikit-learn:** not found")
        try:
            import streamlit as _st
            st.markdown(f"**Streamlit:** {_st.__version__}")
        except Exception:
            pass

    with c2:
        try:
            import numpy as _np
            st.markdown(f"**NumPy:** {_np.__version__}")
        except Exception:
            pass
        try:
            import pandas as _pd
            st.markdown(f"**pandas:** {_pd.__version__}")
        except Exception:
            pass
        try:
            import plotly
            st.markdown(f"**Plotly:** {plotly.__version__}")
        except Exception:
            pass
        try:
            import xgboost
            st.markdown(f"**XGBoost:** {xgboost.__version__} (optional)")
        except Exception:
            st.markdown("**XGBoost:** not installed (optional)")

    # DB info
    if _BACKEND_OK:
        try:
            db_path = get_config().database_path
            st.markdown("---")
            st.markdown(f"**Database:** `{db_path}`")
            if db_path.exists():
                size_kb = db_path.stat().st_size / 1024
                st.markdown(f"**DB size:** {size_kb:.1f} KB")
            else:
                st.markdown("**DB:** Not yet created")
        except Exception:
            pass

    if st.button("Save Settings"):
        st.success("Settings saved to session.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main():
    if not _BACKEND_OK:
        st.error(
            f"DataDecay-X backend failed to import. "
            f"Please install dependencies:\n\n"
            f"```\npip install -r requirements.txt\n```\n\n"
            f"Error: {_import_err_msg}"
        )
        return

    render_sidebar()

    page = st.session_state.get("page", "Overview")

    routes = {
        "Overview": page_overview,
        "New Experiment": page_new_experiment,
        "Dataset": page_dataset,
        "Baseline Model": page_baseline_model,
        "Drift Lab": page_drift_lab,
        "Drift Monitor": page_drift_monitor,
        "Model Reliability": page_model_reliability,
        "Reliability Forecast": page_reliability_forecast,
        "Synthetic Lab": page_synthetic_lab,
        "Quality Gate": page_quality_gate,
        "Recovery Center": page_recovery_center,
        "Recovery Comparison": page_recovery_comparison,
        "Future Evaluation": page_future_evaluation,
        "Experiment History": page_experiment_history,
        "Reports": page_reports,
        "Settings": page_settings,
    }

    handler = routes.get(page, page_overview)
    try:
        handler()
    except Exception as exc:
        st.error(f"Page error: {exc}")
        with st.expander("Debug traceback"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
