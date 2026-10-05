"""
reporting.py — Experiment report generation for DataDecay-X.

Generates JSON and CSV reports from experiment state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import ensure_dir, save_json, save_csv, _NumpyEncoder, iso_now

logger = logging.getLogger(__name__)


def generate_experiment_report(
    experiment_id: str,
    experiment_name: str,
    config: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    split_info: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    drift_results: Optional[Dict[str, Any]],
    reliability_results: Optional[Any],  # DataFrame
    forecast_result: Optional[Dict[str, Any]],
    ttd_result: Optional[Dict[str, Any]],
    intervention_result: Optional[Dict[str, Any]],
    synthetic_metadata: Optional[Dict[str, Any]],
    quality_gate_result: Optional[Dict[str, Any]],
    recovery_comparison: Optional[Any],  # DataFrame
    evaluation_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assemble a complete experiment report as a structured dict.

    All sections are included only if available.
    """
    report: Dict[str, Any] = {
        "report_metadata": {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "generated_at": iso_now(),
            "project": "DataDecay-X v1.0.0",
        },
        "sections": {},
    }

    # 1. Configuration
    report["sections"]["1_configuration"] = {
        "title": "Experiment Configuration",
        "data": config,
    }

    # 2. Dataset
    report["sections"]["2_dataset"] = {
        "title": "Dataset Metadata",
        "data": dataset_summary,
    }

    # 3. Chronological split
    report["sections"]["3_chronological_split"] = {
        "title": "Chronological Split",
        "data": split_info,
    }

    # 4. Baseline model
    report["sections"]["4_baseline_model"] = {
        "title": "Baseline Model Performance",
        "data": baseline_metrics,
    }

    # 5. Drift configuration and results
    if drift_results:
        overall = drift_results.get("overall_severity", {})
        wdt = drift_results.get("window_drift_table")
        report["sections"]["5_drift"] = {
            "title": "Drift Detection Results",
            "overall_severity": overall,
            "window_drift_summary": (
                wdt.to_dict(orient="records") if wdt is not None and hasattr(wdt, "to_dict") else []
            ),
        }

    # 6. Model reliability timeline
    if reliability_results is not None and hasattr(reliability_results, "to_dict"):
        report["sections"]["6_reliability"] = {
            "title": "Model Reliability Timeline",
            "data": reliability_results.to_dict(orient="records"),
        }

    # 7. Forecast
    if forecast_result:
        report["sections"]["7_forecast"] = {
            "title": "Reliability Forecast",
            "method": forecast_result.get("method", ""),
            "horizon": forecast_result.get("horizon", 0),
            "trend_metadata": forecast_result.get("trend_metadata", {}),
            "forecast_values": (
                forecast_result["forecast_values"].tolist()
                if hasattr(forecast_result.get("forecast_values"), "tolist")
                else list(forecast_result.get("forecast_values", []))
            ),
        }

    # 8. TTD estimate
    if ttd_result:
        report["sections"]["8_ttd_estimate"] = {
            "title": "Time-to-Decay Estimate (ESTIMATE ONLY — not a guarantee)",
            "data": ttd_result,
        }

    # 9. Intervention decision
    if intervention_result:
        report["sections"]["9_intervention"] = {
            "title": "Intervention Decision",
            "state": intervention_result.get("state", ""),
            "reasons": intervention_result.get("reasons", []),
            "recommended_action": intervention_result.get("recommended_action", ""),
            "confidence": intervention_result.get("confidence", ""),
            "evidence": intervention_result.get("evidence", {}),
        }

    # 10. Synthetic generation
    if synthetic_metadata:
        report["sections"]["10_synthetic_generation"] = {
            "title": "Synthetic Data Generation",
            "method": synthetic_metadata.get("method", ""),
            "n_samples": synthetic_metadata.get("n_samples_generated", 0),
            "generation_time_s": synthetic_metadata.get("generation_time_s", 0),
            "configuration": {
                k: v for k, v in synthetic_metadata.items()
                if k not in ("source_stats", "synthetic_stats")
            },
        }

    # 11. Quality gate
    if quality_gate_result:
        report["sections"]["11_quality_gate"] = {
            "title": "Synthetic Data Quality Gate",
            "quality_score": quality_gate_result.get("quality_score", 0),
            "decision": quality_gate_result.get("decision", ""),
            "passed_checks": quality_gate_result.get("passed_checks", []),
            "warned_checks": quality_gate_result.get("warned_checks", []),
            "failed_checks": quality_gate_result.get("failed_checks", []),
            "check_results": quality_gate_result.get("check_results", []),
        }

    # 12. Recovery comparison
    if recovery_comparison is not None and hasattr(recovery_comparison, "to_dict"):
        report["sections"]["12_recovery_comparison"] = {
            "title": "Recovery Strategy Comparison",
            "note": (
                "All strategies evaluated on INDEPENDENT future data. "
                "Training performance is shown for reference only."
            ),
            "data": recovery_comparison.to_dict(orient="records"),
        }

    # 13. Final evaluation summary
    if evaluation_summary:
        report["sections"]["13_evaluation_summary"] = {
            "title": "Future Evaluation & Final Decision",
            "note": (
                "Final decision based on future held-out data performance. "
                "Negative results are reported honestly."
            ),
            "data": evaluation_summary,
        }

    # 14. Final decision (top-level)
    final_decision = "UNKNOWN"
    final_explanation = "Evaluation not completed."
    if evaluation_summary:
        final_decision = evaluation_summary.get("final_decision", "UNKNOWN")
        final_explanation = evaluation_summary.get("final_explanation", "")

    report["final_decision"] = final_decision
    report["final_explanation"] = final_explanation

    return report


def save_report_json(
    report: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Save the experiment report as a JSON file.

    Returns the path written.
    """
    ensure_dir(output_path.parent)
    p = save_json(report, output_path)
    logger.info("Report saved to %s", p)
    return p


def save_report_csv(
    report: Dict[str, Any],
    output_dir: Path,
    experiment_id: str,
) -> List[Path]:
    """
    Save key sections of the report as CSV files.

    Returns list of paths written.
    """
    ensure_dir(output_dir)
    paths = []

    # Drift results
    drift_section = report.get("sections", {}).get("5_drift", {})
    drift_data = drift_section.get("window_drift_summary", [])
    if drift_data:
        df = pd.DataFrame(drift_data)
        p = save_csv(df, output_dir / f"{experiment_id}_drift_results.csv")
        paths.append(p)

    # Reliability timeline
    rel_section = report.get("sections", {}).get("6_reliability", {})
    rel_data = rel_section.get("data", [])
    if rel_data:
        df = pd.DataFrame(rel_data)
        p = save_csv(df, output_dir / f"{experiment_id}_reliability_timeline.csv")
        paths.append(p)

    # Recovery comparison
    rec_section = report.get("sections", {}).get("12_recovery_comparison", {})
    rec_data = rec_section.get("data", [])
    if rec_data:
        df = pd.DataFrame(rec_data)
        p = save_csv(df, output_dir / f"{experiment_id}_recovery_comparison.csv")
        paths.append(p)

    # Summary row
    summary_row = {
        "experiment_id": report.get("report_metadata", {}).get("experiment_id", ""),
        "experiment_name": report.get("report_metadata", {}).get("experiment_name", ""),
        "generated_at": report.get("report_metadata", {}).get("generated_at", ""),
        "final_decision": report.get("final_decision", ""),
        "final_explanation": report.get("final_explanation", ""),
    }
    eval_data = report.get("sections", {}).get("13_evaluation_summary", {}).get("data", {})
    if eval_data:
        summary_row.update({
            "baseline_train_f1": eval_data.get("baseline_train_f1", ""),
            "baseline_future_f1": eval_data.get("baseline_future_f1", ""),
            "best_strategy": eval_data.get("best_strategy", ""),
            "best_future_f1": eval_data.get("best_future_f1", ""),
            "overall_recovery_gain": eval_data.get("overall_recovery_gain", ""),
        })

    summary_df = pd.DataFrame([summary_row])
    p = save_csv(summary_df, output_dir / f"{experiment_id}_summary.csv")
    paths.append(p)

    logger.info("CSV reports saved to %s (%d files)", output_dir, len(paths))
    return paths


def generate_html_report_snippet(report: Dict[str, Any]) -> str:
    """
    Generate a simple HTML summary snippet for the experiment report.
    Suitable for embedding in the Streamlit UI or exporting.
    """
    meta = report.get("report_metadata", {})
    final_decision = report.get("final_decision", "UNKNOWN")
    final_expl = report.get("final_explanation", "")

    color_map = {
        "RECOVERY_SUCCESSFUL": "#22c55e",
        "NO_IMPROVEMENT": "#f59e0b",
        "RECOVERY_FAILED": "#ef4444",
        "UNKNOWN": "#6b7280",
    }
    color = color_map.get(final_decision, "#6b7280")

    lines = [
        f"<h2>DataDecay-X Experiment Report</h2>",
        f"<p><strong>ID:</strong> {meta.get('experiment_id', '')}</p>",
        f"<p><strong>Name:</strong> {meta.get('experiment_name', '')}</p>",
        f"<p><strong>Generated:</strong> {meta.get('generated_at', '')}</p>",
        f"<hr>",
        f"<h3 style='color:{color}'>Final Decision: {final_decision}</h3>",
        f"<p>{final_expl}</p>",
    ]

    # Evaluation summary table
    eval_data = report.get("sections", {}).get("13_evaluation_summary", {}).get("data", {})
    if eval_data:
        lines.append("<h3>Evaluation Summary</h3><table border='1' cellpadding='4'>")
        lines.append("<tr><th>Metric</th><th>Value</th></tr>")
        for key in ["baseline_train_f1", "baseline_future_f1", "best_strategy",
                    "best_future_f1", "overall_recovery_gain"]:
            val = eval_data.get(key, "")
            lines.append(f"<tr><td>{key}</td><td>{val}</td></tr>")
        lines.append("</table>")

    return "\n".join(lines)
