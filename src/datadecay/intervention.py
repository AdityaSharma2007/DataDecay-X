"""
intervention.py — Intervention decision engine for DataDecay-X.

Evaluates drift severity, reliability degradation, and forecast trend
to recommend one of: STABLE / WATCH / RECOVERY_RECOMMENDED.

Does NOT automatically retrain. Requires user approval for recovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# State constants
STATE_STABLE = "STABLE"
STATE_WATCH = "WATCH"
STATE_RECOVERY_RECOMMENDED = "RECOVERY_RECOMMENDED"

VALID_STATES = [STATE_STABLE, STATE_WATCH, STATE_RECOVERY_RECOMMENDED]


def determine_intervention(
    current_reliability: float,
    baseline_reliability: float,
    degradation_pct: float,
    drift_severity_index: float,
    forecast_trend: str,
    threshold_crossing_estimate: Optional[str],
    threshold: float = 0.70,
    crossing_detected: bool = False,
    already_below_threshold: bool = False,
    warning_lead_windows: int = 999,
) -> Dict[str, Any]:
    """
    Determine the intervention state based on multiple signals.

    Decision logic (engineering rules — configurable):

    RECOVERY_RECOMMENDED if ANY of:
      - Current reliability already below threshold
      - Threshold crossing detected within forecast horizon AND lead < 3
      - Degradation > 20% AND drift severity > 0.5
      - Drift severity "severe" AND reliability declining

    WATCH if ANY of:
      - Threshold crossing detected (any lead time)
      - Degradation between 10-20%
      - Drift severity "moderate" or "high"
      - Forecast trend is declining

    Otherwise: STABLE

    Parameters
    ----------
    current_reliability         : latest observed reliability (e.g. F1)
    baseline_reliability        : baseline model reliability
    degradation_pct             : % degradation from baseline
    drift_severity_index        : 0-1 index from detection module
    forecast_trend              : 'declining', 'stable', 'improving'
    threshold_crossing_estimate : human-readable TTD string
    threshold                   : reliability threshold
    crossing_detected           : whether TTD estimate found a crossing
    already_below_threshold     : whether current reliability < threshold
    warning_lead_windows        : lead time in windows before crossing

    Returns
    -------
    dict with: state, reasons, recommended_action, evidence_summary,
               confidence (low/medium/high)
    """
    reasons: List[str] = []
    state = STATE_STABLE

    # -----------------------------------------------------------------------
    # Evidence collection
    # -----------------------------------------------------------------------

    # 1. Already below threshold
    if already_below_threshold:
        reasons.append(
            f"Current reliability ({current_reliability:.3f}) is below the "
            f"configured threshold ({threshold})."
        )

    # 2. Threshold crossing soon
    if crossing_detected and not already_below_threshold:
        if warning_lead_windows <= 2:
            reasons.append(
                f"Threshold crossing predicted within {warning_lead_windows} window(s) "
                "[forecast estimate]."
            )
        else:
            reasons.append(
                f"Threshold crossing predicted within forecast horizon "
                f"(lead: {warning_lead_windows} windows) [forecast estimate]."
            )

    # 3. Degradation severity
    if degradation_pct >= 20:
        reasons.append(
            f"Significant reliability degradation: {degradation_pct:.1f}% below baseline."
        )
    elif degradation_pct >= 10:
        reasons.append(
            f"Moderate reliability degradation: {degradation_pct:.1f}% below baseline."
        )

    # 4. Drift severity
    if drift_severity_index >= 0.7:
        reasons.append(
            f"High drift severity index: {drift_severity_index:.3f} "
            "(engineering threshold — not universal)."
        )
    elif drift_severity_index >= 0.4:
        reasons.append(
            f"Moderate drift severity index: {drift_severity_index:.3f}."
        )

    # 5. Forecast trend
    if forecast_trend == "declining":
        reasons.append("Reliability forecast trend is declining.")
    elif forecast_trend == "improving":
        reasons.append("Reliability forecast trend is improving.")

    # -----------------------------------------------------------------------
    # State determination
    # -----------------------------------------------------------------------

    recovery_triggers = 0
    watch_triggers = 0

    if already_below_threshold:
        recovery_triggers += 3  # Strong signal
    if crossing_detected and warning_lead_windows <= 2:
        recovery_triggers += 2
    if degradation_pct >= 20 and drift_severity_index >= 0.5:
        recovery_triggers += 2
    if drift_severity_index >= 0.7 and forecast_trend == "declining":
        recovery_triggers += 2

    if crossing_detected and warning_lead_windows > 2:
        watch_triggers += 2
    if 10 <= degradation_pct < 20:
        watch_triggers += 1
    if 0.4 <= drift_severity_index < 0.7:
        watch_triggers += 1
    if forecast_trend == "declining" and not crossing_detected:
        watch_triggers += 1

    if recovery_triggers >= 2:
        state = STATE_RECOVERY_RECOMMENDED
    elif recovery_triggers == 1 or watch_triggers >= 2:
        state = STATE_WATCH
    else:
        state = STATE_STABLE

    # -----------------------------------------------------------------------
    # Recommended action
    # -----------------------------------------------------------------------

    if state == STATE_RECOVERY_RECOMMENDED:
        recommended_action = (
            "Consider initiating a recovery experiment. "
            "Options: retrain on recent real data, generate synthetic recovery data, "
            "or run a hybrid strategy. Evaluate all candidates on independent future data."
        )
        confidence = "high" if recovery_triggers >= 3 else "medium"
    elif state == STATE_WATCH:
        recommended_action = (
            "Monitor closely for the next window. "
            "Prepare a recovery experiment plan but do not intervene yet. "
            "Check drift metrics and forecast updates."
        )
        confidence = "medium"
    else:
        recommended_action = (
            "No action required. Continue monitoring. "
            "The model appears stable within the current forecast horizon."
        )
        confidence = "high" if len(reasons) == 0 else "medium"

    # -----------------------------------------------------------------------
    # Evidence summary
    # -----------------------------------------------------------------------

    evidence = {
        "current_reliability": round(current_reliability, 4),
        "baseline_reliability": round(baseline_reliability, 4),
        "degradation_pct": round(degradation_pct, 2),
        "drift_severity_index": round(drift_severity_index, 4),
        "forecast_trend": forecast_trend,
        "crossing_detected": crossing_detected,
        "already_below_threshold": already_below_threshold,
        "warning_lead_windows": warning_lead_windows,
        "threshold": threshold,
        "ttd_estimate": threshold_crossing_estimate,
    }

    if not reasons:
        reasons.append(
            "No significant drift, degradation, or forecast warning detected."
        )

    logger.info("Intervention state: %s (confidence=%s)", state, confidence)

    return {
        "state": state,
        "reasons": reasons,
        "recommended_action": recommended_action,
        "confidence": confidence,
        "evidence": evidence,
        "recovery_trigger_count": recovery_triggers,
        "watch_trigger_count": watch_triggers,
    }


def intervention_state_color(state: str) -> str:
    """Return a CSS/display color string for the intervention state."""
    mapping = {
        STATE_STABLE: "green",
        STATE_WATCH: "orange",
        STATE_RECOVERY_RECOMMENDED: "red",
    }
    return mapping.get(state, "gray")


def intervention_state_icon(state: str) -> str:
    """Return a simple text icon for the intervention state."""
    mapping = {
        STATE_STABLE: "[OK]",
        STATE_WATCH: "[!!]",
        STATE_RECOVERY_RECOMMENDED: "[ALERT]",
    }
    return mapping.get(state, "[?]")
