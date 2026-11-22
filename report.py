# report.py
# Builds a downloadable evaluation report (Markdown + JSON).

import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas values into JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.4f}"
    return str(value)


_SECTIONS = {
    "fidelity": [
        ("overall_fidelity", "Overall fidelity"),
        ("ks_mean", "KS similarity (mean)"),
        ("chi2_mean", "Chi-square similarity (mean)"),
        ("correlation_similarity", "Correlation similarity"),
        ("pmse_similarity", "Propensity similarity"),
        ("real_vs_synthetic_auc", "Real vs synthetic AUC"),
    ],
    "privacy": [
        ("privacy_score", "Privacy score"),
        ("membership_inference_auc", "Membership inference AUC"),
        ("epsilon", "Differential privacy ε"),
        ("delta", "Differential privacy δ"),
    ],
    "utility": [
        ("utility_ratio", "Utility ratio"),
        ("real_model_accuracy", "Real model accuracy"),
        ("synthetic_model_accuracy", "Synthetic model accuracy"),
    ],
}


def build_report_data(
    evaluation: Dict[str, Any],
    dcr: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect everything that goes into the report, without the raw DCR arrays."""
    dcr_summary = None
    if dcr:
        dcr_summary = {k: v for k, v in dcr.items() if not k.endswith("_values")}
    return to_serializable({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": meta,
        "evaluation": evaluation,
        "record_level_privacy": dcr_summary,
    })


def build_markdown_report(report_data: Dict[str, Any]) -> str:
    lines = ["# Synthetic Data Evaluation Report", ""]
    lines.append(f"_Generated: {report_data['generated_at']}_")
    lines.append("")

    run = report_data.get("run") or {}
    if run:
        lines += ["## Run details", "", "| Setting | Value |", "|---|---|"]
        for key, value in run.items():
            lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")

    evaluation = report_data.get("evaluation") or {}
    for section, rows in _SECTIONS.items():
        metrics = evaluation.get(section)
        if not metrics:
            continue
        lines += [f"## {section.capitalize()}", "", "| Metric | Value |", "|---|---|"]
        for key, label in rows:
            if key in metrics:
                lines.append(f"| {label} | {_fmt(metrics[key])} |")
        lines.append("")

    dcr = report_data.get("record_level_privacy")
    if dcr:
        lines += [
            "## Record-level privacy (Distance to Closest Record)",
            "",
            f"**Risk level: {dcr['risk_level']}**",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Synthetic → real DCR (median) | {_fmt(dcr['synthetic_dcr_median'])} |",
            f"| Synthetic → real DCR (5th percentile) | {_fmt(dcr['synthetic_dcr_5th_percentile'])} |",
            f"| Real → real baseline DCR (median) | {_fmt(dcr['real_baseline_dcr_median'])} |",
            f"| DCR ratio (synthetic / baseline) | {_fmt(dcr['dcr_ratio'])} |",
            f"| Exact copies of real rows | {dcr['exact_match_count']} ({dcr['exact_match_rate']:.2%}) |",
            "",
            "A DCR ratio close to or above 1 means synthetic rows are no closer to real "
            "records than real records are to each other. Values well below 1, or any "
            "exact copies, suggest the model may be memorising training data.",
            "",
        ]

    return "\n".join(lines)


def build_json_report(report_data: Dict[str, Any]) -> str:
    return json.dumps(report_data, indent=2)
