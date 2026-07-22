"""Active Strategy Doctor repair workbench for Project NERO.

The workbench converts quarantine verdicts into a visible repair queue. It does
not silently change production rules; it maps weak strategies to versioned
repair candidates that must be tested in Strategy TEST Lab.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nero_app.core.strategy_lab_agent import CANDIDATES
from nero_app.core.strategy_quarantine import DEFAULT_QUARANTINE_CSV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SUMMARY_CSV = DEFAULT_REPORT_DIR / "strategy_lab_summary.csv"
DEFAULT_REPAIR_CSV = DEFAULT_REPORT_DIR / "strategy_repair_workbench.csv"
DEFAULT_REPAIR_JSON = DEFAULT_REPORT_DIR / "strategy_repair_workbench.json"

REPAIR_MAP = {
    "BREAKOUT_MOMENTUM_V1": "REPAIR_BREAKOUT_QUALITY_V1",
    "V2_BREAKOUT_RETEST": "REPAIR_BREAKOUT_QUALITY_V1",
    "MR_REGIME_FILTER_V1": "REPAIR_MR_REGIME_LATE_V1",
    "MR_RELAXED_PULLBACK_V1": "REPAIR_MR_REGIME_LATE_V1",
    "MR_TARGET_1R_V1": "REPAIR_MR_1R_ASYMMETRIC_V1",
}


@dataclass(frozen=True)
class RepairWorkbenchRow:
    quarantined_strategy: str
    quarantined_label: str
    quarantined_net_pnl: float
    quarantined_trades: int
    diagnosis: str
    repair_candidate: str
    repair_label: str
    repair_status: str
    repair_trades: int
    repair_net_pnl: float
    next_action: str
    release_gate: str


def build_strategy_repair_workbench(
    quarantine_csv: Path = DEFAULT_QUARANTINE_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    output_csv: Path = DEFAULT_REPAIR_CSV,
    output_json: Path = DEFAULT_REPAIR_JSON,
) -> pd.DataFrame:
    quarantine = _safe_read_csv(quarantine_csv)
    summary = _safe_read_csv(summary_csv)
    rows = [_repair_row(row, summary) for row in quarantine.to_dict("records")]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["repair_status", "quarantined_net_pnl"], ascending=[True, True])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(report.to_dict("records"), indent=2), encoding="utf-8")
    return report


def _repair_row(row: dict[str, Any], summary: pd.DataFrame) -> RepairWorkbenchRow:
    candidate_id = str(row.get("candidate_id", "UNKNOWN") or "UNKNOWN")
    repair_id = REPAIR_MAP.get(candidate_id, "")
    repair_spec = CANDIDATES.get(repair_id)
    repair_summary = _summary_row(summary, repair_id)
    repair_trades = int(_num(repair_summary.get("total_trades"), 0)) if repair_summary else 0
    repair_net = _num(repair_summary.get("net_pnl"), 0.0) if repair_summary else 0.0
    status = _repair_status(repair_id, repair_trades, repair_net)
    return RepairWorkbenchRow(
        quarantined_strategy=candidate_id,
        quarantined_label=str(row.get("display_label", candidate_id) or candidate_id),
        quarantined_net_pnl=round(_num(row.get("net_pnl"), 0.0), 2),
        quarantined_trades=int(_num(row.get("total_trades"), 0)),
        diagnosis=_diagnosis(candidate_id, row),
        repair_candidate=repair_id or "NEEDS_DESIGN",
        repair_label=(repair_spec.display_label if repair_spec else "NEEDS_DESIGN") if repair_id else "NEEDS_DESIGN",
        repair_status=status,
        repair_trades=repair_trades,
        repair_net_pnl=round(repair_net, 2),
        next_action=_next_action(status),
        release_gate="30+ repair trades, expectancy > 0, profit factor >= 1.10, max drawdown acceptable, beats parent and random baseline.",
    )


def _repair_status(repair_id: str, repair_trades: int, repair_net: float) -> str:
    if not repair_id:
        return "DESIGN_REQUIRED"
    if repair_id not in CANDIDATES:
        return "NOT_REGISTERED"
    if repair_trades <= 0:
        return "DEPLOYED_AWAITING_TRADES"
    if repair_trades < 30:
        return "DEPLOYED_COLLECTING_EVIDENCE"
    if repair_net > 0:
        return "REPAIR_PROMISING_REVIEW"
    return "REPAIR_WEAK_REWORK"


def _next_action(status: str) -> str:
    if status == "DESIGN_REQUIRED":
        return "Design a versioned repair candidate before redeployment."
    if status == "NOT_REGISTERED":
        return "Register the repair candidate in Strategy TEST Lab."
    if status == "DEPLOYED_AWAITING_TRADES":
        return "Keep GitHub Strategy Lab running; wait for first repair trades."
    if status == "DEPLOYED_COLLECTING_EVIDENCE":
        return "Continue paper testing until at least 30 repair trades."
    if status == "REPAIR_PROMISING_REVIEW":
        return "Compare against parent losses and prepare promotion review."
    return "Send repair back to Strategy Doctor for new hypothesis split."


def _diagnosis(candidate_id: str, row: dict[str, Any]) -> str:
    reason = str(row.get("reason", "") or "")
    if "BREAKOUT" in candidate_id:
        return "Breakout edge is weak; likely fakeout/poor trend-quality filter. " + reason
    if "TARGET_1R" in candidate_id:
        return "Exit logic is asymmetric; target/stop reward quality needs repair. " + reason
    if "MR_" in candidate_id:
        return "Mean-reversion entry is too early or regime filter is weak. " + reason
    return reason or "Quarantine requires repair analysis."


def _summary_row(summary: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    if summary.empty or not candidate_id or "candidate_id" not in summary:
        return {}
    matches = summary[summary["candidate_id"].astype(str) == candidate_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()
