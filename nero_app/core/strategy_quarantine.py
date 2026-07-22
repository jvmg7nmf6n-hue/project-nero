"""Strategy quarantine automation for Project NERO.

Reads Strategy Verification Engine output and turns QUARANTINE verdicts into an
auditable block list for future paper entries. This module never deletes old
records and never touches real orders.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_VERIFICATION_CSV = DEFAULT_REPORT_DIR / "strategy_verification_report.csv"
DEFAULT_QUARANTINE_CSV = DEFAULT_REPORT_DIR / "strategy_quarantine_report.csv"
DEFAULT_QUARANTINE_JSON = DEFAULT_REPORT_DIR / "strategy_quarantine_report.json"


@dataclass(frozen=True)
class StrategyQuarantineRow:
    candidate_id: str
    display_label: str
    verdict: str
    blocked: bool
    total_trades: int
    net_pnl: float
    expectancy_r: float
    profit_factor: float
    reason: str
    action: str


def build_strategy_quarantine_report(
    verification_csv: Path = DEFAULT_VERIFICATION_CSV,
    output_csv: Path = DEFAULT_QUARANTINE_CSV,
    output_json: Path = DEFAULT_QUARANTINE_JSON,
) -> pd.DataFrame:
    """Build a report of strategies blocked by verification verdicts."""
    verification = _safe_read_csv(verification_csv)
    rows: list[StrategyQuarantineRow] = []
    if not verification.empty:
        for row in verification.to_dict("records"):
            if str(row.get("verdict", "")).upper() == "QUARANTINE":
                rows.append(_quarantine_row(row))
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["net_pnl", "total_trades"], ascending=[True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(report.to_dict("records"), indent=2), encoding="utf-8")
    return report


def load_quarantined_strategy_ids(path: Path = DEFAULT_QUARANTINE_CSV) -> set[str]:
    """Return candidate IDs that should not open new paper entries."""
    report = _safe_read_csv(path)
    if report.empty:
        return set()
    if "blocked" in report:
        report = report[report["blocked"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if "candidate_id" not in report:
        return set()
    return {str(value).strip() for value in report["candidate_id"].dropna() if str(value).strip()}


def _quarantine_row(row: dict[str, Any]) -> StrategyQuarantineRow:
    return StrategyQuarantineRow(
        candidate_id=str(row.get("candidate_id", "UNKNOWN") or "UNKNOWN"),
        display_label=str(row.get("display_label", row.get("candidate_id", "UNKNOWN")) or "UNKNOWN"),
        verdict="QUARANTINE",
        blocked=True,
        total_trades=int(_num(row.get("total_trades"), 0)),
        net_pnl=round(_num(row.get("net_pnl"), 0.0), 2),
        expectancy_r=round(_num(row.get("expectancy_r"), 0.0), 4),
        profit_factor=round(_num(row.get("profit_factor"), 0.0), 4),
        reason=str(row.get("primary_reason", "Verification Engine marked this strategy for quarantine.") or "Verification Engine marked this strategy for quarantine."),
        action=str(row.get("action", "Pause new entries until review.") or "Pause new entries until review."),
    )


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
