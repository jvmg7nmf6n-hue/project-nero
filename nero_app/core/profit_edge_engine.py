"""Profit Edge Engine for Project NERO.

The engine does not place real trades and does not claim profit certainty. It
turns Strategy Lab evidence into a strict paper-allocation and loss-recovery
view so NERO can focus on candidates with positive evidence while keeping
capital-draining strategies blocked.
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
DEFAULT_EDGE_CSV = DEFAULT_REPORT_DIR / "profit_edge_report.csv"
DEFAULT_EDGE_JSON = DEFAULT_REPORT_DIR / "profit_edge_report.json"

MIN_FOCUS_TRADES = 5
TARGET_RECOVERY_MULTIPLE = 1.0


@dataclass(frozen=True)
class ProfitEdgeRow:
    candidate_id: str
    display_label: str
    role: str
    verdict: str
    total_trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    net_pnl: float
    max_drawdown: float
    edge_score: float
    paper_weight: float
    recovery_priority: int
    decision: str
    reason: str


@dataclass(frozen=True)
class ProfitEdgeSummary:
    total_strategies: int
    profit_candidates: int
    capital_drains: int
    evidence_pool_pnl: float
    blocked_drag_pnl: float
    top_candidate: str
    recovery_gap: float
    recovery_ratio: float
    status: str
    notes: list[str]


def build_profit_edge_report(
    verification_csv: Path = DEFAULT_VERIFICATION_CSV,
    quarantine_csv: Path = DEFAULT_QUARANTINE_CSV,
    output_csv: Path = DEFAULT_EDGE_CSV,
    output_json: Path = DEFAULT_EDGE_JSON,
) -> tuple[pd.DataFrame, ProfitEdgeSummary]:
    """Build and persist the profit edge view from existing evidence reports."""
    verification = _safe_read_csv(verification_csv)
    quarantine = _safe_read_csv(quarantine_csv)
    rows = [_edge_row(row) for row in verification.to_dict("records")]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["role", "edge_score", "net_pnl"], ascending=[True, False, False])
    summary = _summary(report, quarantine)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps({"summary": asdict(summary), "rows": report.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    return report, summary


def _edge_row(row: dict[str, Any]) -> ProfitEdgeRow:
    candidate_id = str(row.get("candidate_id", "UNKNOWN") or "UNKNOWN")
    display_label = str(row.get("display_label", candidate_id) or candidate_id)
    verdict = str(row.get("verdict", "UNKNOWN") or "UNKNOWN")
    total_trades = int(_num(row.get("total_trades"), 0))
    win_rate = _num(row.get("win_rate"), 0.0)
    expectancy_r = _num(row.get("expectancy_r"), 0.0)
    profit_factor = _num(row.get("profit_factor"), 0.0)
    net_pnl = _num(row.get("net_pnl"), 0.0)
    max_drawdown = _num(row.get("max_drawdown"), 0.0)
    role = _role(verdict, total_trades, expectancy_r, profit_factor, net_pnl)
    score = _edge_score(role, total_trades, expectancy_r, profit_factor, net_pnl, max_drawdown)
    weight = _paper_weight(role, total_trades, score)
    decision, reason = _decision(role, total_trades, expectancy_r, profit_factor, net_pnl)
    return ProfitEdgeRow(
        candidate_id=candidate_id,
        display_label=display_label,
        role=role,
        verdict=verdict,
        total_trades=total_trades,
        win_rate=round(win_rate, 4),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 4),
        net_pnl=round(net_pnl, 2),
        max_drawdown=round(max_drawdown, 4),
        edge_score=round(score, 2),
        paper_weight=round(weight, 3),
        recovery_priority=_priority(role, score, total_trades),
        decision=decision,
        reason=reason,
    )


def _role(verdict: str, total_trades: int, expectancy_r: float, profit_factor: float, net_pnl: float) -> str:
    if verdict == "QUARANTINE":
        return "CAPITAL_DRAIN"
    if total_trades >= MIN_FOCUS_TRADES and net_pnl > 0 and expectancy_r > 0 and profit_factor >= 1.0:
        return "PROFIT_CANDIDATE"
    if net_pnl < 0 and total_trades >= MIN_FOCUS_TRADES:
        return "WEAK_CANDIDATE"
    return "TOO_EARLY"


def _edge_score(role: str, total_trades: int, expectancy_r: float, profit_factor: float, net_pnl: float, max_drawdown: float) -> float:
    if role == "CAPITAL_DRAIN":
        return 0.0
    sample = min(25.0, total_trades / 30.0 * 25.0)
    expectancy = max(-20.0, min(30.0, expectancy_r * 40.0))
    pf = max(-15.0, min(25.0, (profit_factor - 1.0) * 25.0)) if profit_factor else -10.0
    pnl = 15.0 if net_pnl > 0 else (-15.0 if net_pnl < 0 else 0.0)
    drawdown_penalty = min(20.0, abs(max_drawdown) * 100.0)
    score = 50.0 + sample + expectancy + pf + pnl - drawdown_penalty
    if role == "TOO_EARLY":
        score = min(score, 55.0)
    if role == "WEAK_CANDIDATE":
        score = min(score, 35.0)
    return max(0.0, min(100.0, score))


def _paper_weight(role: str, total_trades: int, score: float) -> float:
    if role != "PROFIT_CANDIDATE":
        return 0.0
    if total_trades < 10:
        return 0.05
    if total_trades < 20:
        return min(0.15, score / 1000.0)
    if total_trades < 30:
        return min(0.25, score / 700.0)
    return min(0.40, score / 500.0)


def _priority(role: str, score: float, total_trades: int) -> int:
    if role != "PROFIT_CANDIDATE":
        return 0
    if total_trades >= 20 and score >= 70:
        return 1
    if total_trades >= 10:
        return 2
    return 3


def _decision(role: str, total_trades: int, expectancy_r: float, profit_factor: float, net_pnl: float) -> tuple[str, str]:
    if role == "CAPITAL_DRAIN":
        return "BLOCK_NEW_ENTRIES", "Negative evidence is strong enough to protect the paper account."
    if role == "PROFIT_CANDIDATE":
        if total_trades < 30:
            return "FOCUS_PAPER_ONLY", "Positive edge is visible but below the promotion sample gate."
        return "EXPAND_PAPER_TEST", "Positive edge reached minimum sample gate; still paper-only until robustness tests pass."
    if role == "WEAK_CANDIDATE":
        return "REDUCE_OR_REPAIR", "Current sample is negative; send to Strategy Doctor if it keeps losing."
    return "COLLECT_MORE_DATA", "Not enough closed trades for a profit decision."


def _summary(report: pd.DataFrame, quarantine: pd.DataFrame) -> ProfitEdgeSummary:
    if report.empty:
        return ProfitEdgeSummary(0, 0, 0, 0.0, 0.0, "-", 0.0, 0.0, "NO_DATA", ["No Strategy Lab evidence is available."])
    profit_pool = report[report["role"] == "PROFIT_CANDIDATE"]
    drains = report[report["role"] == "CAPITAL_DRAIN"]
    evidence_pool_pnl = float(profit_pool["net_pnl"].sum()) if not profit_pool.empty else 0.0
    blocked_drag_pnl = float(drains["net_pnl"].sum()) if not drains.empty else 0.0
    if not quarantine.empty and "net_pnl" in quarantine:
        blocked_drag_pnl = float(pd.to_numeric(quarantine["net_pnl"], errors="coerce").fillna(0.0).sum())
    top_candidate = "-"
    if not profit_pool.empty:
        top_candidate = str(profit_pool.sort_values(["edge_score", "net_pnl"], ascending=False).iloc[0]["display_label"])
    recovery_gap = max(0.0, abs(blocked_drag_pnl) * TARGET_RECOVERY_MULTIPLE - evidence_pool_pnl)
    recovery_ratio = evidence_pool_pnl / abs(blocked_drag_pnl) if blocked_drag_pnl < 0 else 0.0
    notes = [
        "Focus only on positive-expectancy paper candidates; do not add weight to quarantined systems.",
        "A candidate needs 30+ trades, positive expectancy, PF above 1.10, and robustness checks before promotion.",
    ]
    status = "EDGE_FOUND_BUT_EARLY" if len(profit_pool) else "NO_EDGE_YET"
    if recovery_ratio >= 1.0:
        status = "RECOVERY_POOL_COVERS_BLOCKED_DRAG"
    return ProfitEdgeSummary(
        total_strategies=int(len(report)),
        profit_candidates=int(len(profit_pool)),
        capital_drains=int(len(drains)),
        evidence_pool_pnl=round(evidence_pool_pnl, 2),
        blocked_drag_pnl=round(blocked_drag_pnl, 2),
        top_candidate=top_candidate,
        recovery_gap=round(recovery_gap, 2),
        recovery_ratio=round(recovery_ratio, 4),
        status=status,
        notes=notes,
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
