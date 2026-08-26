"""Sunflower-inspired profit discipline bridge for Project NERO.

This module does not import Sunflower directly. It ports the useful discipline:
profit must survive data-trust, sample-size, quarantine, and cost-aware gates
before NERO treats it as an actionable paper edge.
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
DEFAULT_PROFIT_EDGE_CSV = DEFAULT_REPORT_DIR / "profit_edge_report.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "sunflower_profit_bridge.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "sunflower_profit_bridge.json"

MIN_DISCIPLINED_SAMPLE = 30
MIN_EARLY_SAMPLE = 5
MIN_PROFIT_FACTOR = 1.10
MAX_DRAWDOWN_ABS = 0.15


@dataclass(frozen=True)
class SunflowerProfitRow:
    candidate_id: str
    display_label: str
    family: str
    interval: str
    asset_filter: str
    total_trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    max_drawdown: float
    net_pnl: float
    verification_verdict: str
    profit_edge_role: str
    data_status: str
    sunflower_gate: str
    discipline_score: float
    decision: str
    reason: str


@dataclass(frozen=True)
class SunflowerProfitSummary:
    strategies_reviewed: int
    disciplined_profit_candidates: int
    early_profit_watchlist: int
    capital_drains_blocked: int
    untrusted_data: int
    positive_pool_pnl: float
    blocked_pool_pnl: float
    top_candidate: str
    status: str
    notes: list[str]


def build_sunflower_profit_bridge_report(
    verification_csv: Path = DEFAULT_VERIFICATION_CSV,
    profit_edge_csv: Path = DEFAULT_PROFIT_EDGE_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> tuple[pd.DataFrame, SunflowerProfitSummary]:
    """Build a stricter profit report using Sunflower-style evidence gates."""
    verification = _safe_read_csv(verification_csv)
    profit_edge = _safe_read_csv(profit_edge_csv)
    rows = [_row(record, profit_edge) for record in verification.to_dict("records")]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["discipline_score", "net_pnl", "total_trades"], ascending=False)
    summary = _summary(report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps({"summary": asdict(summary), "rows": report.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    return report, summary


def _row(record: dict[str, Any], profit_edge: pd.DataFrame) -> SunflowerProfitRow:
    candidate_id = str(record.get("candidate_id", "UNKNOWN") or "UNKNOWN")
    display_label = str(record.get("display_label", candidate_id) or candidate_id)
    family = str(record.get("family", "-") or "-")
    interval = str(record.get("interval", "-") or "-")
    asset_filter = str(record.get("asset_filter", "-") or "-")
    total_trades = int(_num(record.get("total_trades"), 0))
    win_rate = _num(record.get("win_rate"), 0.0)
    expectancy_r = _num(record.get("expectancy_r"), 0.0)
    profit_factor = _num(record.get("profit_factor"), 0.0)
    max_drawdown = _num(record.get("max_drawdown"), 0.0)
    net_pnl = _num(record.get("net_pnl"), 0.0)
    verification_verdict = str(record.get("verdict", "UNKNOWN") or "UNKNOWN")
    data_status = str(record.get("data_status", "OK") or "OK")
    profit_edge_role = _profit_edge_role(candidate_id, profit_edge)
    sunflower_gate, decision, reason = _gate(
        total_trades=total_trades,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        net_pnl=net_pnl,
        verification_verdict=verification_verdict,
        profit_edge_role=profit_edge_role,
        data_status=data_status,
    )
    score = _discipline_score(
        total_trades=total_trades,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        net_pnl=net_pnl,
        sunflower_gate=sunflower_gate,
    )
    return SunflowerProfitRow(
        candidate_id=candidate_id,
        display_label=display_label,
        family=family,
        interval=interval,
        asset_filter=asset_filter,
        total_trades=total_trades,
        win_rate=round(win_rate, 4),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 4),
        max_drawdown=round(max_drawdown, 4),
        net_pnl=round(net_pnl, 2),
        verification_verdict=verification_verdict,
        profit_edge_role=profit_edge_role,
        data_status=data_status,
        sunflower_gate=sunflower_gate,
        discipline_score=round(score, 2),
        decision=decision,
        reason=reason,
    )


def _gate(
    *,
    total_trades: int,
    expectancy_r: float,
    profit_factor: float,
    max_drawdown: float,
    net_pnl: float,
    verification_verdict: str,
    profit_edge_role: str,
    data_status: str,
) -> tuple[str, str, str]:
    if data_status != "OK":
        return "DATA_NOT_TRUSTED", "NO_PROFIT_CLAIM", "Data status is not clean enough for profit evidence."
    if verification_verdict == "QUARANTINE" or profit_edge_role == "CAPITAL_DRAIN":
        return "CAPITAL_DRAIN_BLOCKED", "BLOCK_NEW_ENTRIES", "Strategy is blocked before any profit allocation."
    if total_trades < MIN_EARLY_SAMPLE:
        return "TOO_SMALL", "COLLECT_ONLY", "Fewer than 5 closed trades; profit reading is noise."
    has_positive_edge = net_pnl > 0 and expectancy_r > 0 and profit_factor >= 1.0
    passes_disciplined_gate = (
        total_trades >= MIN_DISCIPLINED_SAMPLE
        and net_pnl > 0
        and expectancy_r > 0
        and profit_factor >= MIN_PROFIT_FACTOR
        and abs(max_drawdown) <= MAX_DRAWDOWN_ABS
    )
    if passes_disciplined_gate:
        return "DISCIPLINED_PROFIT_CANDIDATE", "FOCUS_PAPER_CAPITAL", "Profit survived sample, PF, expectancy, drawdown, and quarantine gates."
    if has_positive_edge:
        return "EARLY_PROFIT_WATCHLIST", "KEEP_FORWARD_TESTING", "Positive profit exists, but Sunflower gate needs 30+ trades and PF above 1.10."
    return "NO_EDGE_YET", "DO_NOT_ALLOCATE", "No positive cost-aware edge is proven yet."


def _discipline_score(
    *,
    total_trades: int,
    expectancy_r: float,
    profit_factor: float,
    max_drawdown: float,
    net_pnl: float,
    sunflower_gate: str,
) -> float:
    if sunflower_gate in {"CAPITAL_DRAIN_BLOCKED", "DATA_NOT_TRUSTED"}:
        return 0.0
    sample = min(25.0, total_trades / MIN_DISCIPLINED_SAMPLE * 25.0)
    expectancy = max(-25.0, min(25.0, expectancy_r * 35.0))
    pf = max(-20.0, min(25.0, (profit_factor - 1.0) * 20.0)) if profit_factor else -15.0
    pnl = 10.0 if net_pnl > 0 else (-10.0 if net_pnl < 0 else 0.0)
    drawdown_penalty = min(20.0, abs(max_drawdown) * 100.0)
    score = 50.0 + sample + expectancy + pf + pnl - drawdown_penalty
    if sunflower_gate == "TOO_SMALL":
        score = min(score, 45.0)
    if sunflower_gate == "EARLY_PROFIT_WATCHLIST":
        score = min(score, 80.0)
    if sunflower_gate == "NO_EDGE_YET":
        score = min(score, 35.0)
    return max(0.0, min(100.0, score))


def _summary(report: pd.DataFrame) -> SunflowerProfitSummary:
    if report.empty:
        return SunflowerProfitSummary(0, 0, 0, 0, 0, 0.0, 0.0, "-", "NO_DATA", ["No verification report is available yet."])
    gate = report["sunflower_gate"].astype(str)
    disciplined = report[gate.eq("DISCIPLINED_PROFIT_CANDIDATE")]
    early = report[gate.eq("EARLY_PROFIT_WATCHLIST")]
    drains = report[gate.eq("CAPITAL_DRAIN_BLOCKED")]
    untrusted = report[gate.eq("DATA_NOT_TRUSTED")]
    positive_pool = pd.concat([disciplined, early], ignore_index=True)
    top_candidate = "-"
    if not positive_pool.empty:
        top_candidate = str(positive_pool.sort_values(["discipline_score", "net_pnl"], ascending=False).iloc[0]["display_label"])
    status = "NO_EDGE_YET"
    if not disciplined.empty:
        status = "DISCIPLINED_EDGE_FOUND"
    elif not early.empty:
        status = "EARLY_EDGE_FOUND"
    notes = [
        "Sunflower merge adds strict profit discipline: clean data, sample size, cost-aware edge, drawdown, and quarantine gates.",
        "Positive P/L below 30 trades is watchlist evidence only, not a promoted strategy.",
    ]
    return SunflowerProfitSummary(
        strategies_reviewed=int(len(report)),
        disciplined_profit_candidates=int(len(disciplined)),
        early_profit_watchlist=int(len(early)),
        capital_drains_blocked=int(len(drains)),
        untrusted_data=int(len(untrusted)),
        positive_pool_pnl=round(float(pd.to_numeric(positive_pool.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()), 2),
        blocked_pool_pnl=round(float(pd.to_numeric(drains.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()), 2),
        top_candidate=top_candidate,
        status=status,
        notes=notes,
    )


def _profit_edge_role(candidate_id: str, profit_edge: pd.DataFrame) -> str:
    if profit_edge.empty or "candidate_id" not in profit_edge or "role" not in profit_edge:
        return "UNKNOWN"
    matched = profit_edge[profit_edge["candidate_id"].astype(str).eq(candidate_id)]
    if matched.empty:
        return "UNKNOWN"
    return str(matched.iloc[0].get("role", "UNKNOWN") or "UNKNOWN")


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
