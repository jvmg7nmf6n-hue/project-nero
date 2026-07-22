"""Strategy Verification Engine for Project NERO.

This module is the evidence judge for Strategy TEST Lab results. It does not
place trades and it does not change strategy parameters. It reads existing
paper-trade summaries, applies strict gates, and writes auditable verdicts.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_SUMMARY_PATH = DEFAULT_REPORT_DIR / "strategy_lab_summary.csv"
DEFAULT_VERIFICATION_CSV = DEFAULT_REPORT_DIR / "strategy_verification_report.csv"
DEFAULT_VERIFICATION_JSON = DEFAULT_REPORT_DIR / "strategy_verification_report.json"

MIN_PROMOTE_TRADES = 30
MIN_QUARANTINE_TRADES = 20
MIN_WATCHLIST_TRADES = 5
PROMOTE_MIN_PROFIT_FACTOR = 1.10
QUARANTINE_MAX_PROFIT_FACTOR = 0.90
MAX_PROMOTE_DRAWDOWN = 0.15


@dataclass(frozen=True)
class StrategyVerificationRow:
    candidate_id: str
    display_label: str
    bucket: str
    family: str
    interval: str
    asset_filter: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    max_drawdown: float
    net_pnl: float
    rating_score: float
    lab_rating: str
    verdict: str
    evidence_score: float
    primary_reason: str
    action: str
    sample_status: str
    data_status: str


def build_strategy_verification_report(
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    output_csv: Path = DEFAULT_VERIFICATION_CSV,
    output_json: Path = DEFAULT_VERIFICATION_JSON,
) -> pd.DataFrame:
    """Build and persist the strategy verification report."""
    summary = _safe_read_csv(summary_path)
    rows = [verify_strategy(row) for row in summary.to_dict("records")]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["evidence_score", "total_trades"], ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(report.to_dict("records"), indent=2), encoding="utf-8")
    return report


def verify_strategy(row: dict[str, Any]) -> StrategyVerificationRow:
    candidate_id = str(row.get("candidate_id", "UNKNOWN") or "UNKNOWN")
    display_label = str(row.get("display_label", candidate_id) or candidate_id)
    bucket = str(row.get("bucket", "UNKNOWN") or "UNKNOWN")
    family = str(row.get("family", "UNKNOWN") or "UNKNOWN")
    interval = str(row.get("interval", "-") or "-")
    asset_filter = str(row.get("asset_filter", "ALL") or "ALL")
    total_trades = int(_num(row.get("total_trades"), 0))
    win_rate = _num(row.get("win_rate"), 0.0)
    wins = int(round(total_trades * win_rate)) if total_trades else 0
    losses = max(0, total_trades - wins)
    expectancy_r = _num(row.get("expectancy_r"), 0.0)
    profit_factor = _num(row.get("profit_factor"), 0.0)
    max_drawdown = _num(row.get("max_drawdown"), 0.0)
    net_pnl = _num(row.get("net_pnl"), 0.0)
    rating_score = _num(row.get("rating_score"), 0.0)
    lab_rating = str(row.get("rating", "-") or "-")
    data_status = _data_status(row)
    sample_status = _sample_status(total_trades)
    verdict, reason, action = _verdict(
        total_trades=total_trades,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        net_pnl=net_pnl,
        data_status=data_status,
    )
    evidence_score = _cap_score_by_sample(_evidence_score(total_trades, expectancy_r, profit_factor, max_drawdown, net_pnl, data_status), total_trades)
    return StrategyVerificationRow(
        candidate_id=candidate_id,
        display_label=display_label,
        bucket=bucket,
        family=family,
        interval=interval,
        asset_filter=asset_filter,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 4),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 4),
        max_drawdown=round(max_drawdown, 4),
        net_pnl=round(net_pnl, 2),
        rating_score=round(rating_score, 2),
        lab_rating=lab_rating,
        verdict=verdict,
        evidence_score=round(evidence_score, 2),
        primary_reason=reason,
        action=action,
        sample_status=sample_status,
        data_status=data_status,
    )


def _verdict(*, total_trades: int, expectancy_r: float, profit_factor: float, max_drawdown: float, net_pnl: float, data_status: str) -> tuple[str, str, str]:
    if data_status != "OK":
        return "DATA_UNTRUSTED", "Data/source status is not clean enough for a strategy decision.", "Do not promote; refresh data and inspect feed quality."
    if total_trades < MIN_WATCHLIST_TRADES:
        return "INSUFFICIENT_SAMPLE", "Fewer than 5 closed paper trades.", "Keep collecting evidence only."
    if total_trades < MIN_PROMOTE_TRADES:
        if net_pnl > 0 and expectancy_r > 0 and profit_factor >= 1.0:
            return "WATCHLIST", "Positive but sample is below the 30-trade promotion gate.", "Continue forward testing; do not increase allocation."
        if total_trades >= MIN_QUARANTINE_TRADES and (net_pnl < 0 or expectancy_r < 0):
            return "QUARANTINE", "Negative evidence with at least 20 trades.", "Pause new entries until loss autopsy is reviewed."
        return "INSUFFICIENT_SAMPLE", "Sample is not large enough for a reliable verdict.", "Keep collecting evidence only."
    if net_pnl > 0 and expectancy_r > 0 and profit_factor >= PROMOTE_MIN_PROFIT_FACTOR and abs(max_drawdown) <= MAX_PROMOTE_DRAWDOWN:
        return "PROMOTE_PAPER", "Passed minimum paper-promotion gates after 30+ trades.", "Allow larger paper test, not real-money trading."
    if expectancy_r < 0 or net_pnl < 0 or (profit_factor and profit_factor < QUARANTINE_MAX_PROFIT_FACTOR):
        return "QUARANTINE", "30+ trade sample shows negative or weak risk-adjusted edge.", "Stop new paper entries and send to Strategy Doctor."
    return "WATCHLIST", "Mixed evidence after sample threshold.", "Keep testing with strict risk limits."


def _evidence_score(total_trades: int, expectancy_r: float, profit_factor: float, max_drawdown: float, net_pnl: float, data_status: str) -> float:
    if data_status != "OK":
        return 0.0
    sample_score = min(25.0, total_trades / MIN_PROMOTE_TRADES * 25.0)
    expectancy_score = max(-25.0, min(25.0, expectancy_r * 35.0))
    pf_score = max(-20.0, min(20.0, (profit_factor - 1.0) * 20.0)) if profit_factor else -15.0
    pnl_score = 10.0 if net_pnl > 0 else (-10.0 if net_pnl < 0 else 0.0)
    drawdown_penalty = min(20.0, abs(max_drawdown) * 100.0)
    return max(0.0, min(100.0, 50.0 + sample_score + expectancy_score + pf_score + pnl_score - drawdown_penalty))


def _cap_score_by_sample(score: float, total_trades: int) -> float:
    if total_trades < MIN_WATCHLIST_TRADES:
        return min(score, 45.0)
    if total_trades < MIN_QUARANTINE_TRADES:
        return min(score, 70.0)
    if total_trades < MIN_PROMOTE_TRADES:
        return min(score, 85.0)
    return score

def _sample_status(total_trades: int) -> str:
    if total_trades >= MIN_PROMOTE_TRADES:
        return "ADEQUATE"
    if total_trades >= MIN_QUARANTINE_TRADES:
        return "EARLY_BUT_ACTIONABLE"
    if total_trades >= MIN_WATCHLIST_TRADES:
        return "EARLY"
    return "TOO_SMALL"


def _data_status(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ["evidence_note", "rating", "asset_exclude", "bucket"]).lower()
    if "stale" in text or "sample data" in text or "fallback" in text:
        return "CHECK"
    return "OK"


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


