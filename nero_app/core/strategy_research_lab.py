"""Strategy Research Lab for Project Nero.

This module is NERO's controlled self-improvement layer. It reads paper-trade
results and rejected setup logs, then proposes versioned strategy candidates for
human-approved forward testing. It does not change live strategy parameters and
it does not claim a candidate will be profitable.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEAN_REVERSION_REPORT_PATH = PROJECT_ROOT / "reports" / "mean_reversion_report.csv"
DEFAULT_CLOSED_TRADES_PATH = PROJECT_ROOT / "nero_app" / "data" / "mean_reversion" / "trades" / "closed_trades.csv"
DEFAULT_EVALUATIONS_PATH = PROJECT_ROOT / "nero_app" / "data" / "mean_reversion" / "trades" / "evaluations.csv"


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    family: str
    title: str
    hypothesis: str
    proposed_changes: list[str]
    evidence: list[str]
    risks: list[str]
    priority_score: float
    status: str = "RESEARCH_ONLY"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StrategyResearchReport:
    lab_score: float
    label: str
    sample_status: str
    current_edge: str
    top_blockers: list[dict[str, Any]]
    candidates: list[StrategyCandidate]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def candidate_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "Candidate": item.candidate_id,
                "Family": item.family,
                "Priority": f"{item.priority_score:.0f}/100",
                "Status": item.status,
                "Hypothesis": item.hypothesis,
                "Proposed Changes": "; ".join(item.proposed_changes),
                "Main Risks": "; ".join(item.risks),
            }
            for item in self.candidates
        ]


def build_strategy_research_report(
    mean_reversion_report_path: Path = DEFAULT_MEAN_REVERSION_REPORT_PATH,
    closed_trades_path: Path = DEFAULT_CLOSED_TRADES_PATH,
    evaluations_path: Path = DEFAULT_EVALUATIONS_PATH,
) -> StrategyResearchReport:
    mean_report = _read_csv(mean_reversion_report_path)
    closed_trades = _read_csv(closed_trades_path)
    evaluations = _read_csv(evaluations_path)

    combined = _combined_row(mean_report)
    total_trades = _int_value(combined, "total_trades") if combined else len(closed_trades)
    expectancy = _float_value(combined, "expectancy_r") if combined else _mean_numeric(closed_trades, "r_multiple")
    profit_factor = _float_value(combined, "profit_factor") if combined else _profit_factor(closed_trades)
    win_rate = _float_value(combined, "win_rate") if combined else _win_rate(closed_trades)
    top_blockers = _top_blockers(mean_report, evaluations, limit=5)

    sample_status = "INSUFFICIENT_SAMPLE" if total_trades < 20 else "ENOUGH_FOR_FIRST_READ"
    if total_trades == 0:
        current_edge = "UNPROVEN_NO_TRADES"
    elif expectancy > 0 and profit_factor > 1:
        current_edge = "PROMISING_BUT_EARLY" if total_trades < 20 else "POSITIVE_EDGE_CANDIDATE"
    else:
        current_edge = "WEAK_OR_NEGATIVE_EDGE"

    candidates = _build_candidates(total_trades, expectancy, profit_factor, win_rate, top_blockers)
    lab_score = _lab_score(total_trades, expectancy, profit_factor, len(candidates))
    label = _lab_label(lab_score, sample_status)
    notes = _notes(sample_status, current_edge, total_trades, expectancy, profit_factor)

    return StrategyResearchReport(
        lab_score=lab_score,
        label=label,
        sample_status=sample_status,
        current_edge=current_edge,
        top_blockers=top_blockers,
        candidates=candidates,
        notes=notes,
    )


def _build_candidates(
    total_trades: int,
    expectancy: float,
    profit_factor: float,
    win_rate: float,
    top_blockers: list[dict[str, Any]],
) -> list[StrategyCandidate]:
    blocker_names = {str(item.get("reason", "")): int(item.get("count", 0)) for item in top_blockers}
    total_blockers = max(1, sum(blocker_names.values()))
    candidates: list[StrategyCandidate] = []

    close_not_bb = blocker_names.get("CLOSE_NOT_BELOW_LOWER_BB", 0)
    rsi_not_low = blocker_names.get("RSI_NOT_BELOW_35", 0)
    target_not_above = blocker_names.get("TARGET_NOT_ABOVE_ENTRY", 0)
    below_ma200 = blocker_names.get("CLOSE_NOT_ABOVE_MA200", 0)

    if rsi_not_low / total_blockers > 0.18 or close_not_bb / total_blockers > 0.25:
        candidates.append(
            StrategyCandidate(
                candidate_id="MR_RELAXED_PULLBACK_V1",
                family="Mean Reversion",
                title="Relaxed pullback candidate",
                hypothesis="Current rules may be too strict and are rejecting many near-pullback setups before sample size can grow.",
                proposed_changes=["test RSI entry below 40", "allow close within 0.25 ATR of lower Bollinger Band", "keep MA200 filter", "paper-test only"],
                evidence=[f"RSI blocker count: {rsi_not_low}", f"Lower-BB blocker count: {close_not_bb}"],
                risks=["More trades can mean more low-quality entries", "Must compare expectancy, not just trade count"],
                priority_score=_candidate_score(62, total_trades, expectancy, rsi_not_low + close_not_bb, total_blockers),
            )
        )

    if expectancy <= 0 or profit_factor < 1.2 or win_rate < 0.45:
        candidates.append(
            StrategyCandidate(
                candidate_id="MR_DEEP_VALUE_V1",
                family="Mean Reversion",
                title="Deeper value candidate",
                hypothesis="If current mean-reversion quality weakens, a deeper oversold rule may reduce false entries.",
                proposed_changes=["test RSI entry below 30", "require close below lower Bollinger Band", "target frozen MA20", "paper-test only"],
                evidence=[f"Current expectancy: {expectancy:.2f}R", f"Profit factor: {profit_factor:.2f}", f"Win rate: {win_rate:.0%}"],
                risks=["Fewer trades and slower learning", "Can miss shallow reversals"],
                priority_score=_candidate_score(58, total_trades, -expectancy, rsi_not_low, total_blockers),
            )
        )

    if below_ma200 / total_blockers > 0.12:
        candidates.append(
            StrategyCandidate(
                candidate_id="MR_REGIME_FILTER_V1",
                family="Mean Reversion",
                title="Regime-filtered mean reversion",
                hypothesis="Many rejections happen when price is below MA200; NERO should separate bull-regime dips from bear-regime traps.",
                proposed_changes=["keep long-only trades above MA200", "create separate watch-only bear-regime bucket", "do not relax MA200 without separate testing"],
                evidence=[f"MA200 blocker count: {below_ma200}"],
                risks=["May reduce trade frequency", "Does not solve entries during sideways markets"],
                priority_score=_candidate_score(65, total_trades, expectancy, below_ma200, total_blockers),
            )
        )

    if target_not_above / total_blockers > 0.10:
        candidates.append(
            StrategyCandidate(
                candidate_id="MR_TARGET_1R_V1",
                family="Exit Logic",
                title="Fixed reward target candidate",
                hypothesis="MA20 target is often too close or invalid; a fixed 1R/1.25R target may create cleaner reward-risk tests.",
                proposed_changes=["freeze target at entry", "compare MA20 target vs fixed 1R and 1.25R", "score by net R after fees"],
                evidence=[f"Target-not-above-entry blocker count: {target_not_above}"],
                risks=["Fixed target may ignore actual market structure", "Needs separate slippage/fee comparison"],
                priority_score=_candidate_score(60, total_trades, expectancy, target_not_above, total_blockers),
            )
        )

    candidates.append(
        StrategyCandidate(
            candidate_id="BREAKOUT_MOMENTUM_V1",
            family="Momentum",
            title="Alternate non-mean-reversion candidate",
            hypothesis="If mean reversion remains low-frequency, NERO should forward-test a separate breakout family rather than forcing mean reversion.",
            proposed_changes=["test long breakout above 20-bar high", "require quant consensus above 65", "require volatility not extreme", "paper-test separately"],
            evidence=[f"Closed trades so far: {total_trades}", "Mean reversion can stay inactive for long periods"],
            risks=["Breakouts can whipsaw", "Must not mix results with mean-reversion ledger"],
            priority_score=55.0 if total_trades < 20 else 45.0,
        )
    )

    return sorted(candidates, key=lambda item: item.priority_score, reverse=True)[:5]


def _candidate_score(base: float, total_trades: int, edge: float, blocker_count: int, total_blockers: int) -> float:
    score = base
    score += min(12.0, blocker_count / max(1, total_blockers) * 20.0)
    score += 5.0 if total_trades < 20 else 0.0
    score += max(-8.0, min(8.0, edge * 6.0))
    return round(max(0.0, min(100.0, score)), 2)


def _lab_score(total_trades: int, expectancy: float, profit_factor: float, candidate_count: int) -> float:
    score = 45.0
    score += min(20.0, total_trades)
    score += max(-15.0, min(20.0, expectancy * 12.0))
    score += max(-10.0, min(15.0, (profit_factor - 1.0) * 10.0)) if profit_factor else -5.0
    score += min(10.0, candidate_count * 2.0)
    return round(max(0.0, min(100.0, score)), 2)


def _lab_label(score: float, sample_status: str) -> str:
    if sample_status == "INSUFFICIENT_SAMPLE":
        return "RESEARCH_MODE_EARLY"
    if score >= 75:
        return "RESEARCH_PIPELINE_HEALTHY"
    if score >= 55:
        return "RESEARCH_PIPELINE_ACTIVE"
    return "RESEARCH_PIPELINE_WEAK"


def _notes(sample_status: str, current_edge: str, total_trades: int, expectancy: float, profit_factor: float) -> list[str]:
    notes = [f"Current edge state: {current_edge}."]
    notes.append(f"Evidence base: {total_trades} closed paper trade(s), expectancy {expectancy:.2f}R, profit factor {profit_factor:.2f}.")
    if sample_status == "INSUFFICIENT_SAMPLE":
        notes.append("Do not promote any candidate yet; collect at least 20-30 closed trades per strategy family.")
    notes.append("Candidates are research-only. NERO must not auto-change parameters without manual version approval.")
    return notes


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _combined_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty or "asset" not in frame.columns:
        return None
    rows = frame[frame["asset"].astype(str).str.upper() == "COMBINED"]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _float_value(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _int_value(row: dict[str, Any] | None, key: str) -> int:
    if not row:
        return 0
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty or "net_pnl" not in frame.columns:
        return 0.0
    pnl = pd.to_numeric(frame["net_pnl"], errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses == 0:
        return gains if gains else 0.0
    return gains / losses


def _win_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "net_pnl" not in frame.columns:
        return 0.0
    pnl = pd.to_numeric(frame["net_pnl"], errors="coerce").dropna()
    return float((pnl > 0).mean()) if not pnl.empty else 0.0


def _top_blockers(report: pd.DataFrame, evaluations: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    if not report.empty and "rejected_setup_counts" in report.columns:
        source = report
        if "asset" in report.columns:
            combined = report[report["asset"].astype(str).str.upper() == "COMBINED"]
            if not combined.empty:
                source = combined
        for raw in source["rejected_setup_counts"].fillna(""):
            try:
                parsed = json.loads(str(raw)) if str(raw).strip() else {}
            except json.JSONDecodeError:
                parsed = {}
            for key, value in parsed.items():
                counts[str(key)] = counts.get(str(key), 0) + int(value)
    if not counts and not evaluations.empty and "rejection_reasons" in evaluations.columns:
        for raw in evaluations["rejection_reasons"].fillna(""):
            for reason in str(raw).split("|"):
                if reason:
                    counts[reason] = counts.get(reason, 0) + 1
    total = max(1, sum(counts.values()))
    rows = [
        {"reason": key, "count": value, "share": round(value / total, 4)}
        for key, value in counts.items()
    ]
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:limit]


