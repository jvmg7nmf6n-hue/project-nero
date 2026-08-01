"""Hypothesis Quality Gate for Project NERO.

Scores Strategy Doctor and Evolution Lab ideas before they receive more
paper-test budget. This is a research-control layer only: no real orders, no
silent strategy promotion, and no profit guarantee.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_VARIANTS_CSV = DEFAULT_REPORT_DIR / "strategy_evolution_variants.csv"
DEFAULT_RECOMMENDATIONS_CSV = DEFAULT_REPORT_DIR / "strategy_evolution_recommendations.csv"
DEFAULT_VERIFICATION_CSV = DEFAULT_REPORT_DIR / "strategy_verification_report.csv"
DEFAULT_QUARANTINE_CSV = DEFAULT_REPORT_DIR / "strategy_quarantine_report.csv"
DEFAULT_EDGE_JSON = DEFAULT_REPORT_DIR / "profit_edge_report.json"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "hypothesis_quality_gate.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "hypothesis_quality_gate.json"


@dataclass(frozen=True)
class HypothesisGateRow:
    parent: str
    proposed_variant: str
    mode: str
    family: str
    parent_action: str
    parent_trades: int
    parent_expectancy_r: float
    parent_profit_factor: float
    parent_net_pnl: float
    fixes_known_failure: bool
    evidence_quality: str
    overfit_risk: str
    gate_score: float
    decision: str
    reason: str


@dataclass(frozen=True)
class HypothesisGateSummary:
    total_hypotheses: int
    approved_shadow_tests: int
    repair_first: int
    collect_evidence: int
    rejected: int
    top_hypothesis: str
    average_score: float
    status: str
    notes: list[str]


def build_hypothesis_quality_gate(
    variants_csv: Path = DEFAULT_VARIANTS_CSV,
    recommendations_csv: Path = DEFAULT_RECOMMENDATIONS_CSV,
    verification_csv: Path = DEFAULT_VERIFICATION_CSV,
    quarantine_csv: Path = DEFAULT_QUARANTINE_CSV,
    edge_json: Path = DEFAULT_EDGE_JSON,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> tuple[pd.DataFrame, HypothesisGateSummary]:
    variants = _safe_read_csv(variants_csv)
    recommendations = _safe_read_csv(recommendations_csv)
    verification = _safe_read_csv(verification_csv)
    quarantine = _safe_read_csv(quarantine_csv)
    edge_lookup = _edge_lookup(edge_json)
    rows = [_gate_row(row, recommendations, verification, quarantine, edge_lookup) for row in variants.to_dict("records")]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["decision", "gate_score"], ascending=[True, False])
    summary = _summary(report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps({"summary": asdict(summary), "rows": report.to_dict("records")}, indent=2), encoding="utf-8")
    return report, summary


def _gate_row(
    row: dict[str, Any],
    recommendations: pd.DataFrame,
    verification: pd.DataFrame,
    quarantine: pd.DataFrame,
    edge_lookup: dict[str, dict[str, Any]],
) -> HypothesisGateRow:
    parent = str(row.get("Parent", "UNKNOWN") or "UNKNOWN")
    variant = str(row.get("Proposed Variant", f"{parent}_V2") or f"{parent}_V2")
    mode = str(row.get("Mode", "SHADOW_TEST_ONLY") or "SHADOW_TEST_ONLY")
    rec = _record_for(recommendations, "Candidate", parent)
    ver = _record_for(verification, "candidate_id", parent)
    quarantined = _record_for(quarantine, "candidate_id", parent)
    edge = edge_lookup.get(parent, {})
    family = str(rec.get("Family", ver.get("family", "Unknown")) or "Unknown")
    action = str(rec.get("Action", "UNKNOWN") or "UNKNOWN")
    trades = int(_num(rec.get("Trades", ver.get("total_trades")), 0))
    expectancy = _num(rec.get("Expectancy R", ver.get("expectancy_r")), 0.0)
    profit_factor = _num(rec.get("Profit Factor", ver.get("profit_factor")), 0.0)
    net_pnl = _num(ver.get("net_pnl", edge.get("net_pnl")), 0.0)
    fixes_known_failure = _fixes_known_failure(row, action, quarantined)
    evidence_quality = _evidence_quality(trades, expectancy, profit_factor, net_pnl)
    overfit_risk = _overfit_risk(row, trades, family)
    score = _score(action, trades, expectancy, profit_factor, net_pnl, fixes_known_failure, evidence_quality, overfit_risk)
    decision, reason = _decision(score, action, trades, fixes_known_failure, evidence_quality, overfit_risk)
    return HypothesisGateRow(
        parent=parent,
        proposed_variant=variant,
        mode=mode,
        family=family,
        parent_action=action,
        parent_trades=trades,
        parent_expectancy_r=round(expectancy, 4),
        parent_profit_factor=round(profit_factor, 4),
        parent_net_pnl=round(net_pnl, 2),
        fixes_known_failure=fixes_known_failure,
        evidence_quality=evidence_quality,
        overfit_risk=overfit_risk,
        gate_score=round(score, 2),
        decision=decision,
        reason=reason,
    )


def _fixes_known_failure(row: dict[str, Any], action: str, quarantined: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "")) for key in ["Hypothesis", "Proposed Changes"]).lower()
    repair_words = ["retest", "regime", "volatility", "reward", "rsi recovery", "trend support", "fees", "drawdown"]
    if action == "REWORK" or quarantined:
        return any(word in text for word in repair_words)
    return "promotion" in text or "live-data" in text or any(word in text for word in repair_words)


def _evidence_quality(trades: int, expectancy: float, profit_factor: float, net_pnl: float) -> str:
    if trades >= 100 and expectancy > 0 and profit_factor >= 1.25 and net_pnl > 0:
        return "PROMOTION_GRADE"
    if trades >= 30 and expectancy > 0 and profit_factor >= 1.10 and net_pnl > 0:
        return "ACTIONABLE_PAPER"
    if trades >= 10:
        return "EARLY_EVIDENCE"
    if trades > 0:
        return "TOO_SMALL"
    return "NO_TRADES"


def _overfit_risk(row: dict[str, Any], trades: int, family: str) -> str:
    text = " ".join(str(row.get(key, "")) for key in ["Hypothesis", "Proposed Changes"]).lower()
    knobs = sum(1 for token in ["rsi", "adx", "ma", "atr", "volatility", "target", "retest", "reward", "filter"] if token in text)
    if trades < 10 and knobs >= 4:
        return "HIGH"
    if trades < 30 and knobs >= 5:
        return "HIGH"
    if family in {"Momentum", "Mean Reversion"} and knobs >= 4:
        return "MEDIUM"
    return "LOW"


def _score(
    action: str,
    trades: int,
    expectancy: float,
    profit_factor: float,
    net_pnl: float,
    fixes_known_failure: bool,
    evidence_quality: str,
    overfit_risk: str,
) -> float:
    score = 35.0
    score += min(20.0, trades / 100.0 * 20.0)
    if action == "REWORK":
        score += 15.0
    elif action == "KEEP_TESTING":
        score += 10.0
    elif action == "PROMOTE_READY":
        score += 20.0
    if fixes_known_failure:
        score += 18.0
    score += max(-15.0, min(18.0, expectancy * 20.0))
    if profit_factor:
        score += max(-12.0, min(15.0, (profit_factor - 1.0) * 12.0))
    score += 8.0 if net_pnl > 0 else (-8.0 if net_pnl < 0 else 0.0)
    score += {"PROMOTION_GRADE": 15.0, "ACTIONABLE_PAPER": 10.0, "EARLY_EVIDENCE": 5.0, "TOO_SMALL": -5.0, "NO_TRADES": -10.0}[evidence_quality]
    score -= {"LOW": 0.0, "MEDIUM": 8.0, "HIGH": 18.0}[overfit_risk]
    if overfit_risk == "HIGH" and trades < 10:
        score = min(score, 45.0)
    elif overfit_risk == "HIGH" and trades < 30:
        score = min(score, 55.0)
    return max(0.0, min(100.0, score))


def _decision(score: float, action: str, trades: int, fixes_known_failure: bool, evidence_quality: str, overfit_risk: str) -> tuple[str, str]:
    if overfit_risk == "HIGH" and trades < 30:
        return "REJECT_WEAK_IDEA", "Too many filters for a small sample; likely curve-fit risk."
    if fixes_known_failure and trades >= 30 and score >= 55:
        return "APPROVE_SHADOW_TEST", "Hypothesis directly addresses a known failure and has enough logic to test."
    if action == "REWORK" and fixes_known_failure:
        return "REPAIR_FIRST", "Use as a repair hypothesis, but keep it paper-only until evidence improves."
    if evidence_quality in {"NO_TRADES", "TOO_SMALL"}:
        return "COLLECT_EVIDENCE", "Idea is not mature enough; collect more clean paper evidence first."
    return "WATCHLIST_ONLY", "Keep visible but do not spend major paper-test budget yet."


def _summary(report: pd.DataFrame) -> HypothesisGateSummary:
    if report.empty:
        return HypothesisGateSummary(0, 0, 0, 0, 0, "-", 0.0, "NO_HYPOTHESES", ["No variant hypotheses found. Run Strategy Evolution first."])
    counts = report["decision"].value_counts().to_dict()
    ranked = report[~report["decision"].isin(["REJECT_WEAK_IDEA"])]
    if ranked.empty:
        ranked = report
    top = str(ranked.sort_values("gate_score", ascending=False).iloc[0]["proposed_variant"])
    avg_score = float(pd.to_numeric(report["gate_score"], errors="coerce").fillna(0.0).mean())
    approved = int(counts.get("APPROVE_SHADOW_TEST", 0))
    status = "QUALITY_GATE_ACTIVE" if approved else "NO_APPROVED_HYPOTHESIS_YET"
    notes = [
        "Only approved hypotheses should receive fresh paper-test budget.",
        "High overfit risk or weak evidence must stay in watchlist/repair mode.",
        "No hypothesis is allowed to change production rules silently.",
    ]
    return HypothesisGateSummary(
        total_hypotheses=int(len(report)),
        approved_shadow_tests=approved,
        repair_first=int(counts.get("REPAIR_FIRST", 0)),
        collect_evidence=int(counts.get("COLLECT_EVIDENCE", 0)),
        rejected=int(counts.get("REJECT_WEAK_IDEA", 0)),
        top_hypothesis=top,
        average_score=round(avg_score, 2),
        status=status,
        notes=notes,
    )


def _edge_lookup(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return {str(row.get("candidate_id", "")): row for row in rows if row.get("candidate_id")}


def _record_for(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {}
    matches = frame[frame[column].astype(str) == value]
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





