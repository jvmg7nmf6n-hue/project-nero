"""Strategy Intelligence Engine for Project NERO.

This module turns observed failures into constrained, LLM-ready strategy
proposals. It does not place trades, mutate live strategy rules, or claim that
any proposal is profitable. The goal is to give Strategy Doctor a sharper
failure-learning brain while preserving NERO's anti-overfitting culture.
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
DEFAULT_REPAIR_LAB_CSV = DEFAULT_REPORT_DIR / "strategy_repair_lab_attempts.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "strategy_intelligence_engine.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "strategy_intelligence_engine.json"

ENGINE_VERSION = "strategy_intelligence_engine_v1"
MIN_REPAIR_SAMPLE = 20
MAX_ALLOWED_RULE_CHANGES = 2


@dataclass(frozen=True)
class StrategyBrainRow:
    parent_strategy: str
    parent_label: str
    family: str
    failure_class: str
    failure_evidence: str
    parent_trades: int
    parent_net_pnl: float
    parent_expectancy_r: float
    parent_profit_factor: float
    proposed_strategy_id: str
    proposed_label: str
    core_hypothesis: str
    rule_change_count: int
    changed_rules: str
    unchanged_rules: str
    asset_scope: str
    timeframe_scope: str
    fresh_data_plan: str
    random_baseline_plan: str
    anti_overfit_guard: str
    llm_prompt: str
    proposal_contract: str
    intelligence_score: float
    decision: str
    next_action: str


@dataclass(frozen=True)
class StrategyBrainSummary:
    engine_version: str
    failures_read: int
    proposals: int
    test_lab_ready: int
    research_queue: int
    blocked: int
    top_proposal: str
    biggest_failure: str
    status: str
    notes: list[str]


def build_strategy_intelligence_report(
    verification_csv: Path = DEFAULT_VERIFICATION_CSV,
    quarantine_csv: Path = DEFAULT_QUARANTINE_CSV,
    repair_lab_csv: Path = DEFAULT_REPAIR_LAB_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> tuple[pd.DataFrame, StrategyBrainSummary]:
    """Build and persist failure-driven strategy design proposals."""
    verification = _safe_read_csv(verification_csv)
    quarantine = _safe_read_csv(quarantine_csv)
    repair_lab = _safe_read_csv(repair_lab_csv)

    failure_records = _failure_records(verification, quarantine, repair_lab)
    rows = [_brain_row(record) for record in failure_records]
    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["decision", "intelligence_score", "parent_net_pnl"], ascending=[True, False, True])

    summary = _summary(report, failure_records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps({"summary": asdict(summary), "rows": report.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    return report, summary


def _failure_records(verification: pd.DataFrame, quarantine: pd.DataFrame, repair_lab: pd.DataFrame) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    for row in verification.to_dict("records"):
        label = str(row.get("display_label", row.get("candidate_id", "UNKNOWN")) or "UNKNOWN")
        candidate = str(row.get("candidate_id", label) or label)
        verdict = str(row.get("verdict", "") or "")
        trades = int(_num(row.get("total_trades"), 0))
        net = _num(row.get("net_pnl"), 0.0)
        expectancy = _num(row.get("expectancy_r"), 0.0)
        if verdict == "QUARANTINE" or (trades > 0 and net < 0) or (trades >= MIN_REPAIR_SAMPLE and expectancy <= 0):
            records[candidate or label] = {
                "parent_strategy": candidate or label,
                "parent_label": label,
                "family": str(row.get("family", "UNKNOWN") or "UNKNOWN"),
                "trades": trades,
                "net_pnl": net,
                "expectancy_r": expectancy,
                "profit_factor": _num(row.get("profit_factor"), 0.0),
                "source": "verification",
            }

    for row in quarantine.to_dict("records"):
        label = str(row.get("display_label", row.get("candidate_id", "UNKNOWN")) or "UNKNOWN")
        candidate = str(row.get("candidate_id", label) or label)
        current = records.get(candidate or label, {})
        records[candidate or label] = {
            **current,
            "parent_strategy": candidate or label,
            "parent_label": label,
            "family": current.get("family", str(row.get("family", "UNKNOWN") or "UNKNOWN")),
            "trades": int(_num(row.get("total_trades"), current.get("trades", 0))),
            "net_pnl": _num(row.get("net_pnl"), current.get("net_pnl", 0.0)),
            "expectancy_r": _num(row.get("expectancy_r"), current.get("expectancy_r", 0.0)),
            "profit_factor": _num(row.get("profit_factor"), current.get("profit_factor", 0.0)),
            "source": "quarantine",
            "quarantine_reason": str(row.get("reason", "") or ""),
        }

    if not repair_lab.empty:
        design_rows = repair_lab[repair_lab.get("promotion_decision", pd.Series(dtype=str)).astype(str).isin(["DESIGN_REQUIRED", "REJECT_REPAIR", "REJECT_REPAIR_NOT_BETTER_THAN_PARENT", "REJECT_REPAIR_NOT_BETTER_THAN_RANDOM"])]
        for row in design_rows.to_dict("records"):
            parent = str(row.get("parent_strategy", "UNKNOWN") or "UNKNOWN")
            current = records.get(parent, {})
            records[parent] = {
                **current,
                "parent_strategy": parent,
                "parent_label": str(row.get("parent_label", parent) or parent),
                "family": current.get("family", "REPAIR_LINEAGE"),
                "trades": int(_num(row.get("parent_trades"), current.get("trades", 0))),
                "net_pnl": _num(row.get("parent_net_pnl"), current.get("net_pnl", 0.0)),
                "expectancy_r": _num(row.get("parent_expectancy_r"), current.get("expectancy_r", 0.0)),
                "profit_factor": _num(row.get("parent_profit_factor"), current.get("profit_factor", 0.0)),
                "source": "repair_lab",
                "repair_status": str(row.get("promotion_decision", "") or ""),
                "repair_failure_code": str(row.get("failure_reason_code", "") or ""),
            }

    return sorted(records.values(), key=lambda item: float(item.get("net_pnl", 0.0)))


def _brain_row(record: dict[str, Any]) -> StrategyBrainRow:
    parent = str(record.get("parent_strategy", "UNKNOWN") or "UNKNOWN")
    label = str(record.get("parent_label", parent) or parent)
    family = _family(record, parent, label)
    failure_class, evidence = _failure_class(record, parent, label, family)
    proposal = _proposal_for_failure(parent, label, family, failure_class)
    score = _intelligence_score(record, failure_class, proposal)
    decision = _decision(score, int(_num(record.get("trades"), 0)), proposal["rule_change_count"])
    contract = _proposal_contract(record, proposal, failure_class)
    return StrategyBrainRow(
        parent_strategy=parent,
        parent_label=label,
        family=family,
        failure_class=failure_class,
        failure_evidence=evidence,
        parent_trades=int(_num(record.get("trades"), 0)),
        parent_net_pnl=round(_num(record.get("net_pnl"), 0.0), 2),
        parent_expectancy_r=round(_num(record.get("expectancy_r"), 0.0), 4),
        parent_profit_factor=round(_num(record.get("profit_factor"), 0.0), 4),
        proposed_strategy_id=proposal["proposal_id"],
        proposed_label=proposal["label"],
        core_hypothesis=proposal["hypothesis"],
        rule_change_count=proposal["rule_change_count"],
        changed_rules="; ".join(proposal["changed_rules"]),
        unchanged_rules="; ".join(proposal["unchanged_rules"]),
        asset_scope=proposal["asset_scope"],
        timeframe_scope=proposal["timeframe_scope"],
        fresh_data_plan="Forward paper from next workflow run plus any non-overlapping historical window explicitly recorded.",
        random_baseline_plan="Same asset/timeframe/regime pool random-entry baseline required before promotion.",
        anti_overfit_guard="Max two changed rules, no same-window retest promotion, no parameter search after seeing results, paper-only.",
        llm_prompt=_llm_prompt(record, proposal, failure_class),
        proposal_contract=json.dumps(contract, sort_keys=True),
        intelligence_score=round(score, 2),
        decision=decision,
        next_action=_next_action(decision, proposal),
    )


def _family(record: dict[str, Any], parent: str, label: str) -> str:
    raw = str(record.get("family", "") or "").upper()
    text = f"{parent} {label} {raw}".upper()
    if "BREAKOUT" in text or "MOMENTUM" in text:
        return "BREAKOUT"
    if "MR" in text or "MEAN" in text or "RMR" in text:
        return "MEAN_REVERSION"
    if "SHORT" in text:
        return "SHORT_MOMENTUM"
    if "OIL" in text or "GOLD" in text or "SILVER" in text:
        return "COMMODITY"
    return raw.title() if raw else "UNKNOWN"


def _failure_class(record: dict[str, Any], parent: str, label: str, family: str) -> tuple[str, str]:
    text = f"{parent} {label} {record.get('quarantine_reason', '')} {record.get('repair_failure_code', '')}".upper()
    trades = int(_num(record.get("trades"), 0))
    net = _num(record.get("net_pnl"), 0.0)
    expectancy = _num(record.get("expectancy_r"), 0.0)
    profit_factor = _num(record.get("profit_factor"), 0.0)
    if net > 0 and expectancy <= 0:
        return "ACCOUNTING_PROFIT_NOT_R_EDGE", "Dollar P/L is positive but R-expectancy is not, so risk/reward quality is weak."
    if trades >= 30 and net <= -1000:
        return "CAPITAL_DRAIN", "Large negative P/L with enough trades to stop further entries."
    if family == "BREAKOUT":
        return "BREAKOUT_FAKEOUT", "Breakout family is losing; likely entering first breaks without enough trend/retest quality."
    if family == "MEAN_REVERSION":
        return "MEAN_REVERSION_TOO_EARLY", "Mean reversion family is losing; likely catching falling candles before exhaustion is confirmed."
    if "OIL" in text or family == "COMMODITY":
        return "ASSET_MICROSTRUCTURE_MISMATCH", "Commodity-style instruments need session/gap/volatility handling separate from crypto rules."
    if expectancy < 0 or profit_factor < 1:
        return "NEGATIVE_EXPECTANCY", "Expectancy or profit factor is below break-even on observed paper data."
    return "NEEDS_DEEPER_DIAGNOSIS", "Failure is real but not specific enough; Strategy Doctor should demand more diagnostics before coding."


def _proposal_for_failure(parent: str, label: str, family: str, failure_class: str) -> dict[str, Any]:
    base = _safe_id(label or parent)
    if failure_class == "BREAKOUT_FAKEOUT" or (failure_class == "CAPITAL_DRAIN" and family == "BREAKOUT"):
        return {
            "proposal_id": f"SIE_{base}_RETEST_QUALITY_V1",
            "label": f"SIE_{base}_RETEST",
            "hypothesis": "Breakout edge may improve if NERO waits for a retest close and blocks high ATR shock candles.",
            "changed_rules": ["require retest confirmation after breakout", "block entries when ATR percentage is above the strategy ceiling"],
            "unchanged_rules": ["long-only breakout family", "existing fee and paper risk model", "fixed R accounting"],
            "asset_scope": "same assets as parent until evidence shows asset-specific failure",
            "timeframe_scope": "same timeframe as parent; no timeframe expansion before first 30 fresh trades",
            "rule_change_count": 2,
        }
    if failure_class in {"MEAN_REVERSION_TOO_EARLY", "CAPITAL_DRAIN"} and family == "MEAN_REVERSION":
        return {
            "proposal_id": f"SIE_{base}_EXHAUSTION_CONFIRM_V1",
            "label": f"SIE_{base}_CONFIRM",
            "hypothesis": "Mean-reversion losses may shrink if entry waits for exhaustion recovery instead of buying the first oversold print.",
            "changed_rules": ["require RSI or close recovery confirmation after oversold touch", "reject entries when target distance is below realistic fee/slippage threshold"],
            "unchanged_rules": ["mean-reversion family", "paper-only sizing", "existing stop model"],
            "asset_scope": "parent asset universe only; no expansion before 30 fresh trades",
            "timeframe_scope": "parent timeframe only",
            "rule_change_count": 2,
        }
    if failure_class == "ACCOUNTING_PROFIT_NOT_R_EDGE":
        return {
            "proposal_id": f"SIE_{base}_R_QUALITY_V1",
            "label": f"SIE_{base}_R_QUALITY",
            "hypothesis": "Dollar-positive but R-negative systems need stricter reward quality before entry.",
            "changed_rules": ["minimum planned reward must be at least 1.25R after estimated costs", "exit target must be frozen at entry for audit clarity"],
            "unchanged_rules": ["same entry family", "same asset universe", "paper-only execution"],
            "asset_scope": "same assets as current repair",
            "timeframe_scope": "same timeframe as current repair",
            "rule_change_count": 2,
        }
    if failure_class == "ASSET_MICROSTRUCTURE_MISMATCH":
        return {
            "proposal_id": f"SIE_{base}_SESSION_FILTER_V1",
            "label": f"SIE_{base}_SESSION",
            "hypothesis": "Non-crypto assets need session-aware entries and gap protection before technical signals are trusted.",
            "changed_rules": ["trade only during liquid session overlap", "skip entries after large overnight/session gaps"],
            "unchanged_rules": ["paper-only status", "existing cost-aware P/L accounting", "same parent signal family"],
            "asset_scope": "commodity/FX/equity asset class only",
            "timeframe_scope": "same timeframe; daily/weekly expansion blocked until clean intraday evidence exists",
            "rule_change_count": 2,
        }
    return {
        "proposal_id": f"SIE_{base}_DIAGNOSIS_FIRST_V1",
        "label": f"SIE_{base}_DIAGNOSIS",
        "hypothesis": "The failure class is too vague; first add diagnostics before inventing a new strategy.",
        "changed_rules": ["add per-trade failure tags", "compare parent entries against same-regime random-entry baseline"],
        "unchanged_rules": ["no new entries from this proposal", "paper-only audit", "same data source requirements"],
        "asset_scope": "diagnostic only",
        "timeframe_scope": "diagnostic only",
        "rule_change_count": 2,
    }


def _intelligence_score(record: dict[str, Any], failure_class: str, proposal: dict[str, Any]) -> float:
    trades = int(_num(record.get("trades"), 0))
    net = _num(record.get("net_pnl"), 0.0)
    expectancy = _num(record.get("expectancy_r"), 0.0)
    score = 40.0
    score += min(20.0, trades / 50.0 * 20.0)
    score += min(20.0, abs(min(0.0, net)) / 1500.0 * 20.0)
    score += 12.0 if failure_class in {"BREAKOUT_FAKEOUT", "MEAN_REVERSION_TOO_EARLY", "ACCOUNTING_PROFIT_NOT_R_EDGE"} else 6.0
    score += 8.0 if proposal["rule_change_count"] <= MAX_ALLOWED_RULE_CHANGES else -20.0
    score += 8.0 if expectancy <= 0 else 0.0
    return max(0.0, min(100.0, score))


def _decision(score: float, trades: int, rule_change_count: int) -> str:
    if rule_change_count > MAX_ALLOWED_RULE_CHANGES:
        return "BLOCK_OVERFIT"
    if trades < MIN_REPAIR_SAMPLE:
        return "RESEARCH_QUEUE_SMALL_SAMPLE"
    if score >= 72:
        return "READY_FOR_TEST_LAB_DESIGN"
    if score >= 58:
        return "RESEARCH_QUEUE"
    return "DIAGNOSE_MORE_FIRST"


def _next_action(decision: str, proposal: dict[str, Any]) -> str:
    if decision == "READY_FOR_TEST_LAB_DESIGN":
        return f"Implement {proposal['proposal_id']} as a paper-only CandidateSpec after code review and run fresh-data tracking."
    if decision.startswith("RESEARCH_QUEUE"):
        return "Keep proposal visible, but collect more parent/repair evidence before spending more test budget."
    if decision == "BLOCK_OVERFIT":
        return "Reject proposal; too many simultaneous changes."
    return "Improve diagnosis first; do not code a new trading rule yet."


def _proposal_contract(record: dict[str, Any], proposal: dict[str, Any], failure_class: str) -> dict[str, Any]:
    return {
        "schema_version": ENGINE_VERSION,
        "parent_strategy": record.get("parent_strategy", "UNKNOWN"),
        "failure_class": failure_class,
        "proposal_id": proposal["proposal_id"],
        "changed_rules": proposal["changed_rules"],
        "unchanged_rules": proposal["unchanged_rules"],
        "fresh_data_requirement": "forward paper from next run or documented non-overlapping historical window",
        "promotion_gates": {
            "fresh_trades": [30, 50, 100],
            "expectancy_r": "> 0",
            "profit_factor": ">= 1.20",
            "beats_parent": True,
            "beats_random_baseline": True,
        },
        "blocked": ["real_orders", "same_window_retest_promotion", "unlimited_parameter_search", "silent_rule_mutation"],
    }


def _llm_prompt(record: dict[str, Any], proposal: dict[str, Any], failure_class: str) -> str:
    return (
        "Draft only one paper-trading repair candidate using the attached schema. "
        f"Parent={record.get('parent_strategy', 'UNKNOWN')}; failure_class={failure_class}; "
        f"hypothesis={proposal['hypothesis']} "
        "Do not add more than two changed rules. Do not use same-window retest as proof. "
        "Return JSON only."
    )


def _summary(report: pd.DataFrame, failure_records: list[dict[str, Any]]) -> StrategyBrainSummary:
    if report.empty:
        return StrategyBrainSummary(
            engine_version=ENGINE_VERSION,
            failures_read=0,
            proposals=0,
            test_lab_ready=0,
            research_queue=0,
            blocked=0,
            top_proposal="-",
            biggest_failure="-",
            status="NO_FAILURES_TO_LEARN_FROM",
            notes=["No failed strategy evidence was available for the intelligence engine."],
        )
    counts = report["decision"].value_counts().to_dict()
    top = report.sort_values("intelligence_score", ascending=False).iloc[0]
    biggest = min(failure_records, key=lambda item: float(item.get("net_pnl", 0.0))) if failure_records else {}
    ready = int(counts.get("READY_FOR_TEST_LAB_DESIGN", 0))
    return StrategyBrainSummary(
        engine_version=ENGINE_VERSION,
        failures_read=int(len(failure_records)),
        proposals=int(len(report)),
        test_lab_ready=ready,
        research_queue=int(counts.get("RESEARCH_QUEUE", 0) + counts.get("RESEARCH_QUEUE_SMALL_SAMPLE", 0)),
        blocked=int(counts.get("BLOCK_OVERFIT", 0) + counts.get("DIAGNOSE_MORE_FIRST", 0)),
        top_proposal=str(top["proposed_label"]),
        biggest_failure=str(biggest.get("parent_label", "-")),
        status="TEST_LAB_DESIGNS_READY" if ready else "LEARNING_QUEUE_ACTIVE",
        notes=[
            "This is a strategy-design brain, not an auto-trader.",
            "Every proposal is paper-only until fresh-data, random-baseline, and 30/50/100 trade gates pass.",
            "The engine learns from NERO failures by constraining the next hypothesis to the observed failure class.",
        ],
    )


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).upper())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "UNKNOWN"


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
