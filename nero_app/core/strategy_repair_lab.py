"""Closed-loop Strategy Repair Lab for Project NERO.

Plans repair attempts, validates fresh-data eligibility, and records
anti-overfitting guardrails. It does not change strategy parameters, promote
strategies, or place orders.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_WORKBENCH_CSV = DEFAULT_REPORT_DIR / "strategy_repair_workbench.csv"
DEFAULT_ATTEMPTS_CSV = DEFAULT_REPORT_DIR / "strategy_repair_lab_attempts.csv"
DEFAULT_ATTEMPTS_JSON = DEFAULT_REPORT_DIR / "strategy_repair_lab_attempts.json"
DEFAULT_ORIGINAL_WINDOWS_CSV = DEFAULT_REPORT_DIR / "strategy_repair_original_windows.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_REPORT_DIR / "strategy_lab_summary.csv"
DEFAULT_RANDOM_BASELINE_CSV = DEFAULT_REPORT_DIR / "strategy_random_baseline.csv"
DEFAULT_LINEAGE_CSV = DEFAULT_REPORT_DIR / "strategy_repair_lineage.csv"
DEFAULT_LINEAGE_JSON = DEFAULT_REPORT_DIR / "strategy_repair_lineage.json"

MAX_REPAIR_ATTEMPTS = 4
MIN_FORWARD_TRADES_FOR_REVIEW = 30
PROMOTION_TRADE_TARGET = 50
PROMOTION_PROFIT_FACTOR = 1.20
REPAIR_SCHEMA_VERSION = "repair_proposal_v1"


@dataclass(frozen=True)
class FreshDataWindow:
    original_start: str
    original_end: str
    repair_start: str
    repair_end: str
    overlaps_original: bool
    valid: bool
    reason: str


@dataclass(frozen=True)
class RepairLabAttempt:
    parent_strategy: str
    parent_label: str
    repair_candidate: str
    repair_label: str
    attempt_number: int
    max_attempts: int
    status: str
    fresh_data_mode: str
    forward_start: str
    historical_window_status: str
    anti_overfit_guard: str
    repair_trades: int
    repair_net_pnl: float
    parent_trades: int
    parent_net_pnl: float
    parent_expectancy_r: float
    parent_profit_factor: float
    repair_expectancy_r: float
    repair_profit_factor: float
    repair_vs_parent_net_delta: float
    failure_reason_code: str
    failure_reason_detail: str
    repair_hypothesis_schema_version: str
    repair_proposal_schema: str
    random_baseline_status: str
    random_baseline_expectancy_r: float
    beats_parent: bool
    beats_random_baseline: bool
    sample_milestone: str
    promotion_decision: str
    lineage_status: str
    dashboard_lineage_label: str
    next_action: str
    llm_role: str
    allowed_changes: str
    blocked_changes: str
    promotion_gate: str


def build_strategy_repair_lab_report(
    workbench_csv: Path = DEFAULT_WORKBENCH_CSV,
    attempts_csv: Path = DEFAULT_ATTEMPTS_CSV,
    attempts_json: Path = DEFAULT_ATTEMPTS_JSON,
    original_windows_csv: Path = DEFAULT_ORIGINAL_WINDOWS_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    random_baseline_csv: Path = DEFAULT_RANDOM_BASELINE_CSV,
    lineage_csv: Path = DEFAULT_LINEAGE_CSV,
    lineage_json: Path = DEFAULT_LINEAGE_JSON,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Create the auditable repair-loop plan from the current workbench.

    If a genuinely non-overlapping historical window is not recorded, the
    strategy is assigned to forward paper tracking from ``now``. Forward paper
    tracking is a first-class fresh-data source, not a fallback.
    """
    current_time = now or datetime.now(timezone.utc)
    workbench = _safe_read_csv(workbench_csv)
    previous_attempts = _safe_read_csv(attempts_csv)
    original_windows = _safe_read_csv(original_windows_csv)
    strategy_summary = _safe_read_csv(summary_csv)
    random_baselines = _safe_read_csv(random_baseline_csv)

    rows: list[RepairLabAttempt] = []
    for row in workbench.to_dict("records"):
        repair_candidate = str(row.get("repair_candidate", "") or "")
        if not repair_candidate or repair_candidate == "NEEDS_DESIGN":
            rows.append(_design_required_row(row, current_time, strategy_summary, random_baselines))
            continue
        attempt_count = _attempt_count(previous_attempts, repair_candidate)
        next_attempt = attempt_count + 1
        if attempt_count >= MAX_REPAIR_ATTEMPTS:
            rows.append(_cap_reached_row(row, attempt_count, current_time, strategy_summary, random_baselines))
            continue

        historical_status = _historical_window_status(original_windows, row)
        mode = "HISTORICAL_UNSEEN_OR_FORWARD_PAPER"
        guard = "PASS_FRESH_DATA_REQUIRED"
        status = "READY_FOR_REPAIR_ATTEMPT"
        if "NO_ORIGINAL_WINDOW" in historical_status:
            mode = "FORWARD_PAPER_FROM_TODAY"
            guard = "PASS_FORWARD_PAPER_ACCEPTED"
            status = "FORWARD_REPAIR_TRACKING"

        rows.append(_attempt_row(row, next_attempt, status, mode, guard, historical_status, current_time, strategy_summary, random_baselines))

    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["status", "repair_net_pnl"], ascending=[True, True])
    attempts_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(attempts_csv, index=False)
    attempts_json.write_text(json.dumps(report.to_dict("records"), indent=2), encoding="utf-8")
    lineage = _lineage_report(report)
    lineage.to_csv(lineage_csv, index=False)
    lineage_json.write_text(json.dumps(lineage.to_dict("records"), indent=2), encoding="utf-8")
    return report


def validate_fresh_data_window(
    original_start: str,
    original_end: str,
    repair_start: str,
    repair_end: str,
) -> FreshDataWindow:
    """Validate that a repair test window is genuinely unseen.

    Touching boundaries count as overlap because a closed candle near the
    boundary can leak information through indicator warm-up.
    """
    original_a = _parse_date(original_start)
    original_b = _parse_date(original_end)
    repair_a = _parse_date(repair_start)
    repair_b = _parse_date(repair_end)
    if not all([original_a, original_b, repair_a, repair_b]):
        return FreshDataWindow(original_start, original_end, repair_start, repair_end, True, False, "INVALID_DATE")
    if original_a > original_b or repair_a > repair_b:
        return FreshDataWindow(original_start, original_end, repair_start, repair_end, True, False, "INVALID_RANGE")
    overlaps = not (repair_b < original_a or repair_a > original_b)
    reason = "FRESH_UNSEEN_WINDOW" if not overlaps else "OVERLAPS_ORIGINAL_FAILED_WINDOW"
    return FreshDataWindow(original_start, original_end, repair_start, repair_end, overlaps, not overlaps, reason)


def _attempt_row(
    row: dict[str, Any],
    attempt_number: int,
    status: str,
    mode: str,
    guard: str,
    historical_status: str,
    now: datetime,
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
) -> RepairLabAttempt:
    parent_strategy = str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN")
    parent_label = str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN")
    repair_candidate = str(row.get("repair_candidate", "UNKNOWN") or "UNKNOWN")
    repair_label = str(row.get("repair_label", repair_candidate) or repair_candidate)
    metrics = _repair_metrics(row, parent_strategy, repair_candidate, summary, baselines)
    failure_code, failure_detail = _failure_reason(row, metrics)
    milestone = _sample_milestone(metrics["repair_trades"])
    random_status, random_exp = _random_baseline(baselines, repair_candidate)
    beats_parent = metrics["repair_net_pnl"] > metrics["parent_net_pnl"] and metrics["repair_expectancy_r"] > metrics["parent_expectancy_r"]
    beats_random = random_status == "BASELINE_NOT_AVAILABLE" or metrics["repair_expectancy_r"] > random_exp
    promotion = _promotion_decision(status, metrics, beats_parent, beats_random, random_status)
    return RepairLabAttempt(
        parent_strategy=parent_strategy,
        parent_label=parent_label,
        repair_candidate=repair_candidate,
        repair_label=repair_label,
        attempt_number=attempt_number,
        max_attempts=MAX_REPAIR_ATTEMPTS,
        status=status,
        fresh_data_mode=mode,
        forward_start=now.date().isoformat(),
        historical_window_status=historical_status,
        anti_overfit_guard=guard,
        repair_trades=metrics["repair_trades"],
        repair_net_pnl=round(metrics["repair_net_pnl"], 2),
        parent_trades=metrics["parent_trades"],
        parent_net_pnl=round(metrics["parent_net_pnl"], 2),
        parent_expectancy_r=round(metrics["parent_expectancy_r"], 4),
        parent_profit_factor=round(metrics["parent_profit_factor"], 4),
        repair_expectancy_r=round(metrics["repair_expectancy_r"], 4),
        repair_profit_factor=round(metrics["repair_profit_factor"], 4),
        repair_vs_parent_net_delta=round(metrics["repair_net_pnl"] - metrics["parent_net_pnl"], 2),
        failure_reason_code=failure_code,
        failure_reason_detail=failure_detail,
        repair_hypothesis_schema_version=REPAIR_SCHEMA_VERSION,
        repair_proposal_schema=_repair_proposal_schema(failure_code),
        random_baseline_status=random_status,
        random_baseline_expectancy_r=round(random_exp, 4),
        beats_parent=beats_parent,
        beats_random_baseline=beats_random,
        sample_milestone=milestone,
        promotion_decision=promotion,
        lineage_status=_lineage_status(status, promotion),
        dashboard_lineage_label=f"{parent_label} -> {repair_label}: {promotion}",
        next_action=_next_action(status, promotion, milestone),
        llm_role="LLM may draft a repair only in the strict JSON schema; deterministic tests decide promotion.",
        allowed_changes="Max two scoped changes: regime gate, confirmation, target/stop, holding cap, volatility filter, asset/timeframe restriction.",
        blocked_changes="No same-window retest promotion, no unlimited parameter search, no post-result cherry-picking, no silent strategy mutation.",
        promotion_gate=(
            f"30/50/100 fresh-trade milestones, expectancy > 0, profit factor >= {PROMOTION_PROFIT_FACTOR:.2f}, "
            "drawdown acceptable, beats parent and random baseline."
        ),
    )


def _design_required_row(row: dict[str, Any], now: datetime, summary: pd.DataFrame, baselines: pd.DataFrame) -> RepairLabAttempt:
    parent_strategy = str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN")
    parent_label = str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN")
    metrics = _repair_metrics(row, parent_strategy, "NEEDS_DESIGN", summary, baselines)
    failure_code, failure_detail = _failure_reason(row, metrics)
    return RepairLabAttempt(
        parent_strategy=parent_strategy,
        parent_label=parent_label,
        repair_candidate="NEEDS_DESIGN",
        repair_label="NEEDS_DESIGN",
        attempt_number=0,
        max_attempts=MAX_REPAIR_ATTEMPTS,
        status="DESIGN_REQUIRED",
        fresh_data_mode="NOT_ASSIGNED",
        forward_start=now.date().isoformat(),
        historical_window_status="NO_REPAIR_CANDIDATE",
        anti_overfit_guard="BLOCKED_NO_REPAIR_SPEC",
        repair_trades=0,
        repair_net_pnl=0.0,
        parent_trades=metrics["parent_trades"],
        parent_net_pnl=round(metrics["parent_net_pnl"], 2),
        parent_expectancy_r=round(metrics["parent_expectancy_r"], 4),
        parent_profit_factor=round(metrics["parent_profit_factor"], 4),
        repair_expectancy_r=0.0,
        repair_profit_factor=0.0,
        repair_vs_parent_net_delta=round(0.0 - metrics["parent_net_pnl"], 2),
        failure_reason_code=failure_code,
        failure_reason_detail=failure_detail,
        repair_hypothesis_schema_version=REPAIR_SCHEMA_VERSION,
        repair_proposal_schema=_repair_proposal_schema(failure_code),
        random_baseline_status="NO_REPAIR_NO_BASELINE",
        random_baseline_expectancy_r=0.0,
        beats_parent=False,
        beats_random_baseline=False,
        sample_milestone="DESIGN_REQUIRED",
        promotion_decision="DESIGN_REQUIRED",
        lineage_status="NEEDS_DESIGN",
        dashboard_lineage_label=f"{parent_label} -> NEEDS_DESIGN: DESIGN_REQUIRED",
        next_action="Strategy Doctor must create a versioned repair candidate.",
        llm_role="LLM may draft a repair hypothesis only after failure autopsy.",
        allowed_changes="Max two scoped changes after diagnosis.",
        blocked_changes="No test until repair contract exists.",
        promotion_gate="Not eligible.",
    )


def _cap_reached_row(row: dict[str, Any], attempt_count: int, now: datetime, summary: pd.DataFrame, baselines: pd.DataFrame) -> RepairLabAttempt:
    parent_strategy = str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN")
    repair_candidate = str(row.get("repair_candidate", "UNKNOWN") or "UNKNOWN")
    metrics = _repair_metrics(row, parent_strategy, repair_candidate, summary, baselines)
    failure_code, failure_detail = _failure_reason(row, metrics)
    random_status, random_exp = _random_baseline(baselines, repair_candidate)
    return RepairLabAttempt(
        parent_strategy=parent_strategy,
        parent_label=str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN"),
        repair_candidate=repair_candidate,
        repair_label=str(row.get("repair_label", row.get("repair_candidate", "UNKNOWN")) or "UNKNOWN"),
        attempt_number=attempt_count,
        max_attempts=MAX_REPAIR_ATTEMPTS,
        status="ATTEMPT_CAP_REACHED",
        fresh_data_mode="BLOCKED",
        forward_start=now.date().isoformat(),
        historical_window_status="REPAIR_CAP_EXHAUSTED",
        anti_overfit_guard="BLOCKED_MAX_4_ATTEMPTS",
        repair_trades=metrics["repair_trades"],
        repair_net_pnl=round(metrics["repair_net_pnl"], 2),
        parent_trades=metrics["parent_trades"],
        parent_net_pnl=round(metrics["parent_net_pnl"], 2),
        parent_expectancy_r=round(metrics["parent_expectancy_r"], 4),
        parent_profit_factor=round(metrics["parent_profit_factor"], 4),
        repair_expectancy_r=round(metrics["repair_expectancy_r"], 4),
        repair_profit_factor=round(metrics["repair_profit_factor"], 4),
        repair_vs_parent_net_delta=round(metrics["repair_net_pnl"] - metrics["parent_net_pnl"], 2),
        failure_reason_code=failure_code,
        failure_reason_detail=failure_detail,
        repair_hypothesis_schema_version=REPAIR_SCHEMA_VERSION,
        repair_proposal_schema=_repair_proposal_schema(failure_code),
        random_baseline_status=random_status,
        random_baseline_expectancy_r=round(random_exp, 4),
        beats_parent=False,
        beats_random_baseline=False,
        sample_milestone="ATTEMPT_CAP_REACHED",
        promotion_decision="PERMANENTLY_DEAD",
        lineage_status="DEAD_AFTER_4",
        dashboard_lineage_label="Attempt cap reached: permanently dead unless manually reopened.",
        next_action="Mark parent idea permanently rejected unless user manually reopens research.",
        llm_role="LLM blocked from proposing more repairs for this lineage.",
        allowed_changes="None.",
        blocked_changes="No fifth repair attempt.",
        promotion_gate="Not eligible.",
    )


def _repair_metrics(row: dict[str, Any], parent_strategy: str, repair_candidate: str, summary: pd.DataFrame, baselines: pd.DataFrame) -> dict[str, float | int]:
    parent_summary = _summary_row(summary, parent_strategy)
    repair_summary = _summary_row(summary, repair_candidate)
    parent_trades = int(_first_num(parent_summary, "total_trades", row.get("quarantined_trades"), 0))
    repair_trades = int(_first_num(repair_summary, "total_trades", row.get("repair_trades"), 0))
    parent_net = _first_num(parent_summary, "net_pnl", row.get("quarantined_net_pnl"), 0.0)
    repair_net = _first_num(repair_summary, "net_pnl", row.get("repair_net_pnl"), 0.0)
    return {
        "parent_trades": parent_trades,
        "repair_trades": repair_trades,
        "parent_net_pnl": parent_net,
        "repair_net_pnl": repair_net,
        "parent_expectancy_r": _first_num(parent_summary, "expectancy_r", None, _estimate_expectancy(parent_net, parent_trades)),
        "parent_profit_factor": _first_num(parent_summary, "profit_factor", None, 0.0),
        "repair_expectancy_r": _first_num(repair_summary, "expectancy_r", None, _estimate_expectancy(repair_net, repair_trades)),
        "repair_profit_factor": _first_num(repair_summary, "profit_factor", None, 0.0),
    }


def _failure_reason(row: dict[str, Any], metrics: dict[str, float | int]) -> tuple[str, str]:
    diagnosis = str(row.get("diagnosis", "") or "")
    parent = str(row.get("quarantined_strategy", "") or "")
    parent_label = str(row.get("quarantined_label", "") or "")
    trades = int(metrics["parent_trades"])
    net = float(metrics["parent_net_pnl"])
    exp = float(metrics["parent_expectancy_r"])
    pf = float(metrics["parent_profit_factor"])
    text = f"{parent} {parent_label} {diagnosis}".upper()
    if trades >= 30 and net <= -1000:
        return "CAPITAL_DRAIN", "Large negative net P/L with 30+ trades; strategy is draining paper capital."
    if "BREAKOUT" in text:
        return "BREAKOUT_FAKEOUT", diagnosis or "Breakout entries likely fail on fakeouts or weak trend-quality confirmation."
    if "1R" in text or "ASYM" in text:
        return "EXIT_ASYMMETRY", diagnosis or "Reward/stop structure is not paying enough after costs and adverse exits."
    if "MR" in text or "MEAN" in text:
        return "MEAN_REVERSION_EARLY_ENTRY", diagnosis or "Mean-reversion entries may be too early or regime gate too weak."
    if trades < MIN_FORWARD_TRADES_FOR_REVIEW:
        return "INSUFFICIENT_FAILURE_SAMPLE", "Failure sample is below the minimum review threshold."
    if exp < 0 or (pf and pf < 1):
        return "NEGATIVE_EDGE", "Expectancy or profit factor is below break-even on observed data."
    return "NEEDS_DIAGNOSIS", diagnosis or "No specific failure class found; requires manual Strategy Doctor review."


def _promotion_decision(status: str, metrics: dict[str, float | int], beats_parent: bool, beats_random: bool, random_status: str) -> str:
    repair_trades = int(metrics["repair_trades"])
    repair_net = float(metrics["repair_net_pnl"])
    repair_exp = float(metrics["repair_expectancy_r"])
    repair_pf = float(metrics["repair_profit_factor"])
    if status == "ATTEMPT_CAP_REACHED":
        return "PERMANENTLY_DEAD"
    if status == "DESIGN_REQUIRED":
        return "DESIGN_REQUIRED"
    if repair_trades < MIN_FORWARD_TRADES_FOR_REVIEW:
        return "COLLECT_FRESH_DATA"
    if random_status == "BASELINE_NOT_AVAILABLE":
        return "COLLECT_RANDOM_BASELINE"
    if repair_net <= 0 or repair_exp <= 0 or repair_pf < PROMOTION_PROFIT_FACTOR:
        return "REJECT_REPAIR"
    if not beats_parent:
        return "REJECT_REPAIR_NOT_BETTER_THAN_PARENT"
    if not beats_random:
        return "REJECT_REPAIR_NOT_BETTER_THAN_RANDOM"
    if repair_trades >= 100:
        return "PROMOTE_CANDIDATE"
    if repair_trades >= PROMOTION_TRADE_TARGET:
        return "PROMOTE_SHADOW"
    return "WATCHLIST_REPAIR"


def _sample_milestone(trades: int) -> str:
    if trades >= 100:
        return "REVIEW_100"
    if trades >= PROMOTION_TRADE_TARGET:
        return "REVIEW_50"
    if trades >= MIN_FORWARD_TRADES_FOR_REVIEW:
        return "REVIEW_30"
    return "UNDER_30_COLLECTING"


def _next_action(status: str, promotion: str, milestone: str) -> str:
    if promotion == "COLLECT_FRESH_DATA":
        return f"Paper-track until at least {MIN_FORWARD_TRADES_FOR_REVIEW} fresh trades; current milestone {milestone}."
    if promotion == "COLLECT_RANDOM_BASELINE":
        return "Build/run same-universe random-entry baseline before any promotion."
    if promotion.startswith("REJECT"):
        return "Do not trade; send back to Strategy Doctor or mark failed after capped attempts."
    if promotion.startswith("PROMOTE"):
        return "Eligible for controlled shadow promotion after audit review; still paper-only."
    if status == "FORWARD_REPAIR_TRACKING":
        return f"Paper-track from today until at least {MIN_FORWARD_TRADES_FOR_REVIEW} trades; do not reuse failed backtest window."
    if status == "READY_FOR_REPAIR_ATTEMPT":
        return "Run historical unseen-window validation or forward paper tracking before judging repair."
    return "Review Repair Lab status."


def _lineage_status(status: str, promotion: str) -> str:
    if promotion == "PERMANENTLY_DEAD":
        return "DEAD_AFTER_4"
    if promotion == "DESIGN_REQUIRED":
        return "NEEDS_DESIGN"
    if promotion.startswith("PROMOTE"):
        return "REPAIR_ELIGIBLE"
    if promotion.startswith("REJECT"):
        return "REPAIR_REJECTED"
    return "ACTIVE_FORWARD_REPAIR"


def _repair_proposal_schema(failure_code: str) -> str:
    schema = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "failure_reason_code": failure_code,
        "proposal_id": "required_unique_version_id",
        "parent_strategy": "required_parent_id",
        "changed_rules": ["max_two_scoped_rule_changes"],
        "unchanged_rules": ["entry_family", "paper_only", "risk_model"],
        "fresh_data_plan": {"mode": "FORWARD_PAPER_FROM_TODAY_OR_UNSEEN_HISTORICAL", "no_overlap_required": True},
        "promotion_gates": {"min_trades": [30, 50, 100], "expectancy_r": ">0", "profit_factor": f">={PROMOTION_PROFIT_FACTOR:.2f}", "beats_random_baseline": True},
        "blocked": ["same_window_retest", "unlimited_parameter_search", "silent_mutation", "real_orders"],
    }
    return json.dumps(schema, sort_keys=True)


def _lineage_report(report: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "parent_label",
        "repair_label",
        "attempt_number",
        "failure_reason_code",
        "parent_net_pnl",
        "repair_net_pnl",
        "repair_vs_parent_net_delta",
        "repair_trades",
        "sample_milestone",
        "random_baseline_status",
        "promotion_decision",
        "lineage_status",
        "dashboard_lineage_label",
    ]
    if report.empty:
        return pd.DataFrame(columns=columns)
    return report[[column for column in columns if column in report.columns]].copy()


def _historical_window_status(original_windows: pd.DataFrame, row: dict[str, Any]) -> str:
    parent = str(row.get("quarantined_strategy", "") or "")
    if original_windows.empty or "candidate_id" not in original_windows:
        return "NO_ORIGINAL_WINDOW_RECORDED_FORWARD_ONLY"
    matches = original_windows[original_windows["candidate_id"].astype(str) == parent]
    if matches.empty:
        return "NO_ORIGINAL_WINDOW_FOR_PARENT_FORWARD_ONLY"
    return "ORIGINAL_WINDOW_RECORDED_REQUIRE_NON_OVERLAP_CHECK"


def _attempt_count(previous_attempts: pd.DataFrame, repair_candidate: str) -> int:
    if previous_attempts.empty or "repair_candidate" not in previous_attempts:
        return 0
    matches = previous_attempts[previous_attempts["repair_candidate"].astype(str) == repair_candidate]
    if matches.empty:
        return 0
    if "status" in matches:
        closed_statuses = {"DIED_AGAIN", "REPAIRED_FAILED", "ATTEMPT_CLOSED", "REJECTED"}
        matches = matches[matches["status"].astype(str).str.upper().isin(closed_statuses)]
        if matches.empty:
            return 0
    if "attempt_number" not in matches:
        return len(matches)
    return int(matches["attempt_number"].max())


def _summary_row(summary: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    if summary.empty or "candidate_id" not in summary:
        return {}
    matches = summary[summary["candidate_id"].astype(str) == str(candidate_id)]
    if matches.empty and "display_label" in summary:
        matches = summary[summary["display_label"].astype(str) == str(candidate_id)]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def _random_baseline(baselines: pd.DataFrame, candidate_id: str) -> tuple[str, float]:
    if baselines.empty:
        return "BASELINE_NOT_AVAILABLE", 0.0
    id_columns = [column for column in ["candidate_id", "strategy", "strategy_id"] if column in baselines]
    matches = baselines
    if id_columns:
        column = id_columns[0]
        matches = baselines[baselines[column].astype(str) == str(candidate_id)]
    if matches.empty:
        return "BASELINE_NOT_AVAILABLE", 0.0
    row = matches.iloc[0].to_dict()
    return "BASELINE_AVAILABLE", _first_num(row, "expectancy_r", row.get("random_expectancy_r"), 0.0)


def _first_num(row: dict[str, Any], key: str, fallback: Any, default: float = 0.0) -> float:
    if key in row:
        value = _num(row.get(key), None)
        if value is not None:
            return value
    value = _num(fallback, None)
    if value is not None:
        return value
    return default


def _estimate_expectancy(net_pnl: float, trades: int) -> float:
    if trades <= 0:
        return 0.0
    return net_pnl / (trades * 100.0)


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _num(value: Any, default: float | None = 0.0) -> float | None:
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
