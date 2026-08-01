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

MAX_REPAIR_ATTEMPTS = 4
MIN_FORWARD_TRADES_FOR_REVIEW = 30
PROMOTION_TRADE_TARGET = 50


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

    rows: list[RepairLabAttempt] = []
    for row in workbench.to_dict("records"):
        repair_candidate = str(row.get("repair_candidate", "") or "")
        if not repair_candidate or repair_candidate == "NEEDS_DESIGN":
            rows.append(_design_required_row(row, current_time))
            continue
        attempt_count = _attempt_count(previous_attempts, repair_candidate)
        next_attempt = attempt_count + 1
        if attempt_count >= MAX_REPAIR_ATTEMPTS:
            rows.append(_cap_reached_row(row, attempt_count, current_time))
            continue

        historical_status = _historical_window_status(original_windows, row)
        mode = "HISTORICAL_UNSEEN_OR_FORWARD_PAPER"
        guard = "PASS_FRESH_DATA_REQUIRED"
        status = "READY_FOR_REPAIR_ATTEMPT"
        if "NO_ORIGINAL_WINDOW" in historical_status:
            mode = "FORWARD_PAPER_FROM_TODAY"
            guard = "PASS_FORWARD_PAPER_ACCEPTED"
            status = "FORWARD_REPAIR_TRACKING"

        rows.append(
            RepairLabAttempt(
                parent_strategy=str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN"),
                parent_label=str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN"),
                repair_candidate=repair_candidate,
                repair_label=str(row.get("repair_label", repair_candidate) or repair_candidate),
                attempt_number=next_attempt,
                max_attempts=MAX_REPAIR_ATTEMPTS,
                status=status,
                fresh_data_mode=mode,
                forward_start=current_time.date().isoformat(),
                historical_window_status=historical_status,
                anti_overfit_guard=guard,
                repair_trades=int(_num(row.get("repair_trades"), 0)),
                repair_net_pnl=round(_num(row.get("repair_net_pnl"), 0.0), 2),
                next_action=_next_action(status),
                llm_role="LLM may propose a repair, but deterministic tests decide.",
                allowed_changes="Max two scoped changes: regime gate, confirmation, target/stop, holding cap, volatility filter, asset/timeframe restriction.",
                blocked_changes="No same-window retest promotion, no unlimited parameter search, no post-result cherry-picking, no silent strategy mutation.",
                promotion_gate=(
                    f"{PROMOTION_TRADE_TARGET}+ fresh trades, expectancy > 0, profit factor >= 1.20, "
                    "drawdown acceptable, beats parent and random baseline."
                ),
            )
        )

    report = pd.DataFrame([asdict(row) for row in rows])
    if not report.empty:
        report = report.sort_values(["status", "repair_net_pnl"], ascending=[True, True])
    attempts_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(attempts_csv, index=False)
    attempts_json.write_text(json.dumps(report.to_dict("records"), indent=2), encoding="utf-8")
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


def _design_required_row(row: dict[str, Any], now: datetime) -> RepairLabAttempt:
    return RepairLabAttempt(
        parent_strategy=str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN"),
        parent_label=str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN"),
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
        next_action="Strategy Doctor must create a versioned repair candidate.",
        llm_role="LLM may draft a repair hypothesis only after failure autopsy.",
        allowed_changes="Max two scoped changes after diagnosis.",
        blocked_changes="No test until repair contract exists.",
        promotion_gate="Not eligible.",
    )


def _cap_reached_row(row: dict[str, Any], attempt_count: int, now: datetime) -> RepairLabAttempt:
    return RepairLabAttempt(
        parent_strategy=str(row.get("quarantined_strategy", "UNKNOWN") or "UNKNOWN"),
        parent_label=str(row.get("quarantined_label", "UNKNOWN") or "UNKNOWN"),
        repair_candidate=str(row.get("repair_candidate", "UNKNOWN") or "UNKNOWN"),
        repair_label=str(row.get("repair_label", row.get("repair_candidate", "UNKNOWN")) or "UNKNOWN"),
        attempt_number=attempt_count,
        max_attempts=MAX_REPAIR_ATTEMPTS,
        status="ATTEMPT_CAP_REACHED",
        fresh_data_mode="BLOCKED",
        forward_start=now.date().isoformat(),
        historical_window_status="REPAIR_CAP_EXHAUSTED",
        anti_overfit_guard="BLOCKED_MAX_4_ATTEMPTS",
        repair_trades=int(_num(row.get("repair_trades"), 0)),
        repair_net_pnl=round(_num(row.get("repair_net_pnl"), 0.0), 2),
        next_action="Mark parent idea permanently rejected unless user manually reopens research.",
        llm_role="LLM blocked from proposing more repairs for this lineage.",
        allowed_changes="None.",
        blocked_changes="No fifth repair attempt.",
        promotion_gate="Not eligible.",
    )


def _historical_window_status(original_windows: pd.DataFrame, row: dict[str, Any]) -> str:
    parent = str(row.get("quarantined_strategy", "") or "")
    if original_windows.empty or "candidate_id" not in original_windows:
        return "NO_ORIGINAL_WINDOW_RECORDED_FORWARD_ONLY"
    matches = original_windows[original_windows["candidate_id"].astype(str) == parent]
    if matches.empty:
        return "NO_ORIGINAL_WINDOW_FOR_PARENT_FORWARD_ONLY"
    return "ORIGINAL_WINDOW_RECORDED_REQUIRE_NON_OVERLAP_CHECK"


def _next_action(status: str) -> str:
    if status == "FORWARD_REPAIR_TRACKING":
        return f"Paper-track from today until at least {MIN_FORWARD_TRADES_FOR_REVIEW} trades; do not reuse failed backtest window."
    if status == "READY_FOR_REPAIR_ATTEMPT":
        return "Run historical unseen-window validation or forward paper tracking before judging repair."
    return "Review Repair Lab status."


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


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


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
