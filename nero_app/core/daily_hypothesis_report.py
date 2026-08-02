from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PAKISTAN_TZ = ZoneInfo("Asia/Karachi")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_STATE_PATH = DEFAULT_REPORT_DIR / "daily_hypothesis_report_state.json"


@dataclass(frozen=True)
class DailyHypothesisSummary:
    report_date: str
    active_new_hypotheses: int
    design_required: int
    new_watchlist_additions: int
    total_watchlist: int
    successful_strategies: int
    top_strategy: str
    top_strategy_net_pnl: float
    top_strategy_expectancy_r: float
    notification_due: bool


def build_daily_hypothesis_report(
    report_dir: Path | str = DEFAULT_REPORT_DIR,
    state_path: Path | str = DEFAULT_STATE_PATH,
    now: datetime | None = None,
    update_state: bool = True,
) -> tuple[dict[str, Any], DailyHypothesisSummary]:
    report_base = Path(report_dir)
    state_file = Path(state_path)
    current_date = _report_date(now)
    state = _read_json(state_file)

    repair = _read_csv(report_base / "strategy_repair_lab_attempts.csv")
    verification = _read_csv(report_base / "strategy_verification_report.csv")
    profit_edge = _read_csv(report_base / "profit_edge_report.csv")

    active_hypotheses = _active_hypotheses_created_on(repair, current_date)
    design_required = _design_required_on(repair, current_date)
    watchlist = _watchlist_rows(verification, profit_edge)
    successful = _successful_strategy_rows(profit_edge)

    active_ids = _preferred_row_ids(active_hypotheses, ["repair_candidate", "repair_label"])
    watchlist_ids = _preferred_row_ids(watchlist, ["candidate_id", "display_label"])
    seen_hypotheses = set(state.get("seen_hypothesis_ids", []))
    seen_watchlist = set(state.get("seen_watchlist_ids", []))
    new_hypotheses = sorted(active_ids - seen_hypotheses)
    new_watchlist = sorted(watchlist_ids - seen_watchlist)
    notification_due = bool(state.get("last_sent_date") != current_date or new_hypotheses or new_watchlist)

    top = _top_successful_row(successful)
    summary = DailyHypothesisSummary(
        report_date=current_date,
        active_new_hypotheses=len(active_hypotheses),
        design_required=len(design_required),
        new_watchlist_additions=len(new_watchlist),
        total_watchlist=len(watchlist),
        successful_strategies=len(successful),
        top_strategy=str(top.get("display_label", "NONE")),
        top_strategy_net_pnl=float(top.get("net_pnl", 0.0) or 0.0),
        top_strategy_expectancy_r=float(top.get("expectancy_r", 0.0) or 0.0),
        notification_due=notification_due,
    )

    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "new_hypotheses": _records(active_hypotheses),
        "design_required": _records(design_required),
        "watchlist": _records(watchlist),
        "new_watchlist_ids": new_watchlist,
        "successful_strategies": _records(successful),
        "source_files": {
            "repair_lab": str(report_base / "strategy_repair_lab_attempts.csv"),
            "verification": str(report_base / "strategy_verification_report.csv"),
            "profit_edge": str(report_base / "profit_edge_report.csv"),
        },
    }
    _write_outputs(report_base, payload, summary)

    if update_state:
        _write_json(
            state_file,
            {
                "last_sent_date": current_date,
                "seen_hypothesis_ids": sorted(seen_hypotheses | active_ids),
                "seen_watchlist_ids": sorted(seen_watchlist | watchlist_ids),
                "last_summary": asdict(summary),
            },
        )

    return payload, summary


def format_daily_hypothesis_message(summary: DailyHypothesisSummary, payload: dict[str, Any]) -> str:
    top_lines = []
    for row in payload.get("successful_strategies", [])[:4]:
        top_lines.append(
            f"- {row.get('display_label', row.get('candidate_id', 'UNKNOWN'))}: "
            f"net ${float(row.get('net_pnl', 0.0) or 0.0):.2f}, "
            f"ExpR {float(row.get('expectancy_r', 0.0) or 0.0):+.2f}, "
            f"trades {int(float(row.get('total_trades', 0) or 0))}"
        )
    if not top_lines:
        top_lines.append("- No successful strategy candidate today.")

    return "\n".join(
        [
            f"NERO Daily Hypothesis Report | {summary.report_date}",
            f"New active repair hypotheses: {summary.active_new_hypotheses}",
            f"Need Strategy Doctor design: {summary.design_required}",
            f"New watchlist additions: {summary.new_watchlist_additions}",
            f"Total watchlist: {summary.total_watchlist}",
            f"Successful/profit candidates: {summary.successful_strategies}",
            "Top candidates:",
            *top_lines,
            "Paper research only. No real orders.",
        ]
    )


def _report_date(now: datetime | None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(PAKISTAN_TZ).date().isoformat()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_outputs(report_base: Path, payload: dict[str, Any], summary: DailyHypothesisSummary) -> None:
    report_base.mkdir(parents=True, exist_ok=True)
    _write_json(report_base / "daily_hypothesis_report.json", payload)
    (report_base / "daily_hypothesis_report.txt").write_text(
        format_daily_hypothesis_message(summary, payload),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(summary)]).to_csv(report_base / "daily_hypothesis_report.csv", index=False)


def _active_hypotheses_created_on(report: pd.DataFrame, report_date: str) -> pd.DataFrame:
    if report.empty or "forward_start" not in report:
        return pd.DataFrame()
    rows = report[report["forward_start"].astype(str).eq(report_date)].copy()
    if rows.empty:
        return rows
    repair = rows.get("repair_candidate", pd.Series(dtype=str)).astype(str)
    status = rows.get("status", pd.Series(dtype=str)).astype(str)
    return rows[~repair.eq("NEEDS_DESIGN") & status.ne("DESIGN_REQUIRED")]


def _design_required_on(report: pd.DataFrame, report_date: str) -> pd.DataFrame:
    if report.empty or "forward_start" not in report:
        return pd.DataFrame()
    rows = report[report["forward_start"].astype(str).eq(report_date)].copy()
    if rows.empty:
        return rows
    status = rows.get("status", pd.Series(dtype=str)).astype(str)
    repair = rows.get("repair_candidate", pd.Series(dtype=str)).astype(str)
    return rows[status.eq("DESIGN_REQUIRED") | repair.eq("NEEDS_DESIGN")]


def _watchlist_rows(verification: pd.DataFrame, profit_edge: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not verification.empty:
        verdict = verification.get("verdict", pd.Series(dtype=str)).astype(str)
        frames.append(verification[verdict.eq("WATCHLIST")].copy())
    if not profit_edge.empty:
        role = profit_edge.get("role", pd.Series(dtype=str)).astype(str)
        frames.append(profit_edge[role.eq("PROFIT_CANDIDATE")].copy())
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "candidate_id" in merged:
        merged = merged.drop_duplicates(subset=["candidate_id"], keep="first")
    sort_columns = [column for column in ["net_pnl", "expectancy_r"] if column in merged]
    if sort_columns:
        merged = merged.sort_values(sort_columns, ascending=False, na_position="last")
    return merged


def _successful_strategy_rows(profit_edge: pd.DataFrame) -> pd.DataFrame:
    if profit_edge.empty:
        return pd.DataFrame()
    role = profit_edge.get("role", pd.Series(dtype=str)).astype(str)
    net = pd.to_numeric(profit_edge.get("net_pnl", 0), errors="coerce").fillna(0.0)
    exp = pd.to_numeric(profit_edge.get("expectancy_r", 0), errors="coerce").fillna(0.0)
    rows = profit_edge[role.eq("PROFIT_CANDIDATE") & net.gt(0) & exp.gt(0)].copy()
    if rows.empty:
        return rows
    rows["edge_score_sort"] = pd.to_numeric(rows.get("edge_score", 0), errors="coerce").fillna(0.0)
    rows["net_pnl_sort"] = pd.to_numeric(rows.get("net_pnl", 0), errors="coerce").fillna(0.0)
    return rows.sort_values(["edge_score_sort", "net_pnl_sort"], ascending=False).drop(
        columns=["edge_score_sort", "net_pnl_sort"],
        errors="ignore",
    )


def _top_successful_row(successful: pd.DataFrame) -> dict[str, Any]:
    if successful.empty:
        return {"display_label": "NONE", "net_pnl": 0.0, "expectancy_r": 0.0}
    return successful.iloc[0].to_dict()


def _preferred_row_ids(rows: pd.DataFrame, columns: list[str]) -> set[str]:
    if rows.empty:
        return set()
    ids: set[str] = set()
    for _, row in rows.iterrows():
        for column in columns:
            if column not in rows:
                continue
            value = row.get(column)
            if pd.notna(value) and str(value) and str(value) != "nan":
                ids.add(str(value))
                break
    return ids


def _records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    clean = rows.where(pd.notna(rows), None)
    return clean.to_dict(orient="records")

