"""Strategy performance auditor for Project Nero.

The auditor is a proof layer: it summarizes whether NERO's paper strategies
have enough sample size and whether current results are reliable enough to use
as evidence. It never treats a backtest or paper trade as a forward guarantee.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nero_app.core.mean_reversion_agent import DEFAULT_DATA_DIR, DEFAULT_REPORT_DIR
from nero_app.core.prediction_log import DEFAULT_LOG_PATH, build_prediction_truth_report, load_prediction_log


@dataclass(frozen=True)
class StrategyAuditReport:
    grade: str
    score: float
    total_closed_trades: int
    total_saved_signals: int
    evaluated_signals: int
    best_asset: str
    top_blocker: str
    insufficient_sample: bool
    notes: list[str]
    rows: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_strategy_performance_audit(
    mean_reversion_report_path: Path = DEFAULT_REPORT_DIR / "mean_reversion_report.csv",
    closed_trades_path: Path = DEFAULT_DATA_DIR / "trades" / "closed_trades.csv",
    evaluations_path: Path = DEFAULT_DATA_DIR / "trades" / "evaluations.csv",
    prediction_log_path: Path = DEFAULT_LOG_PATH,
) -> StrategyAuditReport:
    mr_report = _read_csv(mean_reversion_report_path)
    closed_trades = _read_csv(closed_trades_path)
    evaluations = _read_csv(evaluations_path)
    prediction_log = load_prediction_log(prediction_log_path)
    truth = build_prediction_truth_report(prediction_log)

    combined = _combined_row(mr_report)
    total_closed = _int_value(combined, "total_trades") if combined else int(len(closed_trades))
    expectancy = _float_value(combined, "expectancy_r") if combined else _mean_numeric(closed_trades, "r_multiple")
    win_rate = _float_value(combined, "win_rate") if combined else _win_rate_from_trades(closed_trades)
    net_pnl = _float_value(combined, "net_pnl") if combined else _sum_numeric(closed_trades, "net_pnl")
    prediction_win_rate = float(truth.get("win_rate", 0.0))
    evaluated_signals = int(truth.get("evaluated", 0))
    total_saved = int(truth.get("total", 0))

    score = 50.0
    score += min(20.0, max(-20.0, expectancy * 10.0))
    score += (win_rate - 0.5) * 20.0 if total_closed else 0.0
    score += (prediction_win_rate - 0.5) * 10.0 if evaluated_signals else 0.0
    if total_closed < 20:
        score -= 15.0
    if evaluated_signals < 20:
        score -= 8.0
    score = round(max(0.0, min(100.0, score)), 2)

    best_asset = _best_asset(mr_report)
    top_blocker = _top_rejection_reason(mr_report, evaluations)
    insufficient = total_closed < 20 or evaluated_signals < 20
    grade = _grade(score, insufficient)
    rows = _audit_rows(mr_report, truth, total_closed, expectancy, win_rate, net_pnl, top_blocker)
    notes = _audit_notes(grade, total_closed, evaluated_signals, insufficient, top_blocker)

    return StrategyAuditReport(
        grade=grade,
        score=score,
        total_closed_trades=total_closed,
        total_saved_signals=total_saved,
        evaluated_signals=evaluated_signals,
        best_asset=best_asset,
        top_blocker=top_blocker,
        insufficient_sample=insufficient,
        notes=notes,
        rows=rows,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _combined_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty or "asset" not in frame.columns:
        return None
    combined = frame[frame["asset"].astype(str).str.upper() == "COMBINED"]
    if combined.empty:
        return None
    return combined.iloc[0].to_dict()


def _float_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _int_value(row: dict[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _win_rate_from_trades(frame: pd.DataFrame) -> float:
    if frame.empty or "net_pnl" not in frame.columns:
        return 0.0
    pnl = pd.to_numeric(frame["net_pnl"], errors="coerce").dropna()
    return float((pnl > 0).mean()) if not pnl.empty else 0.0


def _best_asset(frame: pd.DataFrame) -> str:
    if frame.empty or "asset" not in frame.columns:
        return "none"
    candidates = frame[frame["asset"].astype(str).str.upper() != "COMBINED"].copy()
    if candidates.empty:
        return "none"
    candidates["total_trades_num"] = pd.to_numeric(candidates.get("total_trades", 0), errors="coerce").fillna(0)
    candidates["expectancy_num"] = pd.to_numeric(candidates.get("expectancy_r", 0), errors="coerce").fillna(0)
    traded = candidates[candidates["total_trades_num"] > 0]
    if traded.empty:
        return "none"
    best = traded.sort_values(["expectancy_num", "total_trades_num"], ascending=False).iloc[0]
    return str(best.get("asset", "none"))


def _top_rejection_reason(report: pd.DataFrame, evaluations: pd.DataFrame) -> str:
    counts: dict[str, int] = {}
    if not report.empty and "rejected_setup_counts" in report.columns:
        for raw in report["rejected_setup_counts"].fillna(""):
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
    if not counts:
        return "none"
    key, value = max(counts.items(), key=lambda item: item[1])
    return f"{key} ({value})"


def _grade(score: float, insufficient: bool) -> str:
    if insufficient:
        return "INSUFFICIENT_SAMPLE"
    if score >= 75:
        return "STRONG_EVIDENCE"
    if score >= 60:
        return "WORKABLE_EDGE"
    if score >= 45:
        return "MIXED_EVIDENCE"
    return "WEAK_OR_UNPROVEN"


def _audit_rows(
    mean_reversion_report: pd.DataFrame,
    truth: dict[str, Any],
    total_closed: int,
    expectancy: float,
    win_rate: float,
    net_pnl: float,
    top_blocker: str,
) -> list[dict[str, Any]]:
    rows = [
        {"Layer": "Mean Reversion", "Metric": "Closed Trades", "Reading": total_closed, "Meaning": "Paper trades with final outcomes."},
        {"Layer": "Mean Reversion", "Metric": "Win Rate", "Reading": f"{win_rate:.0%}", "Meaning": "Wins divided by scored paper trades."},
        {"Layer": "Mean Reversion", "Metric": "Expectancy", "Reading": f"{expectancy:.2f}R", "Meaning": "Average R result per closed paper trade."},
        {"Layer": "Mean Reversion", "Metric": "Net P/L", "Reading": f"{net_pnl:.2f}", "Meaning": "Total paper profit/loss after fees and slippage."},
        {"Layer": "Mean Reversion", "Metric": "Top Blocker", "Reading": top_blocker, "Meaning": "Most common reason setups are rejected."},
        {"Layer": "Prediction Lab", "Metric": "Saved Signals", "Reading": int(truth.get("total", 0)), "Meaning": "NERO verdicts stored for truth checking."},
        {"Layer": "Prediction Lab", "Metric": "Evaluated Signals", "Reading": int(truth.get("evaluated", 0)), "Meaning": "Signals whose forward outcome has been measured."},
        {"Layer": "Prediction Lab", "Metric": "Win Rate", "Reading": f"{float(truth.get('win_rate', 0.0)):.0%}", "Meaning": "Directional verdict accuracy among scored signals."},
    ]
    if not mean_reversion_report.empty and "asset" in mean_reversion_report.columns:
        for _, row in mean_reversion_report.iterrows():
            asset = str(row.get("asset", ""))
            if asset.upper() == "COMBINED":
                continue
            rows.append(
                {
                    "Layer": "Asset Breakdown",
                    "Metric": asset,
                    "Reading": f"trades={_int_value(row.to_dict(), 'total_trades')}, expectancy={_float_value(row.to_dict(), 'expectancy_r'):.2f}R",
                    "Meaning": "Per-asset mean-reversion paper performance.",
                }
            )
    return rows


def _audit_notes(grade: str, total_closed: int, evaluated_signals: int, insufficient: bool, top_blocker: str) -> list[str]:
    notes = [f"Strategy audit grade: {grade}."]
    if insufficient:
        notes.append("Sample is still too small for commercial confidence; collect at least 20-30 closed trades/signals before trusting ratios.")
    if top_blocker != "none":
        notes.append(f"Most common blocker is {top_blocker}; this explains why NERO is selective and not over-trading.")
    notes.append(f"Evidence base: {total_closed} closed paper trade(s), {evaluated_signals} evaluated prediction signal(s).")
    return notes
