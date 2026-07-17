from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nero_app.core.schema import NeroResult


DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "prediction_log.csv"
LOG_COLUMNS = [
    "timestamp",
    "asset",
    "headline",
    "direction",
    "confidence",
    "risk_score",
    "thematic_score",
    "momentum_score",
    "rsi",
    "fair_value_gap",
    "liquidity_sweep",
    "data_source",
    "entry_date",
    "entry_close",
    "horizon_days",
    "target_date",
    "evaluation_status",
    "exit_date",
    "exit_close",
    "actual_return",
    "outcome",
]


def append_prediction(
    result: NeroResult,
    data_source: str,
    prices: pd.DataFrame | None = None,
    horizon_days: int = 7,
    path: Path = DEFAULT_LOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry_date = ""
    entry_close = ""
    target_date = ""
    if prices is not None and not prices.empty:
        latest = prices.sort_values("date").iloc[-1]
        entry_timestamp = pd.to_datetime(latest["date"])
        entry_date = entry_timestamp.date().isoformat()
        entry_close = float(latest["close"])
        target_date = (entry_timestamp + timedelta(days=horizon_days)).date().isoformat()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "asset": result.request.asset.value,
        "headline": result.request.headline,
        "direction": result.verdict.direction,
        "confidence": result.verdict.confidence,
        "risk_score": result.verdict.risk_score,
        "thematic_score": result.brain.thematic_score,
        "momentum_score": result.assessment.momentum_score,
        "rsi": result.assessment.rsi,
        "fair_value_gap": result.assessment.fair_value_gap,
        "liquidity_sweep": result.assessment.liquidity_sweep,
        "data_source": data_source,
        "entry_date": entry_date,
        "entry_close": entry_close,
        "horizon_days": horizon_days,
        "target_date": target_date,
        "evaluation_status": "pending" if target_date else "not_evaluable",
        "exit_date": "",
        "exit_close": "",
        "actual_return": "",
        "outcome": "",
    }
    frame = pd.DataFrame([row], columns=LOG_COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_prediction_log(path: Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=LOG_COLUMNS)

        for raw_row in reader:
            if not raw_row:
                continue
            row = {column: "" for column in LOG_COLUMNS}
            for idx, value in enumerate(raw_row):
                if idx < len(header) and header[idx] in LOG_COLUMNS:
                    row[header[idx]] = value
                elif idx < len(LOG_COLUMNS):
                    row[LOG_COLUMNS[idx]] = value
            rows.append(row)

    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def evaluate_prediction_log(prices: pd.DataFrame, path: Path = DEFAULT_LOG_PATH, asset: str | None = None) -> pd.DataFrame:
    frame = load_prediction_log(path)
    if frame.empty or prices.empty:
        return frame
    asset_filter = asset.strip().upper() if asset else ""
    text_columns = ["entry_date", "target_date", "evaluation_status", "exit_date", "outcome"]
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str)
    for column in ["exit_close", "actual_return"]:
        frame[column] = frame[column].astype(object)

    price_frame = prices.sort_values("date").copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.date

    for index, row in frame.iterrows():
        if asset_filter and str(row.get("asset", "")).upper() != asset_filter:
            continue
        if str(row.get("evaluation_status", "")) not in {"pending", ""}:
            continue
        if not row.get("target_date") or not row.get("entry_close"):
            frame.loc[index, "evaluation_status"] = "not_evaluable"
            continue

        target_date = pd.to_datetime(row["target_date"]).date()
        eligible = price_frame[price_frame["date"] >= target_date]
        if eligible.empty:
            frame.loc[index, "evaluation_status"] = "pending"
            continue

        exit_row = eligible.iloc[0]
        entry_close = float(row["entry_close"])
        exit_close = float(exit_row["close"])
        actual_return = (exit_close - entry_close) / entry_close if entry_close else 0.0
        direction = str(row["direction"]).lower()
        if direction == "bullish":
            outcome = "win" if actual_return > 0 else "miss"
        elif direction == "bearish":
            outcome = "win" if actual_return < 0 else "miss"
        else:
            outcome = "neutral"

        frame.loc[index, "evaluation_status"] = "evaluated"
        frame.loc[index, "exit_date"] = exit_row["date"].isoformat()
        frame.loc[index, "exit_close"] = exit_close
        frame.loc[index, "actual_return"] = actual_return
        frame.loc[index, "outcome"] = outcome

    frame.to_csv(path, index=False)
    return frame


def build_prediction_truth_report(frame: pd.DataFrame, asset: str | None = None) -> dict[str, object]:
    if frame.empty:
        return {
            "total": 0,
            "evaluated": 0,
            "pending": 0,
            "wins": 0,
            "misses": 0,
            "neutral": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "high_confidence_win_rate": 0.0,
            "rows": [],
            "notes": ["No prediction records are available yet."],
        }

    report_frame = frame.copy()
    if asset:
        report_frame = report_frame[report_frame.get("asset", pd.Series(dtype=str)).astype(str).str.upper() == asset.upper()]
    if report_frame.empty:
        return {
            "total": 0,
            "evaluated": 0,
            "pending": 0,
            "wins": 0,
            "misses": 0,
            "neutral": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "high_confidence_win_rate": 0.0,
            "rows": [],
            "notes": [f"No prediction records found for {asset}."] if asset else ["No live prediction records found. Sample/fallback rows are excluded from scoring."],
        }

    report_frame["evaluation_status"] = report_frame.get("evaluation_status", pd.Series(dtype=str)).fillna("").astype(str)
    if "data_source" in report_frame:
        source = report_frame["data_source"].fillna("").astype(str).str.lower()
        live_mask = source.str.contains("live", na=False) & ~source.str.contains("generated sample|sample|fallback", na=False)
        report_frame = report_frame[live_mask].copy()
    report_frame["outcome"] = report_frame.get("outcome", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    report_frame["actual_return_num"] = pd.to_numeric(report_frame.get("actual_return", pd.Series(dtype=float)), errors="coerce")
    report_frame["confidence_num"] = pd.to_numeric(report_frame.get("confidence", pd.Series(dtype=float)), errors="coerce")

    evaluated = report_frame[report_frame["evaluation_status"] == "evaluated"]
    pending = report_frame[report_frame["evaluation_status"].isin(["pending", ""])]
    wins = int((evaluated["outcome"] == "win").sum()) if not evaluated.empty else 0
    misses = int((evaluated["outcome"] == "miss").sum()) if not evaluated.empty else 0
    neutral = int((evaluated["outcome"] == "neutral").sum()) if not evaluated.empty else 0
    scored = wins + misses
    win_rate = wins / scored if scored else 0.0
    average_return = float(evaluated["actual_return_num"].dropna().mean()) if not evaluated.empty and not evaluated["actual_return_num"].dropna().empty else 0.0
    high_conf = evaluated[evaluated["confidence_num"] >= 0.60]
    high_conf_scored = high_conf[high_conf["outcome"].isin(["win", "miss"])]
    high_conf_win_rate = float((high_conf_scored["outcome"] == "win").sum() / len(high_conf_scored)) if len(high_conf_scored) else 0.0

    rows: list[dict[str, object]] = []
    for asset_name, group in report_frame.groupby("asset", dropna=False):
        group_eval = group[group["evaluation_status"] == "evaluated"]
        group_wins = int((group_eval["outcome"] == "win").sum()) if not group_eval.empty else 0
        group_misses = int((group_eval["outcome"] == "miss").sum()) if not group_eval.empty else 0
        group_scored = group_wins + group_misses
        rows.append(
            {
                "Asset": asset_name,
                "Total": int(len(group)),
                "Evaluated": int(len(group_eval)),
                "Pending": int(group["evaluation_status"].isin(["pending", ""]).sum()),
                "Wins": group_wins,
                "Misses": group_misses,
                "Win Rate": f"{(group_wins / group_scored if group_scored else 0.0):.0%}",
                "Avg Return": f"{(group_eval['actual_return_num'].dropna().mean() if not group_eval.empty and not group_eval['actual_return_num'].dropna().empty else 0.0):.2%}",
                "Avg Confidence": f"{(group['confidence_num'].dropna().mean() if not group['confidence_num'].dropna().empty else 0.0):.0%}",
            }
        )

    notes = _prediction_truth_notes(scored, win_rate, high_conf_win_rate, len(pending))
    return {
        "total": int(len(report_frame)),
        "evaluated": int(len(evaluated)),
        "pending": int(len(pending)),
        "wins": wins,
        "misses": misses,
        "neutral": neutral,
        "win_rate": float(win_rate),
        "average_return": average_return,
        "high_confidence_win_rate": high_conf_win_rate,
        "rows": rows,
        "notes": notes,
    }


def _prediction_truth_notes(scored: int, win_rate: float, high_confidence_win_rate: float, pending: int) -> list[str]:
    notes: list[str] = []
    if scored < 20:
        notes.append("Insufficient sample: wait for at least 20-30 evaluated directional predictions before trusting accuracy.")
    if scored:
        notes.append(f"Directional prediction win rate is {win_rate:.0%} across {scored} evaluated win/miss calls.")
    if high_confidence_win_rate:
        notes.append(f"High-confidence prediction win rate is {high_confidence_win_rate:.0%}; compare this against the overall win rate.")
    if pending:
        notes.append(f"{pending} prediction(s) are still pending evaluation.")
    if not notes:
        notes.append("No evaluated directional predictions yet.")
    return notes


