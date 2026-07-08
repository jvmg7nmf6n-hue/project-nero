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


def evaluate_prediction_log(prices: pd.DataFrame, path: Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    frame = load_prediction_log(path)
    if frame.empty or prices.empty:
        return frame
    text_columns = ["entry_date", "target_date", "evaluation_status", "exit_date", "outcome"]
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str)
    for column in ["exit_close", "actual_return"]:
        frame[column] = frame[column].astype(object)

    price_frame = prices.sort_values("date").copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.date

    for index, row in frame.iterrows():
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
