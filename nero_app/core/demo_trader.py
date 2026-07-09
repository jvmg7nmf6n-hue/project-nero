from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nero_app.core.trade_desk import IntradayTradePlan


DEFAULT_DEMO_TRADE_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_trades.csv"
DEMO_TRADE_COLUMNS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "asset",
    "side",
    "status",
    "entry",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "exit_price",
    "exit_reason",
    "result",
    "r_multiple",
    "confidence",
    "source",
    "notes",
]


@dataclass(frozen=True)
class DemoTradeSummary:
    opened: int
    activated: int
    closed: int
    pending_trades: int
    open_trades: int
    closed_trades: int
    win_rate: float
    expectancy_r: float


def run_demo_trader(
    asset: str,
    plan: IntradayTradePlan,
    prices: pd.DataFrame,
    source: str,
    path: Path = DEFAULT_DEMO_TRADE_PATH,
    max_age_candles: int = 24,
) -> DemoTradeSummary:
    frame = load_demo_trades(path)
    frame, activated = _activate_pending_trades(frame, prices)
    frame, closed = _close_open_trades(frame, prices, max_age_candles=max_age_candles)
    opened = 0
    if _should_record_setup(frame, asset, plan, prices):
        frame = _append_trade(frame, asset, plan, prices, source)
        opened = 1
    _save_demo_trades(frame, path)
    return _summarize(frame, opened=opened, activated=activated, closed=closed)


def load_demo_trades(path: Path = DEFAULT_DEMO_TRADE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DEMO_TRADE_COLUMNS)

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=DEMO_TRADE_COLUMNS)
        for raw_row in reader:
            row = {column: "" for column in DEMO_TRADE_COLUMNS}
            for idx, value in enumerate(raw_row):
                if idx < len(header) and header[idx] in DEMO_TRADE_COLUMNS:
                    row[header[idx]] = value
            rows.append(row)
    return pd.DataFrame(rows, columns=DEMO_TRADE_COLUMNS)


def accountability_scorecard(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"total": 0, "pending": 0, "closed": 0, "open": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "expectancy_r": 0.0}
    closed = frame[frame["status"] == "closed"].copy()
    wins = int((closed["result"] == "win").sum()) if not closed.empty else 0
    losses = int((closed["result"] == "loss").sum()) if not closed.empty else 0
    r_values = pd.to_numeric(closed["r_multiple"], errors="coerce").fillna(0.0) if not closed.empty else pd.Series(dtype=float)
    return {
        "total": int(len(frame)),
        "pending": int((frame["status"] == "pending").sum()),
        "closed": int(len(closed)),
        "open": int((frame["status"] == "open").sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(closed) if len(closed) else 0.0,
        "expectancy_r": float(r_values.mean()) if len(r_values) else 0.0,
    }


def _activate_pending_trades(frame: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or prices.empty:
        return frame, 0
    frame = frame.astype(object)
    price_frame = prices.sort_values("date").copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    activated = 0

    for index, row in frame[frame["status"] == "pending"].iterrows():
        signal_at = pd.to_datetime(row["opened_at"], errors="coerce")
        if pd.isna(signal_at):
            continue
        future = price_frame[price_frame["date"] >= signal_at]
        if future.empty:
            continue
        side = str(row["side"]).upper()
        entry = float(row["entry"])
        for _, candle in future.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            if (side == "LONG" and high >= entry) or (side == "SHORT" and low <= entry):
                frame.loc[index, "status"] = "open"
                frame.loc[index, "opened_at"] = pd.to_datetime(candle["date"]).isoformat()
                frame.loc[index, "notes"] = f"Triggered from pending setup at {entry:,.4f}"
                activated += 1
                break

    return frame, activated


def _close_open_trades(frame: pd.DataFrame, prices: pd.DataFrame, max_age_candles: int) -> tuple[pd.DataFrame, int]:
    if frame.empty or prices.empty:
        return frame, 0
    frame = frame.astype(object)
    price_frame = prices.sort_values("date").copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    closed = 0

    for index, row in frame[frame["status"] == "open"].iterrows():
        opened_at = pd.to_datetime(row["opened_at"], errors="coerce")
        if pd.isna(opened_at):
            continue
        future = price_frame[price_frame["date"] > opened_at]
        if future.empty:
            continue

        side = str(row["side"]).upper()
        entry = float(row["entry"])
        stop = float(row["stop_loss"])
        target = float(row["take_profit_1"])
        exit_price = 0.0
        exit_reason = ""
        exit_date = None
        result = ""

        for _, candle in future.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            if side == "LONG":
                if low <= stop:
                    exit_price, exit_reason, result = stop, "SL", "loss"
                elif high >= target:
                    exit_price, exit_reason, result = target, "TP1", "win"
            elif side == "SHORT":
                if high >= stop:
                    exit_price, exit_reason, result = stop, "SL", "loss"
                elif low <= target:
                    exit_price, exit_reason, result = target, "TP1", "win"
            if exit_reason:
                exit_date = candle["date"]
                break

        if not exit_reason and len(future) >= max_age_candles:
            final = future.iloc[min(max_age_candles, len(future)) - 1]
            exit_price = float(final["close"])
            exit_reason = "EXPIRED"
            exit_date = final["date"]
            result = _expiry_result(side, entry, exit_price)

        if exit_reason and exit_date is not None:
            frame.loc[index, "status"] = "closed"
            frame.loc[index, "closed_at"] = pd.to_datetime(exit_date).isoformat()
            frame.loc[index, "exit_price"] = round(exit_price, 4)
            frame.loc[index, "exit_reason"] = exit_reason
            frame.loc[index, "result"] = result
            frame.loc[index, "r_multiple"] = round(_r_multiple(side, entry, stop, exit_price), 2)
            closed += 1

    return frame, closed


def _should_record_setup(frame: pd.DataFrame, asset: str, plan: IntradayTradePlan, prices: pd.DataFrame) -> bool:
    if prices.empty or plan.action not in {"WAIT_LONG_TRIGGER", "WAIT_SHORT_TRIGGER"}:
        return False
    if frame.empty:
        return True
    active_for_asset = frame[(frame["asset"] == asset) & (frame["status"].isin(["pending", "open"]))]
    return active_for_asset.empty


def _trigger_touched(plan: IntradayTradePlan, prices: pd.DataFrame) -> bool:
    if prices.empty:
        return False
    latest = prices.sort_values("date").iloc[-1]
    if plan.action == "WAIT_LONG_TRIGGER":
        return float(latest["high"]) >= float(plan.entry_price)
    if plan.action == "WAIT_SHORT_TRIGGER":
        return float(latest["low"]) <= float(plan.entry_price)
    return False


def _append_trade(
    frame: pd.DataFrame,
    asset: str,
    plan: IntradayTradePlan,
    prices: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    latest = prices.sort_values("date").iloc[-1]
    opened_at = pd.to_datetime(latest["date"]).isoformat()
    side = "LONG" if plan.action == "WAIT_LONG_TRIGGER" else "SHORT"
    trade_id = f"{asset}-{side}-{opened_at}"
    status = "open" if _trigger_touched(plan, prices) else "pending"
    row = {
        "trade_id": trade_id,
        "opened_at": opened_at,
        "closed_at": "",
        "asset": asset,
        "side": side,
        "status": status,
        "entry": plan.entry_price,
        "stop_loss": plan.stop_loss,
        "take_profit_1": plan.take_profit_1,
        "take_profit_2": plan.take_profit_2,
        "exit_price": "",
        "exit_reason": "",
        "result": "",
        "r_multiple": "",
        "confidence": plan.confidence,
        "source": source,
        "notes": plan.entry_trigger if status == "open" else f"Pending trigger: {plan.entry_trigger}",
    }
    return pd.concat([frame, pd.DataFrame([row], columns=DEMO_TRADE_COLUMNS)], ignore_index=True)


def _save_demo_trades(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, columns=DEMO_TRADE_COLUMNS, index=False)


def _summarize(frame: pd.DataFrame, opened: int, activated: int, closed: int) -> DemoTradeSummary:
    scorecard = accountability_scorecard(frame)
    return DemoTradeSummary(
        opened=opened,
        activated=activated,
        closed=closed,
        pending_trades=int(scorecard["pending"]),
        open_trades=int(scorecard["open"]),
        closed_trades=int(scorecard["closed"]),
        win_rate=float(scorecard["win_rate"]),
        expectancy_r=float(scorecard["expectancy_r"]),
    )


def _expiry_result(side: str, entry: float, exit_price: float) -> str:
    if side == "LONG":
        return "win" if exit_price > entry else "loss"
    return "win" if exit_price < entry else "loss"


def _r_multiple(side: str, entry: float, stop: float, exit_price: float) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    if side == "LONG":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk
