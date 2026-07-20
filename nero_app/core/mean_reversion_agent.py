from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
from typing import Any

import pandas as pd
import requests

from nero_app.core.market_data import MarketDataClient


STRATEGY_VERSION = "mean-reversion-v1.0.0"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mean_reversion"
DEFAULT_REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class MeanReversionConfig:
    strategy_version: str = STRATEGY_VERSION
    assets: dict[str, str] = field(default_factory=lambda: {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "NEAR": "NEARUSDT", "BNB": "BNBUSDT", "PAXG": "PAXGUSDT"})
    interval: str = "1h"
    initial_equity: float = 10000.0
    risk_per_trade: float = 0.01
    daily_loss_guard_r: float = -3.0
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    max_notional_pct: float = 1.0
    rsi_period: int = 14
    rsi_entry_below: float = 35.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    ma200_period: int = 200
    atr_period: int = 14
    atr_stop_multiple: float = 1.5
    max_holding_hours: int = 24
    candle_limit: int = 260
    stale_after_minutes: int = 180


@dataclass(frozen=True)
class AgentRunSummary:
    evaluated: int
    entries: int
    exits: int
    alerts: list[str]
    missed_runs: int


class MeanReversionAgent:
    def __init__(
        self,
        config: MeanReversionConfig | None = None,
        data_dir: Path = DEFAULT_DATA_DIR,
        report_dir: Path = DEFAULT_REPORT_DIR,
        now: datetime | None = None,
    ) -> None:
        self.config = config or MeanReversionConfig()
        self.data_dir = Path(data_dir)
        self.report_dir = Path(report_dir)
        self.now = now or datetime.now(timezone.utc)
        self.state_dir = self.data_dir / "state"
        self.trade_dir = self.data_dir / "trades"
        self.heartbeat_dir = self.data_dir / "heartbeats"
        for directory in [self.state_dir, self.trade_dir, self.heartbeat_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def run(self, assets: list[str] | None = None) -> AgentRunSummary:
        selected_assets = assets or list(self.config.assets.keys())
        evaluated = 0
        entries = 0
        exits = 0
        alerts: list[str] = []
        missed_runs = 0
        heartbeat_rows: list[dict[str, Any]] = []

        for asset in selected_assets:
            symbol = self.config.assets.get(asset, asset)
            state = self._load_state(asset)
            try:
                candles = self.fetch_closed_candles(asset, symbol)
                stale = self._is_stale(candles)
                missed = self._missed_run_count(state, candles)
                missed_runs += missed
                if stale:
                    self._append_error(asset, "STALE_FEED", f"Latest closed candle is stale for {symbol}")
                    alerts.append(f"{asset}: stale feed")
                    state["missed_run_count"] = int(state.get("missed_run_count", 0)) + 1
                    self._save_state(asset, state)
                    continue
                result = self.process_asset(asset, symbol, candles, state)
                evaluated += result["evaluated"]
                entries += result["entries"]
                exits += result["exits"]
                alerts.extend(result["alerts"])
                state = result["state"]
                state["missed_run_count"] = int(state.get("missed_run_count", 0)) + missed
                self._save_state(asset, state)
            except Exception as exc:  # noqa: BLE001 - runtime audit should catch all failures.
                self._append_error(asset, "ERROR", f"{exc.__class__.__name__}: {exc}")
                alerts.append(f"{asset}: ERROR {exc.__class__.__name__}")
            heartbeat_rows.append(self._heartbeat_row(asset, symbol, state))

        self._append_rows(self.heartbeat_dir / "heartbeats.csv", heartbeat_rows)
        self.write_report()
        return AgentRunSummary(evaluated=evaluated, entries=entries, exits=exits, alerts=alerts, missed_runs=missed_runs)

    def fetch_closed_candles(self, asset: str, symbol: str) -> pd.DataFrame:
        try:
            return self._fetch_binance_closed_candles(symbol)
        except (requests.RequestException, KeyError, TypeError, ValueError):
            return self._fetch_fallback_closed_candles(asset)

    def _fetch_binance_closed_candles(self, symbol: str) -> pd.DataFrame:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": self.config.interval, "limit": self.config.candle_limit},
            timeout=15,
        )
        response.raise_for_status()
        frame = pd.DataFrame(
            response.json(),
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )
        numeric = ["open", "high", "low", "close", "volume"]
        frame[numeric] = frame[numeric].astype(float)
        frame["open_time"] = frame["open_time"].astype("int64")
        frame["close_time"] = frame["close_time"].astype("int64")
        now_ms = int(self.now.timestamp() * 1000)
        frame = frame[frame["close_time"] < now_ms].copy()
        frame["date"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        return frame[["date", "open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def _fetch_fallback_closed_candles(self, asset: str) -> pd.DataFrame:
        result = MarketDataClient(timeout_seconds=15).load_intraday(
            asset=asset,
            prefer_live=True,
            interval=self.config.interval,
            candles=self.config.candle_limit,
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        )
        if result.status != "live":
            raise ValueError(f"No live fallback candles for {asset}: {result.status}")
        frame = result.prices.copy().sort_values("date").reset_index(drop=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["close_time"] = (frame["date"].astype("int64") // 1_000_000).astype("int64")
        frame["open_time"] = frame["close_time"] - _interval_milliseconds(self.config.interval)
        now_ms = int(self.now.timestamp() * 1000)
        frame = frame[frame["close_time"] < now_ms].copy()
        return frame[["date", "open_time", "close_time", "open", "high", "low", "close", "volume"]].tail(self.config.candle_limit).reset_index(drop=True)

    def process_asset(self, asset: str, symbol: str, candles: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
        enriched = add_indicators(candles, self.config)
        last_seen = int(state.get("last_evaluated_close_time", 0))
        rows = enriched[enriched["close_time"] > last_seen].copy()
        rows = rows.dropna(subset=["rsi", "bb_lower", "ma20", "ma200", "atr"])
        rows = rows.sort_values("close_time")
        entries = 0
        exits = 0
        evaluated = 0
        alerts: list[str] = []

        for _, candle in rows.iterrows():
            candle_time = int(candle["close_time"])
            state = self._reset_daily_guard_if_needed(state, candle)
            exit_event = self._maybe_exit(asset, symbol, candle, state)
            if exit_event:
                exits += 1
                alerts.append(f"{asset}: {exit_event['exit_reason']} net={exit_event['net_pnl']:.2f} R={exit_event['r_multiple']:.2f}")

            evaluation = self._evaluate_entry(asset, symbol, candle, state)
            self._append_rows(self.trade_dir / "evaluations.csv", [evaluation])
            evaluated += 1
            if evaluation["passed"]:
                entry_event = self._enter_trade(asset, symbol, candle, state)
                if entry_event:
                    entries += 1
                    alerts.append(f"{asset}: MEAN_REVERSION_ENTRY entry={entry_event['entry_price']:.4f} target={entry_event['target']:.4f}")
            state["last_evaluated_close_time"] = candle_time

        return {"state": state, "evaluated": evaluated, "entries": entries, "exits": exits, "alerts": alerts}

    def _evaluate_entry(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        open_trade = state.get("open_trade")
        if open_trade:
            reasons.append("OPEN_TRADE_EXISTS")
        if float(state.get("daily_r", 0.0)) <= self.config.daily_loss_guard_r:
            reasons.append("DAILY_LOSS_GUARD")
        if float(candle["rsi"]) >= self.config.rsi_entry_below:
            reasons.append("RSI_NOT_BELOW_35")
        if float(candle["close"]) >= float(candle["bb_lower"]):
            reasons.append("CLOSE_NOT_BELOW_LOWER_BB")
        if float(candle["close"]) <= float(candle["ma200"]):
            reasons.append("CLOSE_NOT_ABOVE_MA200")
        if float(candle["ma20"]) <= float(candle["close"]):
            reasons.append("TARGET_NOT_ABOVE_ENTRY")

        passed = not reasons
        return {
            "timestamp": self.now.isoformat(),
            "asset": asset,
            "symbol": symbol,
            "strategy_version": self.config.strategy_version,
            "candle_close_time": int(candle["close_time"]),
            "candle_time": candle["date"].isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "rsi": float(candle["rsi"]),
            "ma20": float(candle["ma20"]),
            "bb_lower": float(candle["bb_lower"]),
            "ma200": float(candle["ma200"]),
            "atr": float(candle["atr"]),
            "passed": passed,
            "rejection_reasons": "|".join(reasons),
        }

    def _enter_trade(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("open_trade"):
            return None
        equity = float(state.get("equity", self.config.initial_equity))
        raw_entry = float(candle["close"])
        entry_price = apply_slippage(raw_entry, self.config.slippage_bps, "buy")
        stop_loss = entry_price - self.config.atr_stop_multiple * float(candle["atr"])
        target = float(candle["ma20"])
        risk_per_unit = entry_price - stop_loss
        reward_per_unit = target - entry_price
        if risk_per_unit <= 0 or reward_per_unit <= 0:
            return None
        risk_dollars = equity * self.config.risk_per_trade
        quantity = risk_dollars / risk_per_unit
        max_notional = equity * self.config.max_notional_pct
        notional = quantity * entry_price
        if notional > max_notional:
            quantity = max_notional / entry_price
            notional = max_notional
            risk_dollars = quantity * risk_per_unit
        fees = notional * self.config.fee_bps / 10000.0
        trade_id = f"MR-{asset}-{int(candle['close_time'])}"
        trade = {
            "trade_id": trade_id,
            "asset": asset,
            "symbol": symbol,
            "strategy_version": self.config.strategy_version,
            "status": "OPEN",
            "opened_at": candle["date"].isoformat(),
            "open_close_time": int(candle["close_time"]),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "target_mode": "FROZEN_MA20",
            "quantity": quantity,
            "notional": notional,
            "risk_dollars": risk_dollars,
            "entry_fee": fees,
            "entry_slippage_bps": self.config.slippage_bps,
            "planned_reward_r": reward_per_unit / risk_per_unit,
            "entry_rsi": float(candle["rsi"]),
            "entry_ma20": float(candle["ma20"]),
            "entry_bb_lower": float(candle["bb_lower"]),
            "entry_ma200": float(candle["ma200"]),
            "entry_atr": float(candle["atr"]),
        }
        state["open_trade"] = trade
        self._append_rows(self.trade_dir / "trade_events.csv", [{**trade, "event": "ENTRY", "timestamp": self.now.isoformat()}])
        return trade

    def _maybe_exit(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        trade = state.get("open_trade")
        if not trade:
            return None
        candle_time = int(candle["close_time"])
        hours_held = (candle_time - int(trade["open_close_time"])) / 3600000.0
        stop_loss = float(trade["stop_loss"])
        target = float(trade["target"])
        low = float(candle["low"])
        high = float(candle["high"])
        close = float(candle["close"])
        exit_reason = ""
        raw_exit = close
        if low <= stop_loss and high >= target:
            exit_reason = "SL"
            raw_exit = stop_loss
        elif low <= stop_loss:
            exit_reason = "SL"
            raw_exit = stop_loss
        elif high >= target:
            exit_reason = "TARGET"
            raw_exit = target
        elif hours_held >= self.config.max_holding_hours:
            exit_reason = "TIME"
            raw_exit = close
        else:
            return None

        exit_price = apply_slippage(raw_exit, self.config.slippage_bps, "sell")
        quantity = float(trade["quantity"])
        gross_pnl = (exit_price - float(trade["entry_price"])) * quantity
        exit_fee = exit_price * quantity * self.config.fee_bps / 10000.0
        total_fees = float(trade["entry_fee"]) + exit_fee
        net_pnl = gross_pnl - total_fees
        risk_dollars = max(float(trade["risk_dollars"]), 1e-9)
        r_multiple = net_pnl / risk_dollars
        equity = float(state.get("equity", self.config.initial_equity)) + net_pnl
        state["equity"] = equity
        state["daily_r"] = float(state.get("daily_r", 0.0)) + r_multiple
        state["open_trade"] = None
        event = {
            **trade,
            "event": "EXIT",
            "timestamp": self.now.isoformat(),
            "status": "CLOSED",
            "closed_at": candle["date"].isoformat(),
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "exit_fee": exit_fee,
            "fees": total_fees,
            "slippage_bps": self.config.slippage_bps,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
            "holding_hours": hours_held,
        }
        self._append_rows(self.trade_dir / "trade_events.csv", [event])
        self._append_rows(self.trade_dir / "closed_trades.csv", [event])
        return event

    def write_report(self) -> None:
        trades_path = self.trade_dir / "closed_trades.csv"
        evaluations_path = self.trade_dir / "evaluations.csv"
        trades = pd.read_csv(trades_path) if trades_path.exists() and trades_path.stat().st_size else pd.DataFrame()
        evaluations = pd.read_csv(evaluations_path) if evaluations_path.exists() and evaluations_path.stat().st_size else pd.DataFrame()
        rows = []
        assets = sorted(set(self.config.assets.keys()) | set(trades["asset"].unique() if not trades.empty else []))
        for asset in assets:
            rows.append(report_row(asset, trades[trades["asset"] == asset] if not trades.empty else pd.DataFrame(), evaluations))
        rows.append(report_row("COMBINED", trades, evaluations))
        report = pd.DataFrame(rows)
        report.to_csv(self.report_dir / "mean_reversion_report.csv", index=False)
        (self.report_dir / "mean_reversion_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _load_state(self, asset: str) -> dict[str, Any]:
        path = self.state_dir / f"{asset}.json"
        if not path.exists():
            return {"equity": self.config.initial_equity, "daily_r": 0.0, "open_trade": None, "strategy_version": self.config.strategy_version}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state must be object")
            data.setdefault("equity", self.config.initial_equity)
            data.setdefault("daily_r", 0.0)
            data.setdefault("open_trade", None)
            data.setdefault("strategy_version", self.config.strategy_version)
            return data
        except (json.JSONDecodeError, ValueError):
            return {"equity": self.config.initial_equity, "daily_r": 0.0, "open_trade": None, "strategy_version": self.config.strategy_version}

    def _save_state(self, asset: str, state: dict[str, Any]) -> None:
        (self.state_dir / f"{asset}.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _reset_daily_guard_if_needed(self, state: dict[str, Any], candle: pd.Series) -> dict[str, Any]:
        day = candle["date"].date().isoformat()
        if state.get("daily_guard_day") != day:
            state["daily_guard_day"] = day
            state["daily_r"] = 0.0
        return state

    def _is_stale(self, candles: pd.DataFrame) -> bool:
        if candles.empty:
            return True
        latest = pd.to_datetime(candles.iloc[-1]["date"]).to_pydatetime()
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age_minutes = (self.now - latest).total_seconds() / 60.0
        return age_minutes > self.config.stale_after_minutes

    def _missed_run_count(self, state: dict[str, Any], candles: pd.DataFrame) -> int:
        last_seen = int(state.get("last_evaluated_close_time", 0))
        if not last_seen or candles.empty:
            return 0
        unseen = candles[candles["close_time"] > last_seen]
        return max(0, len(unseen) - 1)

    def _heartbeat_row(self, asset: str, symbol: str, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": self.now.isoformat(),
            "asset": asset,
            "symbol": symbol,
            "strategy_version": self.config.strategy_version,
            "equity": float(state.get("equity", self.config.initial_equity)),
            "open_trade": bool(state.get("open_trade")),
            "last_evaluated_close_time": int(state.get("last_evaluated_close_time", 0)),
            "missed_run_count": int(state.get("missed_run_count", 0)),
        }

    def _append_error(self, asset: str, code: str, message: str) -> None:
        self._append_rows(self.trade_dir / "runtime_errors.csv", [{"timestamp": self.now.isoformat(), "asset": asset, "code": code, "message": message}])

    @staticmethod
    def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_fields: list[str] = []
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                existing_fields = [field for field in next(reader, []) if field]
        fields = list(dict.fromkeys(existing_fields))
        for row in rows:
            for key in row.keys():
                if key and key not in fields:
                    fields.append(key)
        rewrite = bool(existing_fields) and fields != existing_fields
        existing_rows: list[dict[str, Any]] = []
        if rewrite:
            with path.open("r", newline="", encoding="utf-8") as handle:
                existing_rows = [
                    {key: value for key, value in row.items() if key in fields}
                    for row in csv.DictReader(handle)
                ]
        mode = "w" if rewrite or not existing_fields else "a"
        with path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
                writer.writerows(existing_rows)
            writer.writerows(rows)


def add_indicators(candles: pd.DataFrame, config: MeanReversionConfig) -> pd.DataFrame:
    frame = candles.copy().sort_values("close_time").reset_index(drop=True)
    close = frame["close"].astype(float)
    frame["ma20"] = close.rolling(config.bollinger_period).mean()
    frame["bb_std"] = close.rolling(config.bollinger_period).std(ddof=0)
    frame["bb_lower"] = frame["ma20"] - config.bollinger_std * frame["bb_std"]
    frame["ma200"] = close.rolling(config.ma200_period).mean()
    frame["rsi"] = rsi(close, config.rsi_period)
    frame["atr"] = atr(frame, config.atr_period)
    return frame


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, math.nan)
    values = 100.0 - (100.0 / (1.0 + rs))
    return values.fillna(100.0)


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(period).mean()



def _interval_milliseconds(interval: str) -> int:
    return {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }.get(interval, 3_600_000)

def apply_slippage(price: float, slippage_bps: float, side: str) -> float:
    factor = slippage_bps / 10000.0
    return price * (1.0 + factor) if side == "buy" else price * (1.0 - factor)


def report_row(asset: str, trades: pd.DataFrame, evaluations: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        rejected = rejected_counts(evaluations, asset)
        return {
            "asset": asset,
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_holding_hours": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "fees_paid": 0.0,
            "slippage_paid_estimate": 0.0,
            "insufficient_sample": True,
            "rejected_setup_counts": json.dumps(rejected, sort_keys=True),
        }
    r_values = trades["r_multiple"].astype(float)
    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    equity = trades["equity_after"].astype(float).tolist() if "equity_after" in trades else []
    return {
        "asset": asset,
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy_r": float(r_values.mean()),
        "profit_factor": gross_win / gross_loss if gross_loss else (gross_win if gross_win else 0.0),
        "max_drawdown": max_drawdown(equity),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "average_holding_hours": float(trades["holding_hours"].astype(float).mean()) if "holding_hours" in trades else 0.0,
        "consecutive_wins": longest_streak((pnl > 0).tolist(), True),
        "consecutive_losses": longest_streak((pnl < 0).tolist(), True),
        "fees_paid": float(trades["fees"].astype(float).sum()) if "fees" in trades else 0.0,
        "slippage_paid_estimate": estimate_slippage_paid(trades),
        "insufficient_sample": len(trades) < 20,
        "rejected_setup_counts": json.dumps(rejected_counts(evaluations, asset), sort_keys=True),
    }


def rejected_counts(evaluations: pd.DataFrame, asset: str) -> dict[str, int]:
    if evaluations.empty or "rejection_reasons" not in evaluations:
        return {}
    frame = evaluations if asset == "COMBINED" else evaluations[evaluations["asset"] == asset]
    counts: dict[str, int] = {}
    for value in frame["rejection_reasons"].fillna(""):
        for reason in str(value).split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value - peak) / peak)
    return float(drawdown)


def longest_streak(values: list[bool], target: bool) -> int:
    best = 0
    current = 0
    for value in values:
        if value is target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def estimate_slippage_paid(trades: pd.DataFrame) -> float:
    if trades.empty or "notional" not in trades:
        return 0.0
    bps = trades["slippage_bps"].astype(float) if "slippage_bps" in trades else pd.Series([0.0] * len(trades))
    notional = trades["notional"].astype(float)
    return float((notional * bps / 10000.0 * 2.0).sum())


def load_assets_from_env(default: dict[str, str] | None = None) -> dict[str, str]:
    default_assets = default or MeanReversionConfig().assets
    raw = os.getenv("MR_ASSETS", "").strip()
    if not raw:
        return default_assets
    assets: dict[str, str] = {}
    for item in raw.split(","):
        clean = item.strip()
        if not clean:
            continue
        if ":" in clean:
            name, symbol = clean.split(":", 1)
            assets[name.strip().upper()] = symbol.strip().upper()
        else:
            symbol = clean.upper()
            name = symbol.replace("USDT", "")
            assets[name] = symbol
    return assets

