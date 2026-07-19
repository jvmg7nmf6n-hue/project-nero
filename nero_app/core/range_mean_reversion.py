"""Range-aware mean reversion research engine.

Paper/backtest only. It tests whether Bollinger Band extremes add edge after
filtering for a ranging regime with ADX.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import pandas as pd


RANGE_MR_VERSION = "range-mean-reversion-v1.0.0"


@dataclass(frozen=True)
class RangeMRConfig:
    version: str = RANGE_MR_VERSION
    initial_equity: float = 10000.0
    risk_per_trade: float = 0.01
    fee_bps_crypto: float = 10.0
    fee_bps_forex: float = 2.0
    fee_bps_market: float = 5.0
    slippage_bps: float = 2.0
    adx_period: int = 14
    adx_entry_below: float = 25.0
    adx_exit_at: float = 28.0
    adx_exit_bars: int = 2
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    min_completed_rows: int = 80
    min_train_test_trades: int = 20
    random_seed: int = 3039


@dataclass(frozen=True)
class RangeMRTrade:
    asset: str
    timeframe: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_loss: float
    exit_reason: str
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    r_multiple: float
    equity_after: float
    holding_bars: int
    entry_adx: float
    entry_band_width_pct: float
    strategy_version: str


def add_range_mr_indicators(prices: pd.DataFrame, config: RangeMRConfig | None = None) -> pd.DataFrame:
    cfg = config or RangeMRConfig()
    frame = _clean_ohlc(prices)
    close = frame["close"]
    ma = close.rolling(cfg.bollinger_period).mean()
    std = close.rolling(cfg.bollinger_period).std(ddof=0)
    frame["ma20"] = ma
    frame["bb_upper"] = ma + cfg.bollinger_std * std
    frame["bb_lower"] = ma - cfg.bollinger_std * std
    frame["bb_width_pct"] = ((frame["bb_upper"] - frame["bb_lower"]) / close.replace(0, math.nan)) * 100.0
    frame["atr"] = average_true_range(frame, cfg.atr_period)
    frame["adx"] = average_directional_index(frame, cfg.adx_period)
    frame["close_time"] = _close_time_ms(frame["date"])
    return frame


def average_true_range(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    frame = _clean_ohlc(prices)
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1).rolling(period).mean()


def average_directional_index(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    frame = _clean_ohlc(prices)
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = average_true_range(frame, period)
    plus_di = 100.0 * plus_dm.rolling(period).mean() / atr.replace(0, math.nan)
    minus_di = 100.0 * minus_dm.rolling(period).mean() / atr.replace(0, math.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)
    return dx.rolling(period).mean()


def run_range_mean_reversion_backtest(
    prices: pd.DataFrame,
    asset: str,
    timeframe: str,
    config: RangeMRConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or RangeMRConfig()
    enriched = add_range_mr_indicators(prices, cfg)
    enriched = enriched.dropna(subset=["ma20", "bb_upper", "bb_lower", "atr", "adx"]).reset_index(drop=True)
    evaluations: list[dict[str, Any]] = []
    trades: list[RangeMRTrade] = []
    equity = cfg.initial_equity
    open_trade: dict[str, Any] | None = None
    adx_break_count = 0

    for idx, candle in enriched.iterrows():
        if open_trade is not None and idx > int(open_trade["entry_index"]):
            exit_event = _maybe_exit(open_trade, candle, idx, asset, timeframe, equity, cfg, adx_break_count)
            adx_break_count = int(exit_event.pop("adx_break_count"))
            if exit_event["closed"]:
                trade = exit_event["trade"]
                equity = trade.equity_after
                trades.append(trade)
                open_trade = None
                adx_break_count = 0

        reasons = _entry_rejection_reasons(candle, open_trade, cfg)
        side = ""
        if not reasons:
            side = "LONG" if float(candle["close"]) < float(candle["bb_lower"]) else "SHORT"
        evaluations.append(_evaluation_row(asset, timeframe, candle, side, reasons, cfg))

        if open_trade is None and not reasons:
            open_trade = _open_trade(candle, idx, side, cfg)
            adx_break_count = 0

    return pd.DataFrame([trade.__dict__ for trade in trades]), pd.DataFrame(evaluations)


def run_random_range_baseline(
    prices: pd.DataFrame,
    asset: str,
    timeframe: str,
    trade_count: int,
    config: RangeMRConfig | None = None,
) -> pd.DataFrame:
    cfg = config or RangeMRConfig()
    if trade_count <= 0:
        return pd.DataFrame()
    enriched = add_range_mr_indicators(prices, cfg).dropna(subset=["ma20", "atr", "adx"]).reset_index(drop=True)
    eligible = enriched[(enriched["adx"] < cfg.adx_entry_below) & (enriched["close"] != enriched["ma20"])].index.tolist()
    if not eligible:
        return pd.DataFrame()

    rng = random.Random(cfg.random_seed)
    selected = sorted(rng.sample(eligible, min(trade_count, len(eligible))))
    trades: list[RangeMRTrade] = []
    equity = cfg.initial_equity
    blocked_until = -1
    for idx in selected:
        if idx <= blocked_until or idx >= len(enriched) - 1:
            continue
        candle = enriched.iloc[idx]
        side = "LONG" if float(candle["close"]) < float(candle["ma20"]) else "SHORT"
        open_trade = _open_trade(candle, idx, side, cfg)
        adx_break_count = 0
        for future_idx in range(idx + 1, len(enriched)):
            exit_event = _maybe_exit(open_trade, enriched.iloc[future_idx], future_idx, asset, timeframe, equity, cfg, adx_break_count)
            adx_break_count = int(exit_event.pop("adx_break_count"))
            if exit_event["closed"]:
                trade = exit_event["trade"]
                equity = trade.equity_after
                trades.append(trade)
                blocked_until = future_idx
                break
    return pd.DataFrame([trade.__dict__ for trade in trades])


def summarize_range_mr_result(trades: pd.DataFrame, evaluations: pd.DataFrame | None = None) -> dict[str, Any]:
    evaluations = evaluations if evaluations is not None else pd.DataFrame()
    if trades.empty:
        return _empty_summary(evaluations)
    pnl = trades["net_pnl"].astype(float)
    r_values = trades["r_multiple"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    ci_low, ci_high = _bootstrap_ci(r_values.tolist())
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy_r": float(r_values.mean()),
        "profit_factor": gross_win / gross_loss if gross_loss else (gross_win if gross_win else 0.0),
        "max_drawdown": _max_drawdown(trades["equity_after"].astype(float).tolist()),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "top_rejection": _top_rejection(evaluations),
    }


def classify_range_mr(
    train: dict[str, Any],
    test: dict[str, Any],
    random_summary: dict[str, Any],
    grid_status: str,
    min_trades: int = 20,
) -> str:
    if train["trades"] < min_trades or test["trades"] < min_trades:
        if train["expectancy_r"] > 0 and test["expectancy_r"] > 0:
            return "PROMISING_WATCHLIST"
        return "DIED"
    positive = train["expectancy_r"] > 0 and test["expectancy_r"] > 0
    clears_ci = train["ci_low"] > 0 and test["ci_low"] > 0
    beats_random = test["expectancy_r"] > random_summary.get("expectancy_r", 0.0)
    if positive and clears_ci and beats_random and grid_status == "PASSED":
        return "SURVIVED"
    if positive:
        return "PROMISING_WATCHLIST"
    return "DIED"


def split_train_test(prices: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _clean_ohlc(prices)
    split_at = max(1, min(len(frame) - 1, int(len(frame) * train_fraction)))
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def _entry_rejection_reasons(candle: pd.Series, open_trade: dict[str, Any] | None, cfg: RangeMRConfig) -> list[str]:
    reasons: list[str] = []
    if open_trade is not None:
        reasons.append("OPEN_TRADE_EXISTS")
    if float(candle["adx"]) >= cfg.adx_entry_below:
        reasons.append("ADX_NOT_RANGE")
    close = float(candle["close"])
    if not (close < float(candle["bb_lower"]) or close > float(candle["bb_upper"])):
        reasons.append("CLOSE_NOT_OUTSIDE_BAND")
    if float(candle["atr"]) <= 0:
        reasons.append("ATR_INVALID")
    return reasons


def _open_trade(candle: pd.Series, idx: int, side: str, cfg: RangeMRConfig) -> dict[str, Any]:
    entry = _slipped(float(candle["close"]), cfg.slippage_bps, "buy" if side == "LONG" else "sell")
    atr = float(candle["atr"])
    stop = entry - cfg.atr_stop_multiple * atr if side == "LONG" else entry + cfg.atr_stop_multiple * atr
    return {
        "side": side,
        "entry_index": int(idx),
        "entry_time": pd.to_datetime(candle["date"]).isoformat(),
        "entry_price": entry,
        "stop_loss": stop,
        "entry_adx": float(candle["adx"]),
        "entry_band_width_pct": float(candle["bb_width_pct"]),
    }


def _maybe_exit(
    open_trade: dict[str, Any],
    candle: pd.Series,
    idx: int,
    asset: str,
    timeframe: str,
    equity: float,
    cfg: RangeMRConfig,
    adx_break_count: int,
) -> dict[str, Any]:
    side = str(open_trade["side"])
    exit_price = 0.0
    exit_reason = ""
    adx_break_count = adx_break_count + 1 if float(candle["adx"]) >= cfg.adx_exit_at else 0

    if side == "LONG":
        if float(candle["low"]) <= float(open_trade["stop_loss"]):
            exit_price, exit_reason = float(open_trade["stop_loss"]), "SL"
        elif float(candle["close"]) >= float(candle["ma20"]):
            exit_price, exit_reason = float(candle["ma20"]), "TARGET"
    else:
        if float(candle["high"]) >= float(open_trade["stop_loss"]):
            exit_price, exit_reason = float(open_trade["stop_loss"]), "SL"
        elif float(candle["close"]) <= float(candle["ma20"]):
            exit_price, exit_reason = float(candle["ma20"]), "TARGET"

    if not exit_reason and adx_break_count >= cfg.adx_exit_bars:
        exit_price, exit_reason = float(candle["close"]), "REGIME_BREAK"
    if not exit_reason:
        return {"closed": False, "adx_break_count": adx_break_count}

    exit_price = _slipped(exit_price, cfg.slippage_bps, "sell" if side == "LONG" else "buy")
    trade = _close_trade(open_trade, candle, idx, asset, timeframe, exit_price, exit_reason, equity, cfg)
    return {"closed": True, "trade": trade, "adx_break_count": adx_break_count}


def _close_trade(open_trade: dict[str, Any], candle: pd.Series, idx: int, asset: str, timeframe: str, exit_price: float, exit_reason: str, equity: float, cfg: RangeMRConfig) -> RangeMRTrade:
    side = str(open_trade["side"])
    entry = float(open_trade["entry_price"])
    stop = float(open_trade["stop_loss"])
    risk_per_unit = abs(entry - stop)
    risk_dollars = equity * cfg.risk_per_trade
    quantity = risk_dollars / risk_per_unit if risk_per_unit > 0 else 0.0
    directional_move = exit_price - entry if side == "LONG" else entry - exit_price
    gross_pnl = directional_move * quantity
    fees = (entry * quantity + exit_price * quantity) * _fee_bps(asset, cfg) / 10000.0
    slippage_cost = (entry * quantity + exit_price * quantity) * cfg.slippage_bps / 10000.0
    net_pnl = gross_pnl - fees
    r_multiple = net_pnl / risk_dollars if risk_dollars else 0.0
    return RangeMRTrade(
        asset=asset,
        timeframe=timeframe,
        side=side,
        entry_time=str(open_trade["entry_time"]),
        exit_time=pd.to_datetime(candle["date"]).isoformat(),
        entry_price=entry,
        exit_price=exit_price,
        stop_loss=stop,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        fees=fees,
        slippage_cost=slippage_cost,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        equity_after=equity + net_pnl,
        holding_bars=int(idx) - int(open_trade["entry_index"]),
        entry_adx=float(open_trade["entry_adx"]),
        entry_band_width_pct=float(open_trade["entry_band_width_pct"]),
        strategy_version=cfg.version,
    )


def _evaluation_row(asset: str, timeframe: str, candle: pd.Series, side: str, reasons: list[str], cfg: RangeMRConfig) -> dict[str, Any]:
    return {
        "asset": asset,
        "timeframe": timeframe,
        "evaluated_at": pd.to_datetime(candle["date"]).isoformat(),
        "close": float(candle["close"]),
        "adx": float(candle["adx"]),
        "bb_width_pct": float(candle["bb_width_pct"]),
        "ma20": float(candle["ma20"]),
        "bb_lower": float(candle["bb_lower"]),
        "bb_upper": float(candle["bb_upper"]),
        "side": side,
        "passed": not reasons,
        "rejection_reasons": "|".join(reasons),
        "strategy_version": cfg.version,
    }


def _clean_ohlc(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def _close_time_ms(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates, utc=True).astype("int64") // 1_000_000).astype("int64")


def _slipped(price: float, slippage_bps: float, action: str) -> float:
    factor = slippage_bps / 10000.0
    return price * (1.0 + factor) if action == "buy" else price * (1.0 - factor)


def _fee_bps(asset: str, cfg: RangeMRConfig) -> float:
    asset = asset.upper()
    if asset in {"EURUSD", "USDJPY", "GBPUSD", "USDCHF"}:
        return cfg.fee_bps_forex
    if asset in {"GOLD", "SILVER", "GOLD_FUT", "SILVER_FUT", "OIL_FUT", "BRENT_FUT"}:
        return cfg.fee_bps_market
    return cfg.fee_bps_crypto


def _empty_summary(evaluations: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": 0,
        "win_rate": 0.0,
        "net_pnl": 0.0,
        "expectancy_r": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "top_rejection": _top_rejection(evaluations),
    }


def _max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value - peak) / peak)
    return float(drawdown)


def _bootstrap_ci(values: list[float], iterations: int = 400, seed: int = 3039) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return float(means[int(0.025 * (len(means) - 1))]), float(means[int(0.975 * (len(means) - 1))])


def _top_rejection(evaluations: pd.DataFrame) -> str:
    if evaluations.empty or "rejection_reasons" not in evaluations:
        return "none"
    counts: dict[str, int] = {}
    for value in evaluations["rejection_reasons"].fillna(""):
        for reason in str(value).split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "none"
