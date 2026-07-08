from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TechnicalAnalysisResult:
    rsi: float
    trend_score: float
    fair_value_gap: str
    liquidity_sweep: str
    momentum_score: float
    macd_signal: str
    ma_alignment: str
    atr_pct: float
    confluence_score: float
    confluence_label: str
    market_regime: str
    volatility_regime: str
    bos_signal: str
    technical_bias_score: float


def analyze_technical(prices: pd.DataFrame, lookback_days: int) -> TechnicalAnalysisResult:
    frame = prices.sort_values("date").copy().reset_index(drop=True)
    recent = frame.tail(lookback_days).copy()
    close = frame["close"].astype(float)
    rsi = _rsi(close)
    trend_score = _trend_score(recent["close"].astype(float))
    fvg = _fair_value_gap(recent)
    sweep = _liquidity_sweep(recent)
    macd_signal = _macd_signal(close)
    ma_alignment = _ma_alignment(close)
    atr = _atr(frame)
    last_close = float(close.iloc[-1]) if len(close) else 0.0
    atr_pct = (atr / last_close * 100) if last_close else 0.0
    bos_signal = _break_of_structure(frame)
    volatility_regime = _volatility_regime(frame, atr_pct)
    market_regime = _market_regime(ma_alignment, trend_score)
    momentum = float(np.tanh((rsi - 50) / 18 + trend_score))
    confluence_score, label, technical_bias = _confluence(
        rsi=rsi,
        trend_score=trend_score,
        fvg=fvg,
        sweep=sweep,
        macd_signal=macd_signal,
        ma_alignment=ma_alignment,
        bos_signal=bos_signal,
    )
    return TechnicalAnalysisResult(
        rsi=round(rsi, 2),
        trend_score=round(trend_score, 3),
        fair_value_gap=fvg,
        liquidity_sweep=sweep,
        momentum_score=round(momentum, 3),
        macd_signal=macd_signal,
        ma_alignment=ma_alignment,
        atr_pct=round(float(atr_pct), 3),
        confluence_score=round(confluence_score, 1),
        confluence_label=label,
        market_regime=market_regime,
        volatility_regime=volatility_regime,
        bos_signal=bos_signal,
        technical_bias_score=round(technical_bias, 3),
    )


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    if delta.empty:
        return 50.0
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if pd.isna(gain) or pd.isna(loss):
        return 50.0
    if loss == 0:
        return 100.0
    return float(100 - (100 / (1 + gain / loss)))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd_signal(series: pd.Series) -> str:
    if len(series) < 35:
        return "neutral"
    line = _ema(series, 12) - _ema(series, 26)
    signal = _ema(line, 9)
    if line.iloc[-1] > signal.iloc[-1] and line.iloc[-2] <= signal.iloc[-2]:
        return "bullish"
    if line.iloc[-1] < signal.iloc[-1] and line.iloc[-2] >= signal.iloc[-2]:
        return "bearish"
    if line.iloc[-1] > signal.iloc[-1]:
        return "bullish"
    if line.iloc[-1] < signal.iloc[-1]:
        return "bearish"
    return "neutral"


def _ma_alignment(series: pd.Series) -> str:
    if len(series) < 99:
        return "neutral"
    ma7 = series.rolling(7).mean().iloc[-1]
    ma25 = series.rolling(25).mean().iloc[-1]
    ma99 = series.rolling(99).mean().iloc[-1]
    price = series.iloc[-1]
    if price > ma7 > ma25 > ma99:
        return "bullish"
    if price < ma7 < ma25 < ma99:
        return "bearish"
    if price > ma25:
        return "bullish"
    if price < ma25:
        return "bearish"
    return "neutral"


def _trend_score(series: pd.Series) -> float:
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or y.std() == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    normalized = slope / max(abs(y.mean()), 1)
    return float(np.tanh(normalized * len(y) * 5))


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    if len(frame) < 2:
        return 0.0
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(period).mean())


def _fair_value_gap(frame: pd.DataFrame) -> str:
    if len(frame) < 3:
        return "none"
    last_three = frame.tail(3).reset_index(drop=True)
    body = abs(float(last_three.loc[1, "close"] - last_three.loc[1, "open"]))
    atr = _atr(frame)
    if atr and body < 0.5 * atr:
        return "none"
    if last_three.loc[2, "low"] > last_three.loc[0, "high"]:
        return "bullish"
    if last_three.loc[2, "high"] < last_three.loc[0, "low"]:
        return "bearish"
    return "none"


def _liquidity_sweep(frame: pd.DataFrame) -> str:
    if len(frame) < 8:
        return "none"
    prior = frame.iloc[:-1]
    last = frame.iloc[-1]
    if last["high"] > prior["high"].max() and last["close"] < prior["high"].max():
        return "upside"
    if last["low"] < prior["low"].min() and last["close"] > prior["low"].min():
        return "downside"
    return "none"


def _break_of_structure(frame: pd.DataFrame, swing: int = 10) -> str:
    if len(frame) < swing + 2:
        return "none"
    prior = frame.iloc[-swing - 1 : -1]
    last = frame.iloc[-1]
    if last["close"] > prior["high"].max():
        return "bullish"
    if last["close"] < prior["low"].min():
        return "bearish"
    return "none"


def _volatility_regime(frame: pd.DataFrame, atr_pct: float) -> str:
    if len(frame) < 40:
        if atr_pct >= 3:
            return "High-Vol"
        if atr_pct <= 1:
            return "Low-Vol"
        return "Normal-Vol"
    close = frame["close"].astype(float)
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    bandwidth = (4 * sd / ma * 100).dropna()
    if bandwidth.empty:
        return "Normal-Vol"
    current = bandwidth.iloc[-1]
    median = bandwidth.median()
    if current >= median * 1.3:
        return "High-Vol"
    if current <= median * 0.7:
        return "Low-Vol"
    return "Normal-Vol"


def _market_regime(ma_alignment: str, trend_score: float) -> str:
    if ma_alignment == "bullish" and trend_score > 0.12:
        return "Bull"
    if ma_alignment == "bearish" and trend_score < -0.12:
        return "Bear"
    return "Range"


def _signal_value(signal: str, bullish: str = "bullish", bearish: str = "bearish") -> int:
    if signal == bullish:
        return 1
    if signal == bearish:
        return -1
    return 0


def _confluence(
    rsi: float,
    trend_score: float,
    fvg: str,
    sweep: str,
    macd_signal: str,
    ma_alignment: str,
    bos_signal: str,
) -> tuple[float, str, float]:
    values = []
    if rsi > 58:
        values.append(1)
    elif rsi < 42:
        values.append(-1)
    else:
        values.append(0)
    values.append(1 if trend_score > 0.12 else -1 if trend_score < -0.12 else 0)
    values.append(_signal_value(macd_signal))
    values.append(_signal_value(ma_alignment))
    values.append(_signal_value(fvg))
    values.append(_signal_value(bos_signal))
    if sweep == "downside":
        values.append(1)
    elif sweep == "upside":
        values.append(-1)
    else:
        values.append(0)
    bias = sum(values) / len(values)
    score = (bias + 1) / 2 * 100
    if score >= 66:
        label = "Bullish confluence"
    elif score <= 34:
        label = "Bearish confluence"
    else:
        label = "Mixed confluence"
    return float(score), label, float(bias)
