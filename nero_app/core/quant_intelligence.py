from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantSnapshot:
    asset: str
    source: str
    latest_close: float
    observation_count: int
    zscore_20: float
    realized_vol_30d: float
    realized_vol_90d: float
    sharpe_90d: float
    sortino_90d: float
    max_drawdown_90d: float
    trend_20d: float
    trend_60d: float
    regime: str
    pressure: str
    notes: list[str]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    cleaned = prices.replace(0, np.nan).astype(float)
    return np.log(cleaned / cleaned.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def rolling_correlation(returns: pd.DataFrame, asset_x: str, asset_y: str, window: int = 60) -> pd.Series:
    return returns[asset_x].rolling(window).corr(returns[asset_y])


def rolling_beta(returns: pd.DataFrame, asset: str, benchmark: str, window: int = 60) -> pd.Series:
    cov = returns[asset].rolling(window).cov(returns[benchmark])
    var = returns[benchmark].rolling(window).var()
    return cov / var.replace(0, np.nan)


def zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    price = pd.to_numeric(prices, errors="coerce")
    mean = price.rolling(window).mean()
    std = price.rolling(window).std()
    return (price - mean) / std.replace(0, np.nan)


def realized_volatility(returns: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    vol = pd.to_numeric(returns, errors="coerce").rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    std = excess.std()
    if pd.isna(std) or std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if pd.isna(downside_std) or downside_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    clean = pd.to_numeric(cumulative_returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    running_max = clean.cummax()
    drawdown = (clean - running_max) / running_max.replace(0, np.nan)
    return float(drawdown.min()) if not drawdown.dropna().empty else 0.0


def information_coefficient(predicted_returns: pd.Series, actual_returns: pd.Series) -> float:
    frame = pd.concat([predicted_returns, actual_returns], axis=1).dropna()
    if len(frame) < 3:
        return 0.0
    value = frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank())
    return 0.0 if pd.isna(value) else float(value)


def build_quant_snapshot(price_history: pd.DataFrame, asset: str, source: str = "local prices") -> QuantSnapshot:
    prices = _clean_price_history(price_history)
    if prices.empty:
        return QuantSnapshot(asset, source, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "NO_DATA", "unknown", ["No usable price history."])

    close = prices["close"]
    returns = log_returns(close)
    z20 = _latest(zscore(close, 20))
    vol30 = _latest(realized_volatility(returns, 30))
    vol90 = _latest(realized_volatility(returns, 90))
    ret90 = returns.tail(90)
    cumulative_90 = (1 + ret90).cumprod()
    trend20 = _window_return(close, 20)
    trend60 = _window_return(close, 60)
    snapshot = QuantSnapshot(
        asset=asset,
        source=source,
        latest_close=float(close.iloc[-1]),
        observation_count=len(prices),
        zscore_20=z20,
        realized_vol_30d=vol30,
        realized_vol_90d=vol90,
        sharpe_90d=sharpe_ratio(ret90),
        sortino_90d=sortino_ratio(ret90),
        max_drawdown_90d=max_drawdown(cumulative_90),
        trend_20d=trend20,
        trend_60d=trend60,
        regime=_classify_regime(trend20, trend60, vol30),
        pressure=_classify_pressure(z20, trend20, trend60),
        notes=[],
    )
    return _with_notes(snapshot)


def quant_driver_rows(snapshot: QuantSnapshot) -> list[dict[str, str]]:
    return [
        {"Signal": "20D Z-Score", "Reading": f"{snapshot.zscore_20:.2f}", "Meaning": "Positive = stretched above mean, negative = below mean."},
        {"Signal": "30D Realized Vol", "Reading": f"{snapshot.realized_vol_30d:.1%}", "Meaning": "Annualized volatility; higher vol demands stricter trade quality."},
        {"Signal": "90D Realized Vol", "Reading": f"{snapshot.realized_vol_90d:.1%}", "Meaning": "Medium-term volatility baseline."},
        {"Signal": "20D Return", "Reading": f"{snapshot.trend_20d:.1%}", "Meaning": "Short-term trend pressure."},
        {"Signal": "60D Return", "Reading": f"{snapshot.trend_60d:.1%}", "Meaning": "Broader trend direction."},
        {"Signal": "90D Sharpe", "Reading": f"{snapshot.sharpe_90d:.2f}", "Meaning": "Recent return quality adjusted for volatility."},
        {"Signal": "90D Sortino", "Reading": f"{snapshot.sortino_90d:.2f}", "Meaning": "Recent return quality adjusted for downside volatility."},
        {"Signal": "90D Max Drawdown", "Reading": f"{snapshot.max_drawdown_90d:.1%}", "Meaning": "Worst recent equity curve drop from peak."},
        {"Signal": "Quant Regime", "Reading": snapshot.regime, "Meaning": "Statistical environment from trend and volatility."},
        {"Signal": "Pressure", "Reading": snapshot.pressure, "Meaning": "Mean-reversion/trend pressure summary."},
    ]


def _clean_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty or "close" not in price_history.columns:
        return pd.DataFrame(columns=["date", "close"])
    frame = price_history.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["close"]).reset_index(drop=True)


def _window_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return 0.0
    start = float(close.iloc[-window - 1])
    end = float(close.iloc[-1])
    return (end / start) - 1 if start else 0.0


def _latest(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.iloc[-1])


def _classify_regime(trend20: float, trend60: float, vol30: float) -> str:
    if vol30 >= 0.75:
        vol = "High-Vol"
    elif vol30 <= 0.25:
        vol = "Low-Vol"
    else:
        vol = "Normal-Vol"
    if trend20 > 0.03 and trend60 > 0.05:
        trend = "Bull"
    elif trend20 < -0.03 and trend60 < -0.05:
        trend = "Bear"
    else:
        trend = "Range"
    return f"{trend} / {vol}"


def _classify_pressure(z20: float, trend20: float, trend60: float) -> str:
    if z20 <= -2 and trend60 > 0:
        return "mean-reversion long watch"
    if z20 >= 2 and trend60 < 0:
        return "mean-reversion short risk"
    if trend20 > 0 and trend60 > 0:
        return "upside trend pressure"
    if trend20 < 0 and trend60 < 0:
        return "downside trend pressure"
    return "mixed pressure"


def _with_notes(snapshot: QuantSnapshot) -> QuantSnapshot:
    notes: list[str] = []
    if snapshot.observation_count < 90:
        notes.append("Less than 90 candles; medium-term stats are weak.")
    if snapshot.realized_vol_30d > snapshot.realized_vol_90d * 1.25 and snapshot.realized_vol_90d > 0:
        notes.append("30D volatility is materially above 90D baseline.")
    if abs(snapshot.zscore_20) >= 2:
        notes.append("20D z-score is stretched; avoid chasing without confirmation.")
    if snapshot.sharpe_90d < 0:
        notes.append("90D risk-adjusted return is negative.")
    if not notes:
        notes.append("No major quant warning from local price statistics.")
    return QuantSnapshot(
        asset=snapshot.asset,
        source=snapshot.source,
        latest_close=snapshot.latest_close,
        observation_count=snapshot.observation_count,
        zscore_20=snapshot.zscore_20,
        realized_vol_30d=snapshot.realized_vol_30d,
        realized_vol_90d=snapshot.realized_vol_90d,
        sharpe_90d=snapshot.sharpe_90d,
        sortino_90d=snapshot.sortino_90d,
        max_drawdown_90d=snapshot.max_drawdown_90d,
        trend_20d=snapshot.trend_20d,
        trend_60d=snapshot.trend_60d,
        regime=snapshot.regime,
        pressure=snapshot.pressure,
        notes=notes,
    )
