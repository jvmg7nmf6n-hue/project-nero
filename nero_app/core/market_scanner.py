from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_SCANNER_ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "SOL",
    "ADA",
    "LINK",
    "LTC",
    "AVAX",
    "NEAR",
    "DOT",
    "DOGE",
    "SHIB",
    "PEPE",
    "BONK",
    "WIF",
]


@dataclass(frozen=True)
class ScannerAlert:
    asset: str
    event_type: str
    title: str
    message: str
    priority: str = "default"
    tags: str = ""


def scan_market_activity(
    prices: pd.DataFrame,
    asset: str,
    bar_label: str = "30m",
    move_pct_threshold: float = 3.0,
    rsi_high: float = 75.0,
    rsi_low: float = 25.0,
    volume_multiple: float = 2.5,
) -> list[ScannerAlert]:
    frame = prices.sort_values("date").tail(60).copy()
    if len(frame) < 25:
        return []

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if len(frame) < 25:
        return []

    alerts: list[ScannerAlert] = []
    last = frame.iloc[-1]
    prev = frame.iloc[-2]
    close = float(last["close"])
    prev_close = float(prev["close"])
    move_pct = ((close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    window = frame.iloc[-21:-1]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    average_volume = float(window["volume"].fillna(0.0).mean())
    last_volume = float(last.get("volume", 0.0) or 0.0)
    rsi_value = calculate_rsi(frame["close"].astype(float).tolist())

    if abs(move_pct) >= move_pct_threshold:
        direction = "+" if move_pct >= 0 else ""
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="move",
                title=f"{asset} {direction}{move_pct:.1f}% ({bar_label})",
                message=f"Price ${format_price(close)} par strong {bar_label} move.",
                priority="high",
                tags="rocket" if move_pct >= 0 else "chart_with_downwards_trend",
            )
        )

    if close > range_high:
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="breakout",
                title=f"{asset} breakout",
                message=f"20-bar high ${format_price(range_high)} ke upar, ab ${format_price(close)}.",
                priority="high",
                tags="arrow_up_small",
            )
        )
    elif close < range_low:
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="breakdown",
                title=f"{asset} breakdown",
                message=f"20-bar low ${format_price(range_low)} ke neeche, ab ${format_price(close)}.",
                priority="high",
                tags="arrow_down_small",
            )
        )

    if average_volume > 0 and last_volume >= average_volume * volume_multiple:
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="volume",
                title=f"{asset} volume spike",
                message=f"{last_volume / average_volume:.1f}x average volume.",
                tags="loudspeaker",
            )
        )

    if rsi_value is not None and rsi_value >= rsi_high:
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="rsi_high",
                title=f"{asset} RSI {rsi_value:.0f} overbought",
                message="Chase risk. Confirmation ka wait behtar hai.",
                tags="warning",
            )
        )
    elif rsi_value is not None and rsi_value <= rsi_low:
        alerts.append(
            ScannerAlert(
                asset=asset,
                event_type="rsi_low",
                title=f"{asset} RSI {rsi_value:.0f} oversold",
                message="Bounce zone, lekin reversal confirm hona chahiye.",
                tags="warning",
            )
        )

    return alerts


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for index in range(len(closes) - period, len(closes)):
        change = closes[index] - closes[index - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    average_gain = gains / period
    average_loss = losses / period
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def format_price(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:.0f}"
    if absolute >= 1:
        return f"{value:.2f}"
    if absolute >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3g}"
