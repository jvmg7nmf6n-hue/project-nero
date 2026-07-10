from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os

import pandas as pd
import requests


DEFAULT_HISTORY_DIR = Path("nero_app/data/history")
BINANCE_DAILY_LIMIT = 1000
MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class PriceFetchResult:
    asset: str
    path: Path
    rows: int
    source: str
    status: str


def fetch_binance_daily_history(
    symbol: str,
    start: str,
    end: str | None = None,
    timeout_seconds: int = 12,
) -> pd.DataFrame:
    start_ms = _date_to_ms(start)
    end_ms = _date_to_ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    cursor = start_ms

    while cursor < end_ms:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": BINANCE_DAILY_LIMIT,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break
        frame = _binance_frame(payload)
        frames.append(frame)
        last_open_time = int(payload[-1][0])
        next_cursor = last_open_time + MS_PER_DAY
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(payload) < BINANCE_DAILY_LIMIT:
            break

    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    combined = pd.concat(frames, ignore_index=True)
    return _dedupe_prices(combined)


def fetch_twelve_data_daily_history(
    symbol: str,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 12,
) -> pd.DataFrame:
    key = (api_key or os.getenv("TWELVE_DATA_API_KEY", "")).strip()
    if not key:
        raise ValueError("Twelve Data API key is required")
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start,
        "outputsize": 5000,
        "apikey": key,
    }
    if end:
        params["end_date"] = end
    response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise ValueError(str(payload.get("message", "Twelve Data error")))
    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(values)
    numeric_columns = ["open", "high", "low", "close"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0) if "volume" in frame.columns else 0.0
    frame["date"] = pd.to_datetime(frame["datetime"])
    return _dedupe_prices(frame[["date", "open", "high", "low", "close", "volume"]])


def write_price_history(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _dedupe_prices(frame).to_csv(path, index=False)
    return path


def fetch_and_write_standard_histories(
    start: str = "2021-01-01",
    end: str | None = None,
    output_dir: Path = DEFAULT_HISTORY_DIR,
    twelve_data_api_key: str | None = None,
) -> list[PriceFetchResult]:
    results: list[PriceFetchResult] = []

    btc_path = output_dir / "btc_daily.csv"
    try:
        btc = fetch_binance_daily_history("BTCUSDT", start=start, end=end)
        write_price_history(btc, btc_path)
        results.append(PriceFetchResult("BTC", btc_path, len(btc), "Binance BTCUSDT 1d", "ok"))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        results.append(PriceFetchResult("BTC", btc_path, 0, "Binance BTCUSDT 1d", f"error: {exc.__class__.__name__}"))

    gold_path = output_dir / "gold_daily.csv"
    try:
        gold = fetch_twelve_data_daily_history("XAU/USD", start=start, end=end, api_key=twelve_data_api_key)
        write_price_history(gold, gold_path)
        results.append(PriceFetchResult("GOLD", gold_path, len(gold), "Twelve Data XAU/USD 1d", "ok"))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        results.append(PriceFetchResult("GOLD", gold_path, 0, "Twelve Data XAU/USD 1d", f"error: {exc.__class__.__name__}"))

    return results


def _binance_frame(payload: list[list[object]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        payload,
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
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["date"] = frame["open_time"].map(lambda value: datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).replace(tzinfo=None))
    return frame[["date", "open", "high", "low", "close", "volume"]]


def _dedupe_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _date_to_ms(value: str) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.timestamp() * 1000)

