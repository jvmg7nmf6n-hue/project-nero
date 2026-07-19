from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path

import pandas as pd
import requests

from nero_app.core.data_loader import load_price_history


BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "LINK": "LINKUSDT",
    "LTC": "LTCUSDT",
    "AVAX": "AVAXUSDT",
    "NEAR": "NEARUSDT",
    "BNB": "BNBUSDT",
    "DOT": "DOTUSDT",
    "DOGE": "DOGEUSDT",
    "SHIB": "SHIBUSDT",
    "PEPE": "PEPEUSDT",
    "BONK": "BONKUSDT",
    "WIF": "WIFUSDT",
    "PAXG": "PAXGUSDT",
}

COINBASE_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "LINK": "LINK-USD",
    "LTC": "LTC-USD",
    "AVAX": "AVAX-USD",
    "NEAR": "NEAR-USD",
    "BNB": "BNB-USD",
    "DOT": "DOT-USD",
    "DOGE": "DOGE-USD",
    "SHIB": "SHIB-USD",
    "PEPE": "PEPE-USD",
    "BONK": "BONK-USD",
    "WIF": "WIF-USD",
    "PAXG": "PAXG-USD",
}

KRAKEN_PAIRS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "ADA": "ADAUSD",
    "LINK": "LINKUSD",
    "LTC": "LTCUSD",
    "AVAX": "AVAXUSD",
    "DOT": "DOTUSD",
    "DOGE": "XDGUSD",
    "SHIB": "SHIBUSD",
    "PAXG": "PAXGUSD",
}

TWELVE_DATA_SYMBOLS = {
    "GOLD": "XAU/USD",
    "SILVER": "XAG/USD",
    "OIL": "WTI/USD",
    "FDX": "FDX",
}


YFINANCE_SYMBOLS = {
    # Equity/index and crypto-linked stock drivers
    "SPY": "SPY",
    "QQQ": "QQQ",
    "NVDA": "NVDA",
    "MSTR": "MSTR",
    "COIN": "COIN",
    "MARA": "MARA",
    "RIOT": "RIOT",
    # Gold/miner proxies
    "GLD": "GLD",
    "GDX": "GDX",
    "NEM": "NEM",
    # Metals and energy futures proxies
    "GOLD_FUT": "GC=F",
    "SILVER_FUT": "SI=F",
    "COPPER_FUT": "HG=F",
    "OIL_FUT": "CL=F",
    "BRENT_FUT": "BZ=F",
    # Dollar and currency pairs
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
}


@dataclass(frozen=True)
class MarketDataResult:
    prices: pd.DataFrame
    source: str
    status: str


class MarketDataClient:
    def __init__(self, timeout_seconds: int = 8) -> None:
        self.timeout_seconds = timeout_seconds

    def load(
        self,
        asset: str,
        prefer_live: bool = False,
        days: int = 365,
        twelve_data_api_key: str | None = None,
    ) -> MarketDataResult:
        if prefer_live and asset in BINANCE_SYMBOLS:
            errors: list[str] = []
            try:
                prices = self._load_binance_daily(BINANCE_SYMBOLS[asset], days=days)
                return MarketDataResult(
                    prices=prices,
                    source=f"Binance {BINANCE_SYMBOLS[asset]} daily candles",
                    status="live",
                )
            except requests.RequestException as exc:
                errors.append(f"Binance {exc.__class__.__name__}")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Binance malformed response ({exc.__class__.__name__})")

            if asset in COINBASE_PRODUCTS:
                try:
                    prices = self._load_coinbase_candles(COINBASE_PRODUCTS[asset], granularity=86400, candles=days)
                    return MarketDataResult(
                        prices=prices,
                        source=f"Coinbase {COINBASE_PRODUCTS[asset]} daily candles",
                        status="live",
                    )
                except requests.RequestException as exc:
                    errors.append(f"Coinbase {exc.__class__.__name__}")
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"Coinbase malformed response ({exc.__class__.__name__})")

            if asset in KRAKEN_PAIRS:
                try:
                    prices = self._load_kraken_ohlc(KRAKEN_PAIRS[asset], interval_minutes=1440, candles=days)
                    return MarketDataResult(
                        prices=prices,
                        source=f"Kraken {KRAKEN_PAIRS[asset]} daily candles",
                        status="live",
                    )
                except requests.RequestException as exc:
                    errors.append(f"Kraken {exc.__class__.__name__}")
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"Kraken malformed response ({exc.__class__.__name__})")
            return self._fallback(f"fallback: {'; '.join(errors)}")

        if prefer_live and asset in TWELVE_DATA_SYMBOLS:
            api_key = twelve_data_api_key or os.getenv("TWELVE_DATA_API_KEY", "")
            if not api_key.strip():
                return self._fallback("fallback: missing Twelve Data API key")
            try:
                symbol = TWELVE_DATA_SYMBOLS[asset]
                prices = self._load_twelve_data_daily(symbol=symbol, days=days, api_key=api_key.strip())
                return MarketDataResult(
                    prices=prices,
                    source=f"Twelve Data {symbol} daily candles",
                    status="live",
                )
            except requests.RequestException as exc:
                return self._fallback(f"fallback: {exc.__class__.__name__}")
            except (KeyError, TypeError, ValueError) as exc:
                return self._fallback(f"fallback: malformed Twelve Data response ({exc.__class__.__name__})")

        if prefer_live and asset in YFINANCE_SYMBOLS:
            try:
                symbol = YFINANCE_SYMBOLS[asset]
                prices = self._load_yfinance_ohlc(symbol=symbol, period=_yfinance_period(days=days), interval="1d")
                return MarketDataResult(
                    prices=prices,
                    source=f"yfinance {symbol} daily candles",
                    status="live",
                )
            except (ImportError, KeyError, TypeError, ValueError) as exc:
                return self._fallback(f"fallback: yfinance {exc.__class__.__name__}")

        return MarketDataResult(
            prices=load_price_history(),
            source="Generated sample candles",
            status="sample",
        )
    def load_intraday(
        self,
        asset: str,
        prefer_live: bool = False,
        interval: str = "1h",
        candles: int = 240,
        twelve_data_api_key: str | None = None,
    ) -> MarketDataResult:
        if prefer_live and asset in BINANCE_SYMBOLS:
            errors: list[str] = []
            try:
                prices = self._load_binance_intraday(BINANCE_SYMBOLS[asset], interval=interval, candles=candles)
                return MarketDataResult(
                    prices=prices,
                    source=f"Binance {BINANCE_SYMBOLS[asset]} {interval} candles",
                    status="live",
                )
            except requests.RequestException as exc:
                errors.append(f"Binance {exc.__class__.__name__}")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Binance malformed response ({exc.__class__.__name__})")

            if asset in COINBASE_PRODUCTS:
                try:
                    prices = self._load_coinbase_candles(
                        COINBASE_PRODUCTS[asset],
                        granularity=_coinbase_granularity(interval),
                        candles=candles,
                    )
                    return MarketDataResult(
                        prices=prices,
                        source=f"Coinbase {COINBASE_PRODUCTS[asset]} {interval} candles",
                        status="live",
                    )
                except requests.RequestException as exc:
                    errors.append(f"Coinbase {exc.__class__.__name__}")
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"Coinbase malformed response ({exc.__class__.__name__})")

            if asset in KRAKEN_PAIRS:
                try:
                    prices = self._load_kraken_ohlc(
                        KRAKEN_PAIRS[asset],
                        interval_minutes=_kraken_interval_minutes(interval),
                        candles=candles,
                    )
                    return MarketDataResult(
                        prices=prices,
                        source=f"Kraken {KRAKEN_PAIRS[asset]} {interval} candles",
                        status="live",
                    )
                except requests.RequestException as exc:
                    errors.append(f"Kraken {exc.__class__.__name__}")
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"Kraken malformed response ({exc.__class__.__name__})")
            return self._fallback_intraday(f"fallback: {'; '.join(errors)}")

        if prefer_live and asset in TWELVE_DATA_SYMBOLS:
            api_key = twelve_data_api_key or os.getenv("TWELVE_DATA_API_KEY", "")
            if not api_key.strip():
                return self._fallback_intraday("fallback: missing Twelve Data API key")
            try:
                symbol = TWELVE_DATA_SYMBOLS[asset]
                prices = self._load_twelve_data_intraday(
                    symbol=symbol,
                    interval=interval,
                    candles=candles,
                    api_key=api_key.strip(),
                )
                return MarketDataResult(
                    prices=prices,
                    source=f"Twelve Data {symbol} {interval} candles",
                    status="live",
                )
            except requests.RequestException as exc:
                return self._fallback_intraday(f"fallback: {exc.__class__.__name__}")
            except (KeyError, TypeError, ValueError) as exc:
                return self._fallback_intraday(f"fallback: malformed Twelve Data intraday response ({exc.__class__.__name__})")

        if prefer_live and asset in YFINANCE_SYMBOLS:
            try:
                symbol = YFINANCE_SYMBOLS[asset]
                yf_interval = _yfinance_fetch_interval(interval)
                prices = self._load_yfinance_ohlc(
                    symbol=symbol,
                    period=_yfinance_intraday_period(candles=candles, interval=interval),
                    interval=yf_interval,
                )
                prices = _resample_ohlc(prices, interval).tail(candles).reset_index(drop=True)
                return MarketDataResult(
                    prices=prices,
                    source=f"yfinance {symbol} {interval} candles",
                    status="live",
                )
            except (ImportError, KeyError, TypeError, ValueError) as exc:
                return self._fallback_intraday(f"fallback: yfinance {exc.__class__.__name__}")

        return self._fallback_intraday("sample")

    def _fallback(self, status: str) -> MarketDataResult:
        return MarketDataResult(
            prices=load_price_history(),
            source="Generated sample candles",
            status=status,
        )


    def _fallback_intraday(self, status: str) -> MarketDataResult:
        daily = load_price_history().tail(14).copy()
        hourly = daily.set_index("date").resample("1h").ffill().reset_index()
        hourly["open"] = hourly["close"].shift(1).fillna(hourly["open"])
        hourly["high"] = hourly[["open", "close"]].max(axis=1) * 1.002
        hourly["low"] = hourly[["open", "close"]].min(axis=1) * 0.998
        hourly["volume"] = hourly["volume"] / 24
        return MarketDataResult(
            prices=hourly.tail(240).reset_index(drop=True),
            source="Generated sample intraday candles",
            status=status,
        )

    def _load_binance_daily(self, symbol: str, days: int) -> pd.DataFrame:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": max(30, min(days, 1000))},
            timeout=self.timeout_seconds,
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
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        frame["date"] = frame["open_time"].map(
            lambda value: datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(tzinfo=None)
        )
        return frame[["date", "open", "high", "low", "close", "volume"]]


    def _load_binance_intraday(self, symbol: str, interval: str, candles: int) -> pd.DataFrame:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": max(30, min(candles, 1000))},
            timeout=self.timeout_seconds,
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
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        frame["date"] = frame["open_time"].map(
            lambda value: datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(tzinfo=None)
        )
        return frame[["date", "open", "high", "low", "close", "volume"]]


    def _load_yfinance_ohlc(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            import yfinance as yf  # local import keeps app usable without optional provider
        except ImportError as exc:  # pragma: no cover - exercised via caller fallback.
            raise ImportError("yfinance is not installed") from exc

        cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
        history = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if history.empty:
            raise ValueError(f"empty yfinance history for {symbol}")
        frame = history.reset_index()
        date_column = "Datetime" if "Datetime" in frame.columns else "Date"
        rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", date_column: "date"}
        frame = frame.rename(columns=rename)
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        else:
            frame["volume"] = 0.0
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        if frame.empty:
            raise ValueError(f"no usable yfinance candles for {symbol}")
        return frame[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def _load_coinbase_candles(self, product_id: str, granularity: int, candles: int) -> pd.DataFrame:
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params={"granularity": granularity},
            headers={"User-Agent": "Project-Nero/1.0"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError("empty Coinbase candle response")
        frame = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        frame["date"] = frame["time"].map(
            lambda value: datetime.fromtimestamp(int(value), tz=timezone.utc).replace(tzinfo=None)
        )
        frame = frame.sort_values("date").tail(max(30, min(candles, 300))).reset_index(drop=True)
        return frame[["date", "open", "high", "low", "close", "volume"]]

    def _load_kraken_ohlc(self, pair: str, interval_minutes: int, candles: int) -> pd.DataFrame:
        response = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": pair, "interval": interval_minutes},
            headers={"User-Agent": "Project-Nero/1.0"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(str(payload["error"]))
        result = payload["result"]
        series_key = next(key for key in result if key != "last")
        frame = pd.DataFrame(
            result[series_key],
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
        )
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        frame["date"] = frame["time"].map(
            lambda value: datetime.fromtimestamp(int(value), tz=timezone.utc).replace(tzinfo=None)
        )
        frame = frame.sort_values("date").tail(max(30, min(candles, 720))).reset_index(drop=True)
        return frame[["date", "open", "high", "low", "close", "volume"]]

    def _load_twelve_data_daily(self, symbol: str, days: int, api_key: str) -> pd.DataFrame:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": max(30, min(days, 5000)),
                "apikey": api_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise ValueError(str(payload.get("message", "Twelve Data error")))
        frame = pd.DataFrame(payload["values"])
        numeric_columns = ["open", "high", "low", "close"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        else:
            frame["volume"] = 0.0
        frame["date"] = pd.to_datetime(frame["datetime"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame[["date", "open", "high", "low", "close", "volume"]]

    def _load_twelve_data_intraday(self, symbol: str, interval: str, candles: int, api_key: str) -> pd.DataFrame:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": max(30, min(candles, 5000)),
                "apikey": api_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise ValueError(str(payload.get("message", "Twelve Data error")))
        frame = pd.DataFrame(payload["values"])
        numeric_columns = ["open", "high", "low", "close"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        else:
            frame["volume"] = 0.0
        frame["date"] = pd.to_datetime(frame["datetime"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame[["date", "open", "high", "low", "close", "volume"]]



def _coinbase_granularity(interval: str) -> int:
    return {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }.get(interval, 3600)


def _kraken_interval_minutes(interval: str) -> int:
    return {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(interval, 60)


def _yfinance_fetch_interval(interval: str) -> str:
    return "1h" if interval in {"2h", "4h", "6h", "12h"} else interval


def _yfinance_period(days: int) -> str:
    return f"{max(30, min(days + 5, 3650))}d"


def _yfinance_intraday_period(candles: int, interval: str) -> str:
    hours_per_candle = _interval_milliseconds(interval) / 3_600_000
    days = int(max(30, (candles * hours_per_candle / 6) + 20))
    # yfinance intraday futures/equity feeds are more reliable below 60 days.
    # Higher synthetic intervals are fetched as 1h and resampled locally.
    return f"{min(days, 59)}d"


def _interval_milliseconds(interval: str) -> int:
    return {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }.get(interval, 3_600_000)

def _resample_ohlc(prices: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval not in {"2h", "4h", "6h", "12h"}:
        return prices.sort_values("date").reset_index(drop=True)
    rule = {"2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h"}[interval]
    frame = prices.copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.floor(rule)
    grouped = frame.groupby("date", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return grouped.dropna(subset=["open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


