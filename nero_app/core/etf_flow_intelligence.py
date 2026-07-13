"""
ETF Flow Intelligence Engine
=============================

Purpose
-------
Detect whether BTC price movement is supported by institutional spot-ETF
demand, using a transparent, fully-documented proxy methodology.

There is no free, real-time "creations/redemptions" API for spot BTC ETFs.
Rather than silently guessing or hard-failing, this engine builds an
**explicit proxy score** out of publicly observable market data (price,
volume, correlation with BTC) and always labels itself as a PROXY so
downstream consumers (dashboard, scanner) never mistake it for real
flow data.

Design goals
------------
- Deterministic: given the same input DataFrames, always the same output.
- Network-free at the core: `compute_etf_flow_score()` takes DataFrames in;
  it never calls yfinance itself. A thin `fetch_*` wrapper at the bottom
  does the network I/O and is the only part that can fail/needs mocking.
- Never raises on bad/missing data -> degrades to DATA_INSUFFICIENT.

This module is for research / paper-trading decision support only.
It does not place orders and does not guarantee any outcome.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from io import StringIO
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_ETF_TICKERS: List[str] = ["IBIT", "FBTC", "GBTC", "BITB", "ARKB", "HODL"]

# Minimum number of daily bars required per ETF for the engine to trust it.
MIN_BARS_REQUIRED = 15

# Rolling window used for the abnormal-volume z-score.
VOLUME_Z_WINDOW = 20

# Rolling window used for the ETF/BTC return correlation.
CORRELATION_WINDOW = 20


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class EtfEvidenceRow:
    ticker: str
    latest_close: Optional[float]
    latest_volume: Optional[float]
    volume_zscore: Optional[float]
    correlation_with_btc: Optional[float]
    return_5d_pct: Optional[float]
    flow_proxy_reading: str  # INFLOW_LIKELY / OUTFLOW_LIKELY / NEUTRAL / INSUFFICIENT
    actual_flow_usd: Optional[float] = None
    flow_date: Optional[str] = None
    data_type: str = "proxy"  # proxy / actual

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)


@dataclass
class EtfFlowResult:
    etf_flow_score: float  # 0-100
    etf_flow_label: str
    dominant_etf: Optional[str]
    evidence: List[EtfEvidenceRow]
    notes: List[str]
    is_proxy: bool = True

    def evidence_frame(self) -> pd.DataFrame:
        if not self.evidence:
            return pd.DataFrame(
                columns=[
                    "ticker", "latest_close", "latest_volume", "volume_zscore",
                    "correlation_with_btc", "return_5d_pct", "flow_proxy_reading",
                    "actual_flow_usd", "flow_date", "data_type",
                ]
            )
        return pd.DataFrame([row.as_dict() for row in self.evidence])

    def as_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        d["evidence"] = [row.as_dict() for row in self.evidence]
        return d


LABELS = (
    "STRONG_INFLOW_PRESSURE",
    "MODERATE_INFLOW_PRESSURE",
    "NEUTRAL_FLOW",
    "OUTFLOW_PRESSURE",
    "DATA_INSUFFICIENT",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_pct_return(series: pd.Series, periods: int) -> Optional[float]:
    series = series.dropna()
    if len(series) <= periods:
        return None
    start, end = series.iloc[-periods - 1], series.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return float((end / start - 1.0) * 100.0)


def _rolling_zscore_last(series: pd.Series, window: int) -> Optional[float]:
    series = series.dropna()
    if len(series) < window:
        return None
    recent = series.iloc[-window:]
    std = recent.std(ddof=0)
    if std == 0 or pd.isna(std):
        return None
    return float((recent.iloc[-1] - recent.mean()) / std)


def _rolling_correlation_last(a: pd.Series, b: pd.Series, window: int) -> Optional[float]:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < window:
        return None
    recent = df.iloc[-window:]
    a_ret = recent.iloc[:, 0].pct_change().dropna()
    b_ret = recent.iloc[:, 1].pct_change().dropna()
    joined = pd.concat([a_ret, b_ret], axis=1).dropna()
    if len(joined) < 3:
        return None
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if pd.isna(corr):
        return None
    return float(corr)


def _classify_single_etf(
    volume_z: Optional[float],
    ret_5d: Optional[float],
    corr: Optional[float],
) -> str:
    if volume_z is None or ret_5d is None:
        return "INSUFFICIENT"
    # Abnormally high volume + positive price action + positive co-movement
    # with BTC => consistent with net creations (inflow-like behaviour).
    if volume_z >= 1.0 and ret_5d > 0:
        return "INFLOW_LIKELY"
    if volume_z >= 1.0 and ret_5d < 0:
        return "OUTFLOW_LIKELY"
    if ret_5d < -1.5 and (corr is None or corr > 0.3):
        return "OUTFLOW_LIKELY"
    if ret_5d > 1.5 and (corr is None or corr > 0.3):
        return "INFLOW_LIKELY"
    return "NEUTRAL"




# --------------------------------------------------------------------------- #
# Actual ETF flow scoring (CSV/API-fed, no proxy assumptions)
# --------------------------------------------------------------------------- #

def compute_actual_etf_flow_score(flow_data: pd.DataFrame, lookback_days: int = 5) -> EtfFlowResult:
    """Compute BTC ETF flow score from actual net-flow rows.

    Expected columns are flexible but must include a date, ticker/fund, and one
    net-flow column. Supported flow units: USD or USD millions.
    """
    notes: List[str] = []
    if flow_data is None or flow_data.empty:
        return EtfFlowResult(0.0, "DATA_INSUFFICIENT", None, [], ["No actual ETF flow rows provided."], is_proxy=False)

    frame = _normalise_actual_flow_frame(flow_data)
    if frame.empty:
        return EtfFlowResult(
            0.0,
            "DATA_INSUFFICIENT",
            None,
            [],
            ["Actual ETF flow data could not be parsed. Required: date, ticker/fund, flow_usd or flow_musd."],
            is_proxy=False,
        )

    latest_date = frame["date"].max()
    cutoff = latest_date - pd.Timedelta(days=max(1, lookback_days) - 1)
    window = frame[frame["date"] >= cutoff].copy()
    if window.empty:
        return EtfFlowResult(0.0, "DATA_INSUFFICIENT", None, [], ["No actual ETF flow rows in lookback window."], is_proxy=False)

    by_ticker = window.groupby("ticker", as_index=False)["flow_usd"].sum()
    total_flow = float(by_ticker["flow_usd"].sum())
    dominant = None
    if not by_ticker.empty:
        dominant_row = by_ticker.iloc[by_ticker["flow_usd"].abs().argmax()]
        dominant = str(dominant_row["ticker"])

    score = _actual_flow_score(total_flow)
    label = _actual_flow_label(score)
    evidence = [
        EtfEvidenceRow(
            ticker=str(row["ticker"]),
            latest_close=None,
            latest_volume=None,
            volume_zscore=None,
            correlation_with_btc=None,
            return_5d_pct=None,
            flow_proxy_reading=_actual_flow_reading(float(row["flow_usd"])),
            actual_flow_usd=round(float(row["flow_usd"]), 2),
            flow_date=latest_date.date().isoformat(),
            data_type="actual",
        )
        for _, row in by_ticker.sort_values("flow_usd", key=lambda s: s.abs(), ascending=False).iterrows()
    ]
    notes.append(f"Actual BTC ETF net flow used: {total_flow:,.0f} USD over the last {lookback_days} calendar day(s).")
    notes.append("This is actual net-flow input, not the price/volume proxy. Treat provider coverage and timing as a data-quality dependency.")
    return EtfFlowResult(round(score, 2), label, dominant, evidence, notes, is_proxy=False)


def _normalise_actual_flow_frame(flow_data: pd.DataFrame) -> pd.DataFrame:
    frame = flow_data.copy()
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    date_col = _first_existing(frame, ["date", "day", "as_of", "as_of_date"])
    ticker_col = _first_existing(frame, ["ticker", "fund", "symbol", "etf"])
    usd_col = _first_existing(frame, ["flow_usd", "net_flow_usd", "netflow_usd", "usd", "flow"])
    musd_col = _first_existing(frame, ["flow_musd", "net_flow_musd", "flow_usd_millions", "usd_millions", "million_usd", "flow_million_usd"])
    if not date_col or not ticker_col or (not usd_col and not musd_col):
        return pd.DataFrame(columns=["date", "ticker", "flow_usd"])
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    out["ticker"] = frame[ticker_col].astype(str).str.upper().str.strip()
    raw_flow = frame[usd_col] if usd_col else frame[musd_col]
    flow = pd.to_numeric(raw_flow.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False), errors="coerce")
    if musd_col and not usd_col:
        flow = flow * 1_000_000.0
    out["flow_usd"] = flow
    out = out.dropna(subset=["date", "flow_usd"])
    out = out[out["ticker"] != ""]
    return out.sort_values("date").reset_index(drop=True)


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _actual_flow_score(total_flow_usd: float) -> float:
    # Scale around +/- $1B net flow as strong pressure. This stays bounded and transparent.
    billions = total_flow_usd / 1_000_000_000.0
    return float(np.clip(50.0 + billions * 35.0, 0.0, 100.0))


def _actual_flow_label(score: float) -> str:
    if score >= 75:
        return "STRONG_INFLOW_PRESSURE"
    if score >= 60:
        return "MODERATE_INFLOW_PRESSURE"
    if score > 40:
        return "NEUTRAL_FLOW"
    return "OUTFLOW_PRESSURE"


def _actual_flow_reading(flow_usd: float) -> str:
    if flow_usd > 25_000_000:
        return "INFLOW_ACTUAL"
    if flow_usd < -25_000_000:
        return "OUTFLOW_ACTUAL"
    return "NEUTRAL_ACTUAL"


def fetch_actual_etf_flow_score(csv_path: str | None = None, csv_url: str | None = None, lookback_days: int = 5) -> EtfFlowResult:
    """Fetch actual ETF net-flow data from a configured CSV path or URL."""
    csv_path = csv_path or os.getenv("BTC_ETF_FLOW_CSV_PATH", "").strip()
    csv_url = csv_url or os.getenv("BTC_ETF_FLOW_CSV_URL", "").strip()
    try:
        if csv_path:
            path = Path(csv_path).expanduser()
            if not path.exists():
                return EtfFlowResult(0.0, "DATA_INSUFFICIENT", None, [], [f"BTC_ETF_FLOW_CSV_PATH not found: {path}"], is_proxy=False)
            return compute_actual_etf_flow_score(pd.read_csv(path), lookback_days=lookback_days)
        if csv_url:
            import requests
            response = requests.get(csv_url, timeout=20)
            response.raise_for_status()
            return compute_actual_etf_flow_score(pd.read_csv(StringIO(response.text)), lookback_days=lookback_days)
    except Exception as exc:  # pragma: no cover - defensive external I/O
        return EtfFlowResult(0.0, "DATA_INSUFFICIENT", None, [], [f"Actual ETF flow fetch failed: {exc}"], is_proxy=False)
    return EtfFlowResult(0.0, "DATA_INSUFFICIENT", None, [], ["No actual ETF flow CSV path or URL configured."], is_proxy=False)

# --------------------------------------------------------------------------- #
# Core scoring function (pure, no network)
# --------------------------------------------------------------------------- #

def compute_etf_flow_score(
    etf_price_data: Dict[str, pd.DataFrame],
    btc_price_data: Optional[pd.DataFrame],
    etf_tickers: Optional[List[str]] = None,
) -> EtfFlowResult:
    """
    Compute the ETF Flow Intelligence score from OHLCV-style DataFrames.

    Parameters
    ----------
    etf_price_data : dict[ticker -> DataFrame]
        Each DataFrame must contain at least "Close" and "Volume" columns,
        indexed by date (ascending).
    btc_price_data : DataFrame or None
        Must contain a "Close" column indexed by date (ascending).
    etf_tickers : optional list of tickers to restrict/order evaluation.

    Returns
    -------
    EtfFlowResult
    """
    notes: List[str] = []
    tickers = etf_tickers or list(etf_price_data.keys()) or DEFAULT_ETF_TICKERS

    if btc_price_data is None or "Close" not in getattr(btc_price_data, "columns", []):
        notes.append("BTC price series unavailable — cannot compute ETF/BTC correlation.")
        btc_close = None
    else:
        btc_close = btc_price_data["Close"].dropna()
        if len(btc_close) < MIN_BARS_REQUIRED:
            notes.append("BTC price history too short for reliable correlation.")
            btc_close = None

    evidence: List[EtfEvidenceRow] = []
    usable_scores: List[float] = []
    inflow_votes = 0
    outflow_votes = 0

    for ticker in tickers:
        df = etf_price_data.get(ticker) if etf_price_data else None
        if df is None or df.empty or "Close" not in df.columns:
            evidence.append(EtfEvidenceRow(ticker, None, None, None, None, None, "INSUFFICIENT"))
            continue

        close = df["Close"].dropna()
        volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

        if len(close) < MIN_BARS_REQUIRED:
            evidence.append(EtfEvidenceRow(ticker, None, None, None, None, None, "INSUFFICIENT"))
            notes.append(f"{ticker}: insufficient price history (<{MIN_BARS_REQUIRED} bars).")
            continue

        latest_close = float(close.iloc[-1])
        latest_volume = float(volume.iloc[-1]) if len(volume) else None
        vol_z = _rolling_zscore_last(volume, VOLUME_Z_WINDOW) if len(volume) else None
        ret_5d = _safe_pct_return(close, 5)
        corr = _rolling_correlation_last(close, btc_close, CORRELATION_WINDOW) if btc_close is not None else None

        reading = _classify_single_etf(vol_z, ret_5d, corr)

        evidence.append(
            EtfEvidenceRow(
                ticker=ticker,
                latest_close=round(latest_close, 4),
                latest_volume=latest_volume,
                volume_zscore=round(vol_z, 3) if vol_z is not None else None,
                correlation_with_btc=round(corr, 3) if corr is not None else None,
                return_5d_pct=round(ret_5d, 3) if ret_5d is not None else None,
                flow_proxy_reading=reading,
            )
        )

        if reading == "INFLOW_LIKELY":
            inflow_votes += 1
            usable_scores.append(65 + min(35, max(0, (vol_z or 0) * 10)))
        elif reading == "OUTFLOW_LIKELY":
            outflow_votes += 1
            usable_scores.append(35 - min(35, max(0, (abs(vol_z) if vol_z else 0) * 10)))
        elif reading == "NEUTRAL":
            usable_scores.append(50.0)

    if not usable_scores:
        return EtfFlowResult(
            etf_flow_score=0.0,
            etf_flow_label="DATA_INSUFFICIENT",
            dominant_etf=None,
            evidence=evidence,
            notes=notes or ["No usable ETF data available."],
        )

    raw_score = float(np.clip(np.mean(usable_scores), 0, 100))

    # Dominant ETF = highest |volume z-score| among usable rows (proxy for
    # "which ETF is driving the observed flow signal").
    scored_rows = [e for e in evidence if e.volume_zscore is not None]
    dominant_etf = None
    if scored_rows:
        dominant_etf = max(scored_rows, key=lambda r: abs(r.volume_zscore)).ticker

    if raw_score >= 75:
        label = "STRONG_INFLOW_PRESSURE"
    elif raw_score >= 60:
        label = "MODERATE_INFLOW_PRESSURE"
    elif raw_score > 40:
        label = "NEUTRAL_FLOW"
    else:
        label = "OUTFLOW_PRESSURE"

    if inflow_votes > outflow_votes and label in ("NEUTRAL_FLOW", "OUTFLOW_PRESSURE"):
        notes.append("Mixed signal: more ETFs show inflow-like behaviour than outflow-like, "
                      "but aggregate score remains muted.")
    if label in ("STRONG_INFLOW_PRESSURE", "MODERATE_INFLOW_PRESSURE"):
        notes.append("BTC move appears institutionally supported by ETF-proxy demand "
                      "(elevated volume + positive co-movement).")
    elif label == "OUTFLOW_PRESSURE":
        notes.append("BTC move looks weak/unsupported: ETF proxy shows outflow-like "
                      "volume/price behaviour.")
    else:
        notes.append("ETF proxy shows no strong directional flow signal right now.")

    notes.append("NOTE: this is a PROXY (price/volume/correlation based) — not real "
                  "creation/redemption flow data. Treat as directional context only.")

    return EtfFlowResult(
        etf_flow_score=round(raw_score, 2),
        etf_flow_label=label,
        dominant_etf=dominant_etf,
        evidence=evidence,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Optional network-fetching wrapper (kept separate so tests never need it)
# --------------------------------------------------------------------------- #

def fetch_etf_flow_score(
    etf_tickers: Optional[List[str]] = None,
    btc_ticker: str = "BTC-USD",
    period: str = "3mo",
    interval: str = "1d",
) -> EtfFlowResult:
    """
    Fetch live data via yfinance and compute the score. Never raises —
    on any failure returns a DATA_INSUFFICIENT result with an explanatory
    note. Not covered by unit tests (network I/O); tests exercise
    compute_etf_flow_score() directly with synthetic data.
    """
    actual = fetch_actual_etf_flow_score()
    if actual.etf_flow_label != "DATA_INSUFFICIENT":
        return actual

    tickers = etf_tickers or DEFAULT_ETF_TICKERS
    try:
        import yfinance as yf  # local import: optional dependency
    except Exception as exc:  # pragma: no cover
        return EtfFlowResult(
            0.0, "DATA_INSUFFICIENT", None, [],
            [f"yfinance not available: {exc}"],
        )

    etf_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist is not None and not hist.empty:
                etf_data[ticker] = hist[["Close", "Volume"]]
        except Exception:  # pragma: no cover
            continue

    btc_data = None
    try:
        btc_hist = yf.Ticker(btc_ticker).history(period=period, interval=interval)
        if btc_hist is not None and not btc_hist.empty:
            btc_data = btc_hist[["Close"]]
    except Exception:  # pragma: no cover
        btc_data = None

    if not etf_data:
        return EtfFlowResult(
            0.0, "DATA_INSUFFICIENT", None, [],
            ["No ETF data could be fetched from yfinance."],
        )

    proxy = compute_etf_flow_score(etf_data, btc_data, tickers)
    proxy.notes = actual.notes + proxy.notes
    return proxy
