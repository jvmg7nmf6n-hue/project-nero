"""
Gold Real Yield Engine
=======================

Purpose
-------
Estimate macro pressure on Gold using the classic real-yield framework:

    real_yield â‰ˆ nominal_10Y_yield - breakeven_inflation

Falling / negative real yields are historically supportive of gold
(opportunity cost of holding a non-yielding asset falls); rising real
yields plus a strengthening dollar are historically a headwind.

Data reality
------------
Free breakeven-inflation series (e.g. FRED T10YIE) aren't always reachable
from a sandboxed app, so this engine supports two modes:

1. "official" mode â€” caller supplies nominal yield + breakeven inflation
   directly (e.g. pulled from FRED).
2. "proxy" mode â€” caller supplies only yfinance-style tickers:
     ^TNX  -> 10Y nominal yield proxy (note: ^TNX quotes *10, e.g. 42.5 = 4.25%)
     TIP   -> inflation-protected bond ETF, used as an inverse-ish inflation
              expectations proxy via its price trend
     UUP / DX-Y.NYB -> dollar index proxy
     GLD / PAXG / XAUUSD -> gold price proxy

Both modes flow through the same pure scoring function, and the result is
always explicitly labeled proxy vs official so nothing downstream is
misled about data quality.

This module is for research / macro-context only. It does not predict
gold prices and does not guarantee any outcome.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_BARS_REQUIRED = 15
CORRELATION_WINDOW = 20
DXY_RETURN_WINDOW = 5

LABELS = (
    "GOLD_MACRO_SUPPORTIVE",
    "GOLD_MACRO_NEUTRAL",
    "GOLD_MACRO_PRESSURE",
    "DATA_INSUFFICIENT",
)


@dataclass
class GoldRealYieldResult:
    real_yield_score: float  # 0-100, higher = more supportive of gold
    real_yield_label: str
    latest_nominal_yield: Optional[float]
    latest_inflation_proxy_or_breakeven: Optional[float]
    estimated_real_yield: Optional[float]
    dxy_pressure: Optional[str]  # RISING / FALLING / FLAT / UNKNOWN
    gold_real_yield_correlation: Optional[float]
    notes: List[str]
    is_proxy: bool = True

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)


def _safe_pct_return(series: pd.Series, periods: int) -> Optional[float]:
    series = series.dropna()
    if len(series) <= periods:
        return None
    start, end = series.iloc[-periods - 1], series.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return float((end / start - 1.0) * 100.0)


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
    return None if pd.isna(corr) else float(corr)


def compute_gold_real_yield_score(
    nominal_yield_series: Optional[pd.Series] = None,
    breakeven_inflation_series: Optional[pd.Series] = None,
    tip_price_series: Optional[pd.Series] = None,
    dxy_price_series: Optional[pd.Series] = None,
    gold_price_series: Optional[pd.Series] = None,
    tnx_is_scaled_x10: bool = True,
) -> GoldRealYieldResult:
    """
    Pure, network-free scoring function.

    Provide EITHER:
      - nominal_yield_series (as a %, e.g. 4.25) AND breakeven_inflation_series (as a %),
    OR (proxy mode):
      - nominal_yield_series populated from ^TNX (tnx_is_scaled_x10=True by default,
        i.e. raw ^TNX value / 10 = percent) AND tip_price_series (used as an
        inflation-expectations proxy: rising TIP price/falling TIP yield trend implies
        falling real yields â€” approximated here via TIP's own price momentum).

    dxy_price_series and gold_price_series are optional context inputs used for
    dxy_pressure and gold_real_yield_correlation.
    """
    notes: List[str] = []

    if nominal_yield_series is None or nominal_yield_series.dropna().empty:
        return GoldRealYieldResult(
            0.0, "DATA_INSUFFICIENT", None, None, None, None, None,
            notes=["No nominal yield data supplied."],
        )

    nominal_series = nominal_yield_series.dropna()
    if len(nominal_series) < 3:
        return GoldRealYieldResult(
            0.0, "DATA_INSUFFICIENT", None, None, None, None, None,
            notes=["Nominal yield series too short."],
        )

    latest_nominal = float(nominal_series.iloc[-1])
    if tnx_is_scaled_x10 and latest_nominal > 25:
        # heuristic: ^TNX values look like 42.xx representing 4.2xx%
        latest_nominal = latest_nominal / 10.0
        notes.append("Nominal yield interpreted from ^TNX-style scaled quote (/10).")

    inflation_value: Optional[float] = None
    real_yield: Optional[float] = None
    is_proxy = True

    if breakeven_inflation_series is not None and not breakeven_inflation_series.dropna().empty:
        inflation_value = float(breakeven_inflation_series.dropna().iloc[-1])
        real_yield = latest_nominal - inflation_value
        is_proxy = False
    elif tip_price_series is not None and not tip_price_series.dropna().empty:
        tip = tip_price_series.dropna()
        if len(tip) >= MIN_BARS_REQUIRED:
            tip_mom = _safe_pct_return(tip, 20)  # ~1 month momentum
            if tip_mom is not None:
                # Rising TIP price momentum is *loosely* associated with rising
                # inflation expectations / falling real yields; this is a coarse
                # proxy, clearly labeled as such.
                inflation_value = round(2.0 + (tip_mom / 5.0), 3)  # anchor near 2% CPI target
                real_yield = latest_nominal - inflation_value
                notes.append("Inflation expectations proxied from TIP price momentum "
                              "(anchored near 2% target) â€” not a true breakeven rate.")
            else:
                notes.append("TIP momentum could not be computed; inflation proxy unavailable.")
        else:
            notes.append("TIP price history too short for inflation proxy.")
    else:
        notes.append("No breakeven inflation or TIP proxy supplied; real yield left unestimated.")

    dxy_pressure = "UNKNOWN"
    if dxy_price_series is not None and not dxy_price_series.dropna().empty:
        dxy = dxy_price_series.dropna()
        if len(dxy) > DXY_RETURN_WINDOW:
            dxy_ret = _safe_pct_return(dxy, DXY_RETURN_WINDOW)
            if dxy_ret is not None:
                if dxy_ret > 0.5:
                    dxy_pressure = "RISING"
                elif dxy_ret < -0.5:
                    dxy_pressure = "FALLING"
                else:
                    dxy_pressure = "FLAT"

    gold_corr = None
    if gold_price_series is not None and not gold_price_series.dropna().empty:
        gold_corr = _rolling_correlation_last(
            gold_price_series.dropna(), nominal_series, CORRELATION_WINDOW
        )

    if real_yield is None:
        return GoldRealYieldResult(
            0.0, "DATA_INSUFFICIENT",
            round(latest_nominal, 3), None, None, dxy_pressure, gold_corr,
            notes=notes + ["Real yield could not be estimated from available inputs."],
        )

    # Score: lower/more-negative real yield => more gold-supportive.
    # Map real_yield in roughly [-2%, +3%] onto a 0-100 supportive score.
    score = 100 - ((real_yield - (-2.0)) / (3.0 - (-2.0))) * 100
    score = float(np.clip(score, 0, 100))

    # Dollar strength adjustment: a rising dollar is an added headwind for gold.
    if dxy_pressure == "RISING":
        score = max(0.0, score - 8.0)
        notes.append("Dollar strength (DXY proxy rising) adds headwind pressure on gold.")
    elif dxy_pressure == "FALLING":
        score = min(100.0, score + 8.0)
        notes.append("Dollar weakness (DXY proxy falling) adds tailwind support for gold.")

    if score >= 65:
        label = "GOLD_MACRO_SUPPORTIVE"
        notes.append(f"Estimated real yield ({real_yield:.2f}%) is low/negative â€” "
                      "historically supportive of gold.")
    elif score >= 40:
        label = "GOLD_MACRO_NEUTRAL"
        notes.append(f"Estimated real yield ({real_yield:.2f}%) is middling â€” "
                      "no strong macro push either way.")
    else:
        label = "GOLD_MACRO_PRESSURE"
        notes.append(f"Estimated real yield ({real_yield:.2f}%) is elevated â€” "
                      "historically a headwind for gold.")

    if is_proxy:
        notes.append("NOTE: real yield computed via yfinance PROXY tickers, not official "
                      "FRED 10Y/breakeven data. Treat as directional context only.")

    return GoldRealYieldResult(
        real_yield_score=round(score, 2),
        real_yield_label=label,
        latest_nominal_yield=round(latest_nominal, 3),
        latest_inflation_proxy_or_breakeven=round(inflation_value, 3) if inflation_value is not None else None,
        estimated_real_yield=round(real_yield, 3),
        dxy_pressure=dxy_pressure,
        gold_real_yield_correlation=round(gold_corr, 3) if gold_corr is not None else None,
        notes=notes,
        is_proxy=is_proxy,
    )


def fetch_gold_real_yield_score(period: str = "6mo", interval: str = "1d") -> GoldRealYieldResult:
    """
    Fetch live data via yfinance/FRED-style tickers and compute the score.
    Never raises â€” degrades to DATA_INSUFFICIENT on failure. Not covered by
    unit tests (network I/O); tests use compute_gold_real_yield_score directly.
    """
    try:
        import yfinance as yf  # local import: optional dependency
    except Exception as exc:  # pragma: no cover
        return GoldRealYieldResult(0.0, "DATA_INSUFFICIENT", None, None, None, None, None,
                                    [f"yfinance not available: {exc}"])

    def _hist_close(ticker: str) -> Optional[pd.Series]:
        try:
            h = yf.Ticker(ticker).history(period=period, interval=interval)
            if h is not None and not h.empty:
                return h["Close"]
        except Exception:  # pragma: no cover
            return None
        return None

    tnx = _hist_close("^TNX")
    tip = _hist_close("TIP")
    dxy = _hist_close("DX-Y.NYB")
    if dxy is None:
        dxy = _hist_close("UUP")
    gold = _hist_close("GLD")

    if tnx is None:
        return GoldRealYieldResult(0.0, "DATA_INSUFFICIENT", None, None, None, None, None,
                                    ["Could not fetch ^TNX nominal yield proxy."])

    return compute_gold_real_yield_score(
        nominal_yield_series=tnx,
        breakeven_inflation_series=None,
        tip_price_series=tip,
        dxy_price_series=dxy,
        gold_price_series=gold,
        tnx_is_scaled_x10=True,
    )

