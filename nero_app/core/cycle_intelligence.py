from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import pandas as pd

if TYPE_CHECKING:
    from nero_app.core.market_data import MarketDataClient


CYCLE_ASSET_MAP: dict[str, str] = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "PAXG": "PAXG",
}

GOLD_PROXY_NOTE = "PAXG is tokenized gold. It is not spot XAU/USD."


@dataclass(frozen=True)
class CycleIntelligenceReport:
    rows: list[dict[str, object]]
    availability_rows: list[dict[str, str]]
    correlation_rows: list[dict[str, object]]
    notes: list[str]


def build_cycle_intelligence_report(
    assets: list[str] | None = None,
    price_frames: Mapping[str, pd.DataFrame] | None = None,
    provided_sources: Mapping[str, tuple[str, str]] | None = None,
    prefer_live: bool = True,
    days: int = 1000,
    market_client: "MarketDataClient | None" = None,
) -> CycleIntelligenceReport:
    """Build price-only cycle metrics without inventing unavailable data."""
    selected_assets = assets or list(CYCLE_ASSET_MAP)
    client = market_client
    client_import_error = ""
    if client is None and not price_frames:
        try:
            from nero_app.core.market_data import MarketDataClient

            client = MarketDataClient()
        except ModuleNotFoundError as exc:
            client_import_error = f"market data dependency missing: {exc.name}"
    rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    notes: list[str] = [
        "Cycle Intelligence is research context only. It does not change live entry, quarantine, promotion, confidence, or scoring rules.",
        "Tier A uses daily closes only: Mayer Multiple, percentile, SMA200 slope, and drawdown from available high.",
    ]

    for raw_asset in selected_assets:
        asset = _normalize_cycle_asset(raw_asset)
        source_asset = CYCLE_ASSET_MAP.get(asset)
        if source_asset is None:
            rows.append(_unavailable_row(raw_asset, "asset is not configured for Cycle Intelligence"))
            continue

        if price_frames and raw_asset in price_frames:
            prices = price_frames[raw_asset]
            source, status = (provided_sources or {}).get(raw_asset, ("provided daily candles", "provided"))
        elif price_frames and asset in price_frames:
            prices = price_frames[asset]
            source, status = (provided_sources or {}).get(asset, ("provided daily candles", "provided"))
        else:
            if client is None:
                rows.append(_unavailable_row(raw_asset, client_import_error or "no market client or provided price frame available"))
                continue
            result = client.load(source_asset, prefer_live=prefer_live, days=days)
            prices = result.prices
            source = result.source
            status = result.status

        row, prepared = build_cycle_intelligence_row(raw_asset, prices, source=source, status=status)
        rows.append(row)
        correlation_rows.extend(build_cycle_correlation_rows(raw_asset, prepared))

    availability_rows = build_layer_availability_report()
    return CycleIntelligenceReport(rows=rows, availability_rows=availability_rows, correlation_rows=correlation_rows, notes=notes)


def build_cycle_intelligence_row(asset: str, price_history: pd.DataFrame, source: str, status: str) -> tuple[dict[str, object], pd.DataFrame]:
    cycle_asset = _normalize_cycle_asset(asset)
    prepared = _prepare_daily_prices(price_history)
    data_method = _data_method(source, status)
    utc_boundary = "UTC midnight from exchange/provider daily candle timestamps"
    missing: list[str] = []

    if prepared.empty:
        return _unavailable_row(asset, "no usable daily close data", source=source, status=status, data_method=data_method), prepared

    close = pd.to_numeric(prepared["close"], errors="coerce").dropna()
    total_daily_closes = int(len(close))
    first_daily_close = _date_string(prepared.iloc[0]["date"]) if "date" in prepared else ""
    latest_daily_close = _date_string(prepared.iloc[-1]["date"]) if "date" in prepared else ""

    if _is_untrusted_source(source, status):
        return _unavailable_row(
            asset,
            "source is fallback/sample; Cycle Intelligence requires real daily close data",
            source=source,
            status=status,
            total_daily_closes=total_daily_closes,
            first_daily_close=first_daily_close,
            latest_daily_close=latest_daily_close,
            data_method=data_method,
        ), prepared

    if total_daily_closes < 200:
        return _unavailable_row(
            asset,
            f"needs at least 200 clean daily closes; found {total_daily_closes}",
            source=source,
            status=status,
            total_daily_closes=total_daily_closes,
            first_daily_close=first_daily_close,
            latest_daily_close=latest_daily_close,
            data_method=data_method,
        ), prepared

    sma_200_series = close.rolling(200).mean()
    sma_200 = float(sma_200_series.iloc[-1])
    latest_close = float(close.iloc[-1])
    mayer_multiple = latest_close / sma_200 if sma_200 else None
    distance_pct = ((latest_close - sma_200) / sma_200) * 100 if sma_200 else None

    mm_history = (close / sma_200_series).replace([float("inf"), float("-inf")], pd.NA).dropna()
    computable_history_days = int(len(mm_history))
    if mm_history.empty or mayer_multiple is None:
        missing.append("mayer_multiple_history")
        mm_percentile = None
        mm_percentile_label = "UNAVAILABLE"
    else:
        mm_percentile = float((mm_history <= mayer_multiple).mean() * 100.0)
        mm_percentile_label = _classify_percentile(mm_percentile)

    if len(sma_200_series.dropna()) < 31:
        missing.append("sma200_slope_30d")
        slope_value = None
        slope_pct = None
        slope_label = "UNAVAILABLE"
    else:
        prior_sma = float(sma_200_series.dropna().iloc[-31])
        slope_value = sma_200 - prior_sma
        slope_pct = (slope_value / prior_sma) * 100.0 if prior_sma else None
        slope_label = _classify_slope(slope_pct)

    historical_high = float(close.max())
    drawdown_pct = ((latest_close - historical_high) / historical_high) * 100.0 if historical_high else None
    drawdown_label = _classify_drawdown(drawdown_pct)

    row = {
        "asset": asset,
        "cycle_asset": cycle_asset,
        "symbol": _symbol_for_cycle_asset(cycle_asset),
        "source": source,
        "status": "OK",
        "unavailable_reason": "",
        "missing_data_fields": ", ".join(missing),
        "latest_close": latest_close,
        "sma_200": sma_200,
        "mayer_multiple": mayer_multiple,
        "distance_from_sma200_pct": distance_pct,
        "mm_percentile_rank": mm_percentile,
        "mm_percentile_label": mm_percentile_label,
        "sma200_slope_value_30d": slope_value,
        "sma200_slope_pct_30d": slope_pct,
        "sma200_slope_label": slope_label,
        "historical_high_available": historical_high,
        "drawdown_from_high_pct": drawdown_pct,
        "drawdown_label": drawdown_label,
        "total_daily_closes": total_daily_closes,
        "computable_history_days": computable_history_days,
        "history_days_used": total_daily_closes,
        "first_daily_close": first_daily_close,
        "latest_daily_close": latest_daily_close,
        "data_method": data_method,
        "utc_boundary": utc_boundary,
        "consumer": "dashboard Cycle Intelligence tab; reports/cycle_intelligence_report.*",
        "paxg_caveat": GOLD_PROXY_NOTE if cycle_asset == "PAXG" else "",
    }
    return row, prepared.assign(sma_200=sma_200_series, mayer_multiple=close / sma_200_series)


def build_layer_availability_report() -> list[dict[str, str]]:
    etf_source = _configured_source("BTC_ETF_FLOW_CSV_PATH", "BTC_ETF_FLOW_CSV_URL")
    real_yield_source = _configured_source("GOLD_REAL_YIELD_CSV_PATH", "GOLD_REAL_YIELD_CSV_URL")
    return [
        {
            "layer": "MM Percentile Engine",
            "status": "REAL",
            "raw_inputs": "daily close, SMA200",
            "consumer": "Cycle Intelligence dashboard/report",
            "reason": "Pure price arithmetic from real daily close history.",
        },
        {
            "layer": "200D Slope Engine",
            "status": "REAL",
            "raw_inputs": "daily close, rolling SMA200",
            "consumer": "Cycle Intelligence dashboard/report",
            "reason": "Pure price arithmetic; unavailable per asset if less than 230 daily closes.",
        },
        {
            "layer": "Drawdown From High Engine",
            "status": "REAL",
            "raw_inputs": "daily close history",
            "consumer": "Cycle Intelligence dashboard/report",
            "reason": "Uses highest close inside available clean history; not claimed as full lifetime ATH unless data covers full life.",
        },
        {
            "layer": "ETF Flow Confirmation",
            "status": "REAL" if etf_source else "UNAVAILABLE",
            "raw_inputs": etf_source or "none configured",
            "consumer": "Not consumed by Cycle score; raw pass-through only when configured.",
            "reason": "No ETF proxy fallback is allowed in Cycle Intelligence.",
        },
        {
            "layer": "Liquidity Pressure Layer",
            "status": "PARTIAL" if real_yield_source else "UNAVAILABLE",
            "raw_inputs": real_yield_source or "DXY/SPX/yields not fully configured as official synchronized daily inputs",
            "consumer": "Not consumed by Cycle score.",
            "reason": "Composite liquidity score is intentionally not built in this tier.",
        },
        {
            "layer": "Derivatives Heat Layer",
            "status": "UNAVAILABLE",
            "raw_inputs": "funding/open-interest/liquidation feed not configured",
            "consumer": "Not consumed by Cycle score.",
            "reason": "No derivative heat proxy is built.",
        },
        {
            "layer": "Cycle Similarity Memory",
            "status": "NOT_BUILT",
            "raw_inputs": "requires separately validated historical regimes",
            "consumer": "none",
            "reason": "Tier C is explicitly deferred by the directive.",
        },
    ]


def build_cycle_correlation_rows(asset: str, prepared_prices: pd.DataFrame) -> list[dict[str, object]]:
    if prepared_prices.empty or "mayer_multiple" not in prepared_prices.columns:
        return []
    fields = {
        "close": pd.to_numeric(prepared_prices.get("close"), errors="coerce"),
        "sma_200": pd.to_numeric(prepared_prices.get("sma_200"), errors="coerce"),
        "mayer_multiple": pd.to_numeric(prepared_prices.get("mayer_multiple"), errors="coerce"),
    }
    if "close" in prepared_prices:
        close = pd.to_numeric(prepared_prices["close"], errors="coerce")
        high_to_date = close.cummax()
        fields["distance_from_sma200_pct"] = ((close - fields["sma_200"]) / fields["sma_200"]) * 100.0
        fields["drawdown_from_high_pct"] = ((close - high_to_date) / high_to_date) * 100.0
    frame = pd.DataFrame(fields).dropna()
    rows: list[dict[str, object]] = []
    columns = list(frame.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            pair = frame[[left, right]].dropna()
            if len(pair) >= 3 and pair[left].std() > 0 and pair[right].std() > 0:
                corr = pair[left].corr(pair[right])
            else:
                corr = None
            rows.append(
                {
                    "asset": asset,
                    "field_x": left,
                    "field_y": right,
                    "n": int(len(pair)),
                    "correlation": None if corr is None or pd.isna(corr) else float(corr),
                    "note": "Price-derived redundancy check; MM and SMA distance should be near-perfectly related.",
                }
            )
    return rows


def cycle_dashboard_rows(row: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"Field": "Latest Close", "Reading": _format_number(row.get("latest_close")), "Meaning": "Latest clean daily close used for cycle context."},
        {"Field": "SMA200", "Reading": _format_number(row.get("sma_200")), "Meaning": "Long-cycle mean baseline."},
        {"Field": "Mayer Multiple", "Reading": _format_number(row.get("mayer_multiple"), 3), "Meaning": "Close divided by SMA200. Above 1 means price is above its 200D average."},
        {"Field": "Distance from SMA200", "Reading": _format_pct_value(row.get("distance_from_sma200_pct")), "Meaning": "Same relationship as Mayer Multiple, shown as distance."},
        {"Field": "MM Percentile", "Reading": _format_pct_value(row.get("mm_percentile_rank")), "Meaning": "Current MM rank versus available computable MM history."},
        {"Field": "SMA200 Slope 30D", "Reading": _format_pct_value(row.get("sma200_slope_pct_30d")), "Meaning": "Whether the 200D baseline is rising or falling."},
        {"Field": "Drawdown from Available High", "Reading": _format_pct_value(row.get("drawdown_from_high_pct")), "Meaning": "Distance from highest close inside available history."},
        {"Field": "History Used", "Reading": str(row.get("history_days_used", "n/a")), "Meaning": "Clean daily closes used; MM needs 200-day warmup."},
    ]


def _prepare_daily_prices(price_history: pd.DataFrame) -> pd.DataFrame:
    if price_history is None or price_history.empty or "close" not in price_history.columns:
        return pd.DataFrame(columns=["date", "close"])
    frame = price_history.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        frame["date"] = pd.RangeIndex(len(frame))
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["daily_date"] = frame["date"].dt.floor("D")
        frame = frame.groupby("daily_date", as_index=False).agg(date=("date", "max"), close=("close", "last"))
    return frame[["date", "close"]].dropna().reset_index(drop=True)


def _unavailable_row(
    asset: str,
    reason: str,
    source: str = "",
    status: str = "UNAVAILABLE",
    total_daily_closes: int = 0,
    first_daily_close: str = "",
    latest_daily_close: str = "",
    data_method: str = "daily close",
) -> dict[str, object]:
    cycle_asset = _normalize_cycle_asset(asset)
    return {
        "asset": asset,
        "cycle_asset": cycle_asset,
        "symbol": _symbol_for_cycle_asset(cycle_asset),
        "source": source,
        "status": "UNAVAILABLE",
        "unavailable_reason": reason,
        "missing_data_fields": "latest_close, sma_200, mayer_multiple, distance_from_sma200_pct, mm_percentile_rank, sma200_slope_pct_30d, drawdown_from_high_pct",
        "latest_close": None,
        "sma_200": None,
        "mayer_multiple": None,
        "distance_from_sma200_pct": None,
        "mm_percentile_rank": None,
        "mm_percentile_label": "UNAVAILABLE",
        "sma200_slope_value_30d": None,
        "sma200_slope_pct_30d": None,
        "sma200_slope_label": "UNAVAILABLE",
        "historical_high_available": None,
        "drawdown_from_high_pct": None,
        "drawdown_label": "UNAVAILABLE",
        "total_daily_closes": total_daily_closes,
        "computable_history_days": 0,
        "history_days_used": total_daily_closes,
        "first_daily_close": first_daily_close,
        "latest_daily_close": latest_daily_close,
        "data_method": data_method,
        "utc_boundary": "UTC midnight from exchange/provider daily candle timestamps",
        "consumer": "dashboard Cycle Intelligence tab; reports/cycle_intelligence_report.*",
        "paxg_caveat": GOLD_PROXY_NOTE if cycle_asset == "PAXG" else "",
    }


def _normalize_cycle_asset(asset: str) -> str:
    upper = str(asset).upper()
    return "PAXG" if upper == "GOLD" else upper


def _symbol_for_cycle_asset(asset: str) -> str:
    return {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "PAXG": "PAXGUSDT",
    }.get(asset, asset)


def _data_method(source: str, status: str) -> str:
    lower = f"{source} {status}".lower()
    if "resample" in lower:
        return "resampled daily close"
    return "true daily close"


def _is_untrusted_source(source: str, status: str) -> bool:
    lower = f"{source} {status}".lower()
    return "fallback" in lower or "generated sample" in lower or lower.strip() == "sample"


def _configured_source(path_key: str, url_key: str) -> str:
    path = os.getenv(path_key, "").strip()
    url = os.getenv(url_key, "").strip()
    if path:
        return f"{path_key}={Path(path).name}"
    if url:
        return f"{url_key}=configured"
    return ""


def _classify_percentile(value: float | None) -> str:
    if value is None:
        return "UNAVAILABLE"
    if value >= 90:
        return "MM_EXTREME"
    if value >= 75:
        return "MM_HIGH"
    if value <= 20:
        return "MM_LOW"
    return "MM_MID"


def _classify_slope(value: float | None) -> str:
    if value is None:
        return "UNAVAILABLE"
    if value > 1:
        return "SMA200_RISING"
    if value < -1:
        return "SMA200_FALLING"
    return "SMA200_FLAT"


def _classify_drawdown(value: float | None) -> str:
    if value is None:
        return "UNAVAILABLE"
    if value <= -60:
        return "DEEP_CYCLE_DRAWDOWN"
    if value <= -30:
        return "MID_CYCLE_DRAWDOWN"
    if value <= -10:
        return "SHALLOW_DRAWDOWN"
    return "NEAR_AVAILABLE_HIGH"


def _date_string(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _format_number(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def _format_pct_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"
