from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_WATCHLIST_PATH = Path("nero_app/data/social_intelligence_watchlist.csv")
DEFAULT_CALL_LEDGER_PATH = Path("nero_app/data/social_call_ledger.csv")

ASSET_ALIASES = {
    "BTC": {"BTC", "Bitcoin", "Crypto"},
    "GOLD": {"Gold", "Commodities", "Forex", "Macro"},
    "SOL": {"SOL", "Altcoins", "Crypto"},
    "ETH": {"ETH", "Ethereum", "Crypto"},
}

CALL_LEDGER_COLUMNS = [
    "call_id", "source_name", "handle", "posted_at", "asset", "direction", "timeframe",
    "entry", "stop", "target", "post_text", "status", "outcome", "r_multiple",
    "evaluated_at", "notes",
]


@dataclass(frozen=True)
class SocialIntelSummary:
    asset: str
    tracked_voices: int
    average_reliability: float
    high_reliability_voices: int
    dominant_styles: list[str]
    caution_flags: list[str]
    note: str


@dataclass(frozen=True)
class SocialCallScorecard:
    handle: str
    source_name: str
    total_calls: int
    evaluated_calls: int
    win_rate: float
    average_r: float
    reliability_score: float


def load_social_watchlist(path: Path = DEFAULT_WATCHLIST_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_social_call_ledger(path: Path = DEFAULT_CALL_LEDGER_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=CALL_LEDGER_COLUMNS)
    try:
        frame = pd.read_csv(path).fillna("")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=CALL_LEDGER_COLUMNS)
    for column in CALL_LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[CALL_LEDGER_COLUMNS]


def save_social_call_ledger(frame: pd.DataFrame, path: Path = DEFAULT_CALL_LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in CALL_LEDGER_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output[CALL_LEDGER_COLUMNS].to_csv(path, index=False)


def filter_watchlist_for_asset(asset: str, watchlist: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = load_social_watchlist() if watchlist is None else watchlist.fillna("")
    if frame.empty or "asset_focus" not in frame.columns:
        return pd.DataFrame()
    aliases = ASSET_ALIASES.get(asset.upper(), {asset.upper()})
    mask = frame["asset_focus"].astype(str).apply(lambda value: bool(_split_pipe(value) & aliases))
    return frame[mask].copy()


def summarize_social_intel(asset: str, watchlist: pd.DataFrame | None = None) -> SocialIntelSummary:
    filtered = filter_watchlist_for_asset(asset, watchlist)
    if filtered.empty:
        return SocialIntelSummary(
            asset=asset.upper(),
            tracked_voices=0,
            average_reliability=0.0,
            high_reliability_voices=0,
            dominant_styles=[],
            caution_flags=["no_social_watchlist_matches"],
            note="No social intelligence watchlist entries are calibrated for this asset yet.",
        )

    reliability = pd.to_numeric(filtered.get("starting_reliability", pd.Series(dtype=float)), errors="coerce").fillna(50.0)
    styles = _top_values(filtered.get("style", pd.Series(dtype=str)).astype(str), limit=4)
    flags = _top_values(filtered.get("risk_flags", pd.Series(dtype=str)).astype(str), limit=5)
    return SocialIntelSummary(
        asset=asset.upper(),
        tracked_voices=int(len(filtered)),
        average_reliability=round(float(reliability.mean()), 1),
        high_reliability_voices=int((reliability >= 55).sum()),
        dominant_styles=styles,
        caution_flags=flags,
        note="Use these voices as context only; NERO must score their historical calls before trusting them.",
    )


def score_social_post(text: str) -> dict[str, object]:
    lowered = text.lower()
    assets = [asset for asset, aliases in ASSET_ALIASES.items() if any(alias.lower() in lowered for alias in aliases)]
    bullish_words = ["breakout", "bullish", "long", "accumulate", "support", "upside", "rally"]
    bearish_words = ["breakdown", "bearish", "short", "sell", "resistance", "downside", "crash"]
    bullish = sum(word in lowered for word in bullish_words)
    bearish = sum(word in lowered for word in bearish_words)
    if bullish > bearish:
        sentiment = "bullish"
    elif bearish > bullish:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    has_trade_plan = any(token in lowered for token in ["entry", "stop", "sl", "target", "tp", "invalid"])
    return {
        "assets": assets,
        "sentiment": sentiment,
        "has_trade_plan": has_trade_plan,
        "bullish_terms": bullish,
        "bearish_terms": bearish,
    }


def evaluate_social_calls(calls: pd.DataFrame, prices: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    if calls.empty or prices.empty:
        return calls.copy()
    evaluated = calls.copy()
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame = price_frame.sort_values("date")
    for idx, row in evaluated.iterrows():
        if str(row.get("status", "")).lower() == "evaluated":
            continue
        posted_at = pd.to_datetime(row.get("posted_at", ""), errors="coerce")
        if pd.isna(posted_at):
            evaluated.loc[idx, "status"] = "error"
            evaluated.loc[idx, "notes"] = "invalid posted_at"
            continue
        window = price_frame[(price_frame["date"] >= posted_at) & (price_frame["date"] <= posted_at + pd.Timedelta(hours=horizon_hours))]
        if window.empty:
            evaluated.loc[idx, "status"] = "pending"
            evaluated.loc[idx, "notes"] = "no price candles in evaluation window"
            continue
        outcome, r_multiple = _evaluate_one_call(row, window)
        evaluated.loc[idx, "status"] = "evaluated"
        evaluated.loc[idx, "outcome"] = outcome
        evaluated.loc[idx, "r_multiple"] = round(r_multiple, 3)
        evaluated.loc[idx, "evaluated_at"] = str(window.iloc[-1]["date"])
    return evaluated


def build_social_reliability_report(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame(columns=["handle", "source_name", "total_calls", "evaluated_calls", "win_rate", "average_r", "reliability_score"])
    frame = calls.copy()
    frame["r_multiple"] = pd.to_numeric(frame.get("r_multiple", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rows = []
    for (handle, source_name), group in frame.groupby(["handle", "source_name"], dropna=False):
        evaluated = group[group.get("status", "").astype(str).str.lower() == "evaluated"]
        wins = int((evaluated.get("outcome", "").astype(str).str.upper() == "WIN").sum()) if not evaluated.empty else 0
        win_rate = wins / len(evaluated) if len(evaluated) else 0.0
        average_r = float(evaluated["r_multiple"].mean()) if len(evaluated) else 0.0
        sample_factor = min(1.0, len(evaluated) / 20)
        reliability = 50 + (win_rate - 0.5) * 50 + max(-20, min(20, average_r * 12))
        reliability = max(0.0, min(100.0, 50 + (reliability - 50) * sample_factor))
        rows.append(
            {
                "handle": handle,
                "source_name": source_name,
                "total_calls": int(len(group)),
                "evaluated_calls": int(len(evaluated)),
                "win_rate": round(win_rate, 3),
                "average_r": round(average_r, 3),
                "reliability_score": round(reliability, 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["reliability_score", "evaluated_calls"], ascending=False)


def _evaluate_one_call(row: pd.Series, window: pd.DataFrame) -> tuple[str, float]:
    direction = str(row.get("direction", "")).upper()
    entry = _float_or_none(row.get("entry"))
    stop = _float_or_none(row.get("stop"))
    target = _float_or_none(row.get("target"))
    if entry is None:
        entry = float(window.iloc[0]["open"])
    if direction not in {"LONG", "SHORT"}:
        return "ERROR", 0.0
    if stop is None or stop == entry:
        final_close = float(window.iloc[-1]["close"])
        raw = (final_close - entry) / entry if direction == "LONG" else (entry - final_close) / entry
        return ("WIN" if raw > 0 else "LOSS" if raw < 0 else "FLAT"), raw * 10
    risk = abs(entry - stop)
    if target is not None:
        for _, candle in window.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            if direction == "LONG":
                if low <= stop:
                    return "LOSS", -1.0
                if high >= target:
                    return "WIN", abs(target - entry) / risk
            else:
                if high >= stop:
                    return "LOSS", -1.0
                if low <= target:
                    return "WIN", abs(entry - target) / risk
    final_close = float(window.iloc[-1]["close"])
    r_multiple = (final_close - entry) / risk if direction == "LONG" else (entry - final_close) / risk
    if r_multiple > 0.1:
        return "WIN", r_multiple
    if r_multiple < -0.1:
        return "LOSS", r_multiple
    return "FLAT", r_multiple


def _float_or_none(value: object) -> float | None:
    try:
        if str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_pipe(value: str) -> set[str]:
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _top_values(series: pd.Series, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for value in series:
        for item in _split_pipe(value):
            counts[item] = counts.get(item, 0) + 1
    return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]
