from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_WATCHLIST_PATH = Path("nero_app/data/social_intelligence_watchlist.csv")

ASSET_ALIASES = {
    "BTC": {"BTC", "Bitcoin", "Crypto"},
    "GOLD": {"Gold", "Commodities", "Forex", "Macro"},
    "SOL": {"SOL", "Altcoins", "Crypto"},
    "ETH": {"ETH", "Ethereum", "Crypto"},
}


@dataclass(frozen=True)
class SocialIntelSummary:
    asset: str
    tracked_voices: int
    average_reliability: float
    high_reliability_voices: int
    dominant_styles: list[str]
    caution_flags: list[str]
    note: str


def load_social_watchlist(path: Path = DEFAULT_WATCHLIST_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


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


def _split_pipe(value: str) -> set[str]:
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _top_values(series: pd.Series, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for value in series:
        for item in _split_pipe(value):
            counts[item] = counts.get(item, 0) + 1
    return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]
