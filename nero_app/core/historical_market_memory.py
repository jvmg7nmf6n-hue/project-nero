from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_MEMORY_PATH = Path("nero_app/data/historical_market_events.csv")

BTC_120K_REQUIRED_TAGS = {
    "etf_inflows": 20,
    "institutional_inflows": 15,
    "dxy_weak": 15,
    "fed_dovish": 15,
    "risk_on": 10,
    "tech_strength": 10,
    "crypto_friendly_policy": 10,
    "structural_adoption": 5,
}

GOLD_RALLY_REQUIRED_TAGS = {
    "fed_dovish": 20,
    "dxy_weak": 15,
    "geopolitical_risk": 20,
    "safe_haven": 15,
    "inflation_risk": 10,
    "central_bank_demand": 15,
    "risk_off": 5,
}

RISK_TAGS = {
    "whale_selling": "large holder selling can cap upside",
    "liquidation_risk": "leverage/liquidation stress can force sharp drawdowns",
    "policy_delay": "delayed legislation can weaken the policy catalyst",
    "overheated_market": "overextended markets can reject bullish news",
    "dxy_strong": "strong dollar usually pressures BTC and Gold",
    "fed_hawkish": "hawkish Fed tone pressures liquidity-sensitive assets",
    "risk_off": "risk-off markets can pressure BTC even if Gold benefits",
}


@dataclass(frozen=True)
class RegimeSimilarity:
    asset: str
    reference_regime: str
    score: float
    supportive_factors: list[str]
    missing_factors: list[str]
    risk_factors: list[str]
    matched_events: int
    verdict: str


def load_historical_events(path: Path = DEFAULT_MEMORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def infer_environment_tags(
    asset: str,
    news_text: str = "",
    dxy_trend: str = "neutral",
    fed_tone: str = "neutral",
    risk_appetite: str = "neutral",
    etf_flow: str = "neutral",
    policy_tone: str = "neutral",
) -> set[str]:
    text = news_text.lower()
    tags: set[str] = set()

    if dxy_trend.lower() in {"weak", "falling", "down"} or "weak dollar" in text or "dollar weakness" in text:
        tags.add("dxy_weak")
    if dxy_trend.lower() in {"strong", "rising", "up"} or "strong dollar" in text:
        tags.add("dxy_strong")

    if fed_tone.lower() in {"dovish", "cuts", "cut"} or "rate cut" in text or "dovish" in text:
        tags.add("fed_dovish")
    if fed_tone.lower() in {"hawkish", "hikes", "higher"} or "hawkish" in text or "rate hike" in text:
        tags.add("fed_hawkish")

    if risk_appetite.lower() in {"risk_on", "risk-on", "strong"} or "risk appetite" in text or "tech stocks" in text:
        tags.add("risk_on")
    if risk_appetite.lower() in {"risk_off", "risk-off", "weak"} or "risk-off" in text or "geopolitical risks weigh" in text:
        tags.add("risk_off")

    if etf_flow.lower() in {"inflow", "inflows", "positive"} or "etf inflow" in text or "fund inflow" in text:
        tags.add("etf_inflows")
        tags.add("institutional_inflows")
    if etf_flow.lower() in {"outflow", "outflows", "negative"} or "etf outflow" in text:
        tags.add("etf_outflows")

    if policy_tone.lower() in {"friendly", "supportive", "positive"} or "crypto-friendly" in text or "stablecoin" in text:
        tags.add("crypto_friendly_policy")
    if policy_tone.lower() in {"hostile", "negative"} or "crackdown" in text:
        tags.add("policy_hostile")

    if "institutional" in text or "corporate" in text:
        tags.add("institutional_inflows")
    if "structural" in text or "adoption" in text:
        tags.add("structural_adoption")
    if "nasdaq" in text or "tech" in text:
        tags.add("tech_strength")
    if "geopolitical" in text or "war" in text or "sanctions" in text:
        tags.add("geopolitical_risk")
    if "safe haven" in text or "safe-haven" in text:
        tags.add("safe_haven")
    if "inflation" in text:
        tags.add("inflation_risk")
    if "central bank" in text:
        tags.add("central_bank_demand")
    if "liquidation" in text:
        tags.add("liquidation_risk")
    if "whale" in text:
        tags.add("whale_selling")

    if asset.upper() == "GOLD" and "geopolitical_risk" in tags:
        tags.add("safe_haven")
    return tags


def score_regime_similarity(asset: str, current_tags: set[str], events: pd.DataFrame | None = None) -> RegimeSimilarity:
    asset_key = asset.upper()
    if asset_key == "BTC":
        reference_regime = "BTC_120K_RALLY"
        required = BTC_120K_REQUIRED_TAGS
    elif asset_key == "GOLD":
        reference_regime = "GOLD_RALLY"
        required = GOLD_RALLY_REQUIRED_TAGS
    else:
        reference_regime = f"{asset_key}_UNKNOWN"
        required = {}

    score = sum(weight for tag, weight in required.items() if tag in current_tags)
    supportive = [tag for tag in required if tag in current_tags]
    missing = [tag for tag in required if tag not in current_tags]
    risks = [f"{tag}: {RISK_TAGS[tag]}" for tag in sorted(current_tags & set(RISK_TAGS))]

    matched_events = 0
    if events is not None and not events.empty and "reference_regime" in events:
        matched_events = int((events["reference_regime"].astype(str) == reference_regime).sum())

    return RegimeSimilarity(
        asset=asset_key,
        reference_regime=reference_regime,
        score=float(min(100, score)),
        supportive_factors=supportive,
        missing_factors=missing,
        risk_factors=risks,
        matched_events=matched_events,
        verdict=_verdict(score, risks),
    )


def format_regime_report(result: RegimeSimilarity) -> str:
    return "\n".join(
        [
            f"{result.asset} Historical Regime Similarity: {result.score:.0f}/100",
            f"Reference regime: {result.reference_regime}",
            f"Matched memory events: {result.matched_events}",
            f"Verdict: {result.verdict}",
            f"Supportive: {', '.join(result.supportive_factors) if result.supportive_factors else 'none'}",
            f"Missing: {', '.join(result.missing_factors) if result.missing_factors else 'none'}",
            f"Risks: {'; '.join(result.risk_factors) if result.risk_factors else 'none'}",
        ]
    )


def _verdict(score: float, risks: list[str]) -> str:
    if score >= 76 and not risks:
        return "high similarity to historical rally regime"
    if score >= 56:
        return "strong setup forming, but confirm missing/risk factors"
    if score >= 31:
        return "partial similarity only"
    return "weak similarity to historical rally regime"
