from __future__ import annotations

from nero_app.core.schema import VerdictOutput
from nero_app.core.white_house_impact import WhiteHouseImpactResult, score_white_house_impact


SUPPORTED_WHITE_HOUSE_ASSETS = {"BTC", "GOLD"}


def apply_white_house_modifier(asset: str, headline: str, verdict: VerdictOutput) -> tuple[VerdictOutput, WhiteHouseImpactResult | None]:
    asset_key = asset.upper()
    if asset_key not in SUPPORTED_WHITE_HOUSE_ASSETS:
        return verdict, None

    impact = score_white_house_impact(headline)
    if impact.matched_events == 0 or impact.confidence <= 0:
        return verdict, impact

    asset_score = impact.btc_average_impact if asset_key == "BTC" else impact.gold_average_impact
    adjustment = _impact_adjustment(asset_key, asset_score, impact)
    if adjustment == 0:
        return _append_white_house_context(verdict, impact, asset_key, asset_score, adjustment), impact

    base_score = {"bullish": 0.45, "neutral": 0.0, "bearish": -0.45}[verdict.direction]
    adjusted_score = max(-1.0, min(1.0, base_score + adjustment))
    direction = _direction_from_score(adjusted_score)

    alignment_bonus = 0.04 if direction == verdict.direction else 0.0
    confidence = min(0.95, max(0.25, verdict.confidence + abs(adjustment) * 0.22 + alignment_bonus))
    conflict_penalty = 0.08 if verdict.direction != "neutral" and direction != verdict.direction else 0.0
    risk_score = min(0.95, max(0.08, verdict.risk_score + conflict_penalty - abs(adjustment) * 0.06))

    modified = verdict.model_copy(
        update={
            "direction": direction,
            "confidence": round(confidence, 3),
            "risk_score": round(risk_score, 3),
        }
    )
    return _append_white_house_context(modified, impact, asset_key, asset_score, adjustment), impact


def _impact_adjustment(asset: str, score: float, impact: WhiteHouseImpactResult) -> float:
    confidence = max(0.0, min(1.0, impact.confidence))
    if score >= 65:
        return 0.34 * confidence
    if score >= 35:
        return 0.14 * confidence
    if asset == "BTC" and "policy_hostile" in impact.query_tags:
        return -0.22 * confidence
    return 0.0


def _direction_from_score(score: float) -> str:
    if score > 0.25:
        return "bullish"
    if score < -0.25:
        return "bearish"
    return "neutral"


def _append_white_house_context(
    verdict: VerdictOutput,
    impact: WhiteHouseImpactResult,
    asset: str,
    score: float,
    adjustment: float,
) -> VerdictOutput:
    direction = impact.btc_direction if asset == "BTC" else impact.gold_direction
    driver = (
        f"White House impact: {asset} {direction} ({score:.0f}/100, "
        f"confidence {impact.confidence:.0%}, adjustment {adjustment:+.2f})"
    )
    summary = (
        f"{verdict.summary} White House impact memory adds {asset} context: "
        f"{direction} at {score:.0f}/100 based on {impact.matched_events} similar event(s)."
    )
    return verdict.model_copy(update={"summary": summary, "drivers": [*verdict.drivers, driver]})
