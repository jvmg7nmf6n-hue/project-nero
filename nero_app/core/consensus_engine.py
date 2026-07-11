from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nero_app.core.historical_market_memory import RegimeSimilarity
from nero_app.core.schema import AssessmentOutput, VerdictOutput
from nero_app.core.trade_desk import IntradayTradePlan
from nero_app.core.white_house_impact import WhiteHouseImpactResult

DecisionClass = Literal["NO_TRADE", "SCALP_ONLY", "NORMAL_TRADE", "HIGH_CONVICTION"]


@dataclass(frozen=True)
class ConsensusDecision:
    decision_class: DecisionClass
    trade_quality: float
    direction: str
    human_note: str
    reasons: list[str]
    blockers: list[str]


def build_consensus_decision(
    verdict: VerdictOutput,
    assessment: AssessmentOutput,
    trade_plan: IntradayTradePlan,
    news_sentiment: str,
    news_score: int,
    market_memory: RegimeSimilarity | None = None,
    white_house_impact: WhiteHouseImpactResult | None = None,
) -> ConsensusDecision:
    direction = _direction_from_plan_or_verdict(trade_plan, verdict)
    score = 0.0
    reasons: list[str] = []
    blockers: list[str] = []

    technical_points = _scale(assessment.confluence_score, 0, 100) * 28
    score += technical_points
    reasons.append(f"Technical confluence contributes {technical_points:.0f}/28.")

    if trade_plan.action in {"WAIT_LONG_TRIGGER", "WAIT_SHORT_TRIGGER"}:
        trigger_points = 22 * max(0.35, min(1.0, trade_plan.confidence))
        score += trigger_points
        reasons.append(f"Trade Desk has an actionable trigger: {trade_plan.action.replace('_', ' ')}.")
    else:
        trigger_points = 0.0
        blockers.append("Trade Desk is standing aside; no clean trigger yet.")

    verdict_alignment = _verdict_alignment(direction, verdict.direction)
    verdict_points = verdict_alignment * 16 * max(0.35, verdict.confidence)
    score += verdict_points
    reasons.append(f"NERO verdict alignment contributes {verdict_points:.0f}/16.")

    news_alignment = _news_alignment(direction, news_sentiment, news_score)
    news_points = news_alignment * 10
    score += news_points
    reasons.append(f"News sentiment contributes {news_points:.0f}/10.")

    memory_points = _market_memory_points(market_memory, direction)
    score += memory_points
    if market_memory is not None:
        reasons.append(f"Market Memory contributes {memory_points:.0f}/12 for {market_memory.reference_regime}.")

    white_house_points = _white_house_points(white_house_impact, direction)
    score += white_house_points
    if white_house_impact is not None:
        reasons.append(f"White House impact contributes {white_house_points:.0f}/6.")

    risk_penalty = max(0.0, min(1.0, verdict.risk_score)) * 22
    score -= risk_penalty
    if verdict.risk_score >= 0.68:
        blockers.append(f"Risk score is high at {verdict.risk_score:.0%}.")
    reasons.append(f"Risk penalty subtracts {risk_penalty:.0f} points.")

    if assessment.volatility_regime == "High-Vol":
        score -= 8
        blockers.append("High volatility regime reduces trade quality.")
    if assessment.market_regime == "Range" and trade_plan.action != "NO_TRADE":
        score -= 4
        reasons.append("Range regime caps conviction; breakout must confirm.")

    trade_quality = round(max(0.0, min(100.0, score)), 1)
    decision_class = _decision_class(trade_quality, trade_plan, verdict, blockers)
    return ConsensusDecision(
        decision_class=decision_class,
        trade_quality=trade_quality,
        direction=direction,
        human_note=_human_note(decision_class, direction, blockers),
        reasons=reasons,
        blockers=blockers,
    )


def _direction_from_plan_or_verdict(trade_plan: IntradayTradePlan, verdict: VerdictOutput) -> str:
    if trade_plan.action == "WAIT_LONG_TRIGGER":
        return "LONG"
    if trade_plan.action == "WAIT_SHORT_TRIGGER":
        return "SHORT"
    if verdict.direction == "bullish":
        return "LONG"
    if verdict.direction == "bearish":
        return "SHORT"
    return "NEUTRAL"


def _verdict_alignment(direction: str, verdict_direction: str) -> float:
    if direction == "NEUTRAL" or verdict_direction == "neutral":
        return 0.35
    if direction == "LONG" and verdict_direction == "bullish":
        return 1.0
    if direction == "SHORT" and verdict_direction == "bearish":
        return 1.0
    return -0.6


def _news_alignment(direction: str, sentiment: str, score: int) -> float:
    magnitude = min(1.0, abs(score) / 10)
    sentiment_key = sentiment.lower()
    if direction == "NEUTRAL" or sentiment_key == "neutral":
        return 0.25 * magnitude
    if direction == "LONG" and sentiment_key == "bullish":
        return magnitude
    if direction == "SHORT" and sentiment_key == "bearish":
        return magnitude
    return -0.5 * magnitude


def _market_memory_points(memory: RegimeSimilarity | None, direction: str) -> float:
    if memory is None or direction == "NEUTRAL":
        return 0.0
    if direction == "LONG":
        return _scale(memory.score, 0, 100) * 12
    if memory.score >= 65:
        return -4.0
    return 2.0


def _white_house_points(impact: WhiteHouseImpactResult | None, direction: str) -> float:
    if impact is None or direction == "NEUTRAL" or impact.confidence <= 0:
        return 0.0
    score = max(impact.btc_average_impact, impact.gold_average_impact)
    points = _scale(score, 0, 100) * 6 * impact.confidence
    return points if direction == "LONG" else min(2.0, points / 3)


def _decision_class(
    trade_quality: float,
    trade_plan: IntradayTradePlan,
    verdict: VerdictOutput,
    blockers: list[str],
) -> DecisionClass:
    if trade_plan.action == "NO_TRADE" or trade_quality < 35:
        return "NO_TRADE"
    if trade_quality < 55 or verdict.risk_score >= 0.68:
        return "SCALP_ONLY"
    if trade_quality < 76 or blockers:
        return "NORMAL_TRADE"
    return "HIGH_CONVICTION"


def _human_note(decision_class: DecisionClass, direction: str, blockers: list[str]) -> str:
    if decision_class == "NO_TRADE":
        return "Wait. Do not force a trade; capital protection is the decision."
    if decision_class == "SCALP_ONLY":
        return f"{direction} idea is tactical only. Keep size small and demand trigger confirmation."
    if decision_class == "NORMAL_TRADE":
        return f"{direction} setup is tradable after trigger, but manage risk tightly."
    return f"{direction} setup has broad confirmation. Still use stop-loss and avoid overconfidence."


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))
