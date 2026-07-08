from __future__ import annotations

from collections import Counter

import pandas as pd

from nero_app.core.knowledge_store import LocalKnowledgeStore
from nero_app.core.schema import AssessmentOutput, BrainOutput, HistoricalMatch, VerdictOutput
from nero_app.core.technical_analysis import analyze_technical


class BrainAgent:
    def __init__(self, store: LocalKnowledgeStore) -> None:
        self.store = store

    def run(self, headline: str, asset: str) -> BrainOutput:
        matches = self.store.search(headline, asset=asset, limit=5)
        tags = Counter(tag for match in matches for tag in match.tags)
        return BrainOutput(
            matches=matches,
            thematic_score=max(-1.0, min(1.0, _weighted_average(matches))),
            dominant_tags=[tag for tag, _ in tags.most_common(5)],
        )


class MarketAssessmentAgent:
    def run(self, prices: pd.DataFrame, lookback_days: int) -> AssessmentOutput:
        technical = analyze_technical(prices, lookback_days=lookback_days)
        return AssessmentOutput(**technical.__dict__)


class VerdictAgent:
    def run(self, brain: BrainOutput, assessment: AssessmentOutput) -> VerdictOutput:
        fvg_bonus = {"bullish": 0.12, "bearish": -0.12, "none": 0.0}[assessment.fair_value_gap]
        sweep_bonus = {"downside": 0.06, "upside": -0.06, "none": 0.0}[assessment.liquidity_sweep]
        regime_adjustment = {"Bull": 0.08, "Bear": -0.08, "Range": 0.0}[assessment.market_regime]
        composite = (
            0.42 * brain.thematic_score
            + 0.24 * assessment.momentum_score
            + 0.28 * assessment.technical_bias_score
            + fvg_bonus
            + sweep_bonus
            + regime_adjustment
        )

        if composite > 0.16:
            direction = "bullish"
        elif composite < -0.16:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(0.92, max(0.35, abs(composite) * 0.72 + _match_quality(brain.matches) * 0.35))
        volatility_penalty = {"High-Vol": 0.12, "Normal-Vol": 0.04, "Low-Vol": 0.0}[assessment.volatility_regime]
        risk_score = min(0.95, max(0.1, 1 - confidence + abs(assessment.rsi - 50) / 150 + volatility_penalty))
        drivers = [
            f"Thematic score: {brain.thematic_score:.2f}",
            f"Momentum score: {assessment.momentum_score:.2f}",
            f"Technical confluence: {assessment.confluence_score:.0f}/100 ({assessment.confluence_label})",
            f"Market regime: {assessment.market_regime} / {assessment.volatility_regime}",
            f"FVG: {assessment.fair_value_gap}",
            f"Liquidity sweep: {assessment.liquidity_sweep}",
        ]
        summary = (
            f"Nero produces a {direction} research verdict with {confidence:.0%} confidence. "
            f"The decision combines historical macro similarity, technical confluence, market regime, "
            f"momentum, gap structure, and liquidity sweep context. Dominant macro tags: {', '.join(brain.dominant_tags) or 'none'}."
        )
        return VerdictOutput(
            direction=direction,
            confidence=round(confidence, 3),
            risk_score=round(risk_score, 3),
            summary=summary,
            drivers=drivers,
        )


def _weighted_average(matches: list[HistoricalMatch]) -> float:
    total_weight = sum(match.similarity for match in matches)
    if total_weight == 0:
        return 0.0
    return sum(match.forward_bias * match.similarity for match in matches) / total_weight


def _match_quality(matches: list[HistoricalMatch]) -> float:
    if not matches:
        return 0.0
    return sum(match.similarity for match in matches[:3]) / min(3, len(matches))

