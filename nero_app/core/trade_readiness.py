"""Final trade-readiness layer for Project Nero.

This module converts NERO's existing analytical evidence into one operational
readiness label. It is deliberately conservative: it does not place trades and
it does not override the paper-trading strategy rules.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable


READINESS_LABELS = (
    "TRADE_READY",
    "WAIT_FOR_CONFIRMATION",
    "NO_TRADE_RISK",
    "NO_TRADE_DATA_WEAK",
)


@dataclass(frozen=True)
class ReadinessInputs:
    asset: str
    opportunity_decision: str
    opportunity_score: float
    quant_score: float | None = None
    volatility_regime: str | None = None
    sentiment_score: float | None = None
    has_active_paper_trade: bool = False
    missing_inputs: Iterable[str] = ()


@dataclass(frozen=True)
class ReadinessReport:
    asset: str
    readiness_score: float
    label: str
    action: str
    reasons: list[str]
    blockers: list[str]
    missing_inputs: list[str]

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_trade_readiness_report(inputs: ReadinessInputs) -> ReadinessReport:
    """Build a conservative final readiness report from existing NERO signals."""
    reasons: list[str] = []
    blockers: list[str] = []
    missing = [str(item) for item in inputs.missing_inputs if str(item).strip()]

    score = float(max(0.0, min(100.0, inputs.opportunity_score)))

    if inputs.quant_score is None:
        missing.append("quant consensus")
    else:
        if inputs.quant_score >= 65:
            reasons.append(f"quant consensus supportive ({inputs.quant_score:.0f}/100)")
            score += 5
        elif inputs.quant_score < 45:
            blockers.append(f"quant consensus weak ({inputs.quant_score:.0f}/100)")
            score -= 15
        else:
            reasons.append(f"quant consensus mixed ({inputs.quant_score:.0f}/100)")

    if inputs.volatility_regime:
        regime = inputs.volatility_regime.upper()
        if regime in {"VOL_STRESS", "EXTREME"}:
            blockers.append(f"volatility risk is high ({inputs.volatility_regime})")
            score -= 25
        elif regime in {"VOL_NORMAL", "NORMAL", "VOL_COMPRESSED", "LOW"}:
            reasons.append(f"volatility acceptable ({inputs.volatility_regime})")
            score += 5
        else:
            reasons.append(f"volatility requires caution ({inputs.volatility_regime})")
            score -= 5
    else:
        missing.append("volatility regime")

    if inputs.sentiment_score is None:
        missing.append("news sentiment")
    else:
        sentiment = _normalise_sentiment(inputs.sentiment_score)
        if sentiment >= 60:
            reasons.append(f"sentiment supportive ({sentiment:.0f}/100)")
            score += 3
        elif sentiment <= 35:
            blockers.append(f"sentiment negative ({sentiment:.0f}/100)")
            score -= 8
        else:
            reasons.append(f"sentiment mixed ({sentiment:.0f}/100)")

    decision = inputs.opportunity_decision.upper()
    if inputs.has_active_paper_trade:
        blockers.append("active paper trade already exists")
        score = min(score, 20)
    elif decision == "TRADE_ALLOWED":
        reasons.append("opportunity scanner allows a paper setup")
        score += 8
    elif decision == "BLOCKED_BY_RISK":
        blockers.append("opportunity scanner blocked the setup")
        score = min(score, 30)
    elif decision == "DATA_INSUFFICIENT":
        missing.append("trade opportunity scanner inputs")
        score = min(score, 35)
    else:
        reasons.append("opportunity scanner is waiting for confirmation")

    score = float(max(0.0, min(100.0, score)))
    missing = sorted(set(missing))

    if blockers:
        label = "NO_TRADE_RISK"
        action = "Do not open a new paper trade; risk gates are blocking the setup."
    elif missing and score < 55:
        label = "NO_TRADE_DATA_WEAK"
        action = "Wait; NERO does not have enough clean evidence yet."
    elif score >= 75 and decision == "TRADE_ALLOWED":
        label = "TRADE_READY"
        action = "Paper-trade conditions are aligned; allow only strategy-defined paper execution."
    else:
        label = "WAIT_FOR_CONFIRMATION"
        action = "Wait for a stronger trigger or cleaner cross-module confirmation."

    return ReadinessReport(
        asset=inputs.asset,
        readiness_score=round(score, 2),
        label=label,
        action=action,
        reasons=reasons or ["no strong supportive reason yet"],
        blockers=blockers,
        missing_inputs=missing,
    )


def _normalise_sentiment(score: float) -> float:
    value = float(score)
    if -10 <= value <= 10:
        return (value + 10.0) * 5.0
    if -100 <= value < 0:
        return (value + 100.0) / 2.0
    return max(0.0, min(100.0, value))
