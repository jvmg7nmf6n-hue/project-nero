"""Actionable trade-path guidance for Project Nero.

This module converts conservative NO_TRADE / WAIT readings into a plain
next-step map. It does not create financial advice and does not override the
paper-trading strategy; it explains what evidence must improve before NERO can
allow a paper setup.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TradePathInput:
    asset: str
    readiness_label: str
    readiness_score: float
    opportunity_decision: str
    opportunity_score: float
    direction_bias: str
    quant_score: float | None = None
    external_score: float | None = None
    external_label: str | None = None
    sentiment_score: float | None = None
    volatility_regime: str | None = None
    blockers: Iterable[str] = ()
    failed_conditions: Iterable[str] = ()
    near_miss_conditions: Iterable[str] = ()
    has_active_paper_trade: bool = False


@dataclass(frozen=True)
class TradePathReport:
    path_label: str
    action: str
    missing_confirmations: list[str] = field(default_factory=list)
    blocker_explanations: list[str] = field(default_factory=list)
    watch_triggers: list[str] = field(default_factory=list)
    next_check: str = "Wait for the next closed candle and refresh NERO."

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_trade_path_report(inputs: TradePathInput) -> TradePathReport:
    """Build plain-language guidance for the next possible paper trade."""
    blockers = [str(item) for item in inputs.blockers if str(item).strip()]
    failed = [str(item) for item in inputs.failed_conditions if str(item).strip()]
    near = [str(item) for item in inputs.near_miss_conditions if str(item).strip()]

    if inputs.has_active_paper_trade or any("active paper trade" in b.lower() or "duplicate" in b.lower() for b in blockers + failed):
        return TradePathReport(
            path_label="WAIT_EXISTING_TRADE",
            action="Do not hunt a new setup. NERO already has an active/pending paper-trade path, so the next action is to let that trade resolve first.",
            blocker_explanations=blockers or failed or ["Existing paper trade protection is active."],
            watch_triggers=["Current paper trade closes by target, stop, or expiry", "Then re-check the same asset on a fresh closed candle"],
            next_check="After the existing paper trade is closed or cancelled.",
        )

    missing: list[str] = []
    watch: list[str] = []
    explanations: list[str] = []

    if inputs.quant_score is None:
        missing.append("Quant consensus data must load cleanly.")
    elif inputs.quant_score < 45:
        missing.append(f"Quant consensus is weak ({inputs.quant_score:.0f}/100); needs at least 55-65 before NERO should trust the setup.")
        watch.append("Quant score rising above 55, ideally 65+")
    elif inputs.quant_score < 65:
        missing.append(f"Quant consensus is only mixed ({inputs.quant_score:.0f}/100); needs stronger confirmation.")
        watch.append("Quant score crossing 65+")

    if inputs.external_score is not None:
        label = inputs.external_label or "external pressure"
        if inputs.external_score <= 40:
            missing.append(f"{label} is unsupportive ({inputs.external_score:.0f}/100).")
            if inputs.asset.upper() == "BTC":
                watch.append("ETF flow flips from outflow/weak to neutral or inflow pressure")
            elif inputs.asset.upper() == "GOLD":
                watch.append("Real-yield backdrop improves from pressure to neutral/supportive")
        elif inputs.external_score < 60:
            missing.append(f"{label} is still neutral ({inputs.external_score:.0f}/100).")

    if inputs.sentiment_score is None:
        missing.append("News/sentiment confirmation is missing.")
    elif inputs.sentiment_score < 45:
        missing.append(f"Sentiment is not supportive enough ({inputs.sentiment_score:.0f}/100).")
        watch.append("Sentiment improves above 55-60")

    if inputs.volatility_regime and inputs.volatility_regime.upper() in {"VOL_STRESS", "EXTREME", "HIGH"}:
        missing.append(f"Volatility regime is risky ({inputs.volatility_regime}).")
        watch.append("Volatility returns to NORMAL/LOW")

    for item in failed[:5]:
        explanations.append(item)
    for item in near[:5]:
        watch.append(item)

    if inputs.readiness_label == "TRADE_READY" and inputs.opportunity_decision == "TRADE_ALLOWED":
        return TradePathReport(
            path_label="PAPER_TRADE_READY",
            action="NERO conditions are aligned for strategy-defined paper execution. Still use paper trading only and let the strategy rules define entry, stop, and target.",
            missing_confirmations=[],
            blocker_explanations=[],
            watch_triggers=["Record the setup", "Let accountability judge target/stop/expiry"],
            next_check="On the current closed-candle setup.",
        )

    label = "NO_TRADE_PATH" if inputs.readiness_score < 35 else "WAIT_FOR_SETUP"
    if not missing:
        missing.append("NERO needs cleaner cross-module agreement before it can justify a paper trade.")
    if not watch:
        watch.append("Opportunity score above 70 with no failed conditions")
        watch.append("Readiness score above 75")

    return TradePathReport(
        path_label=label,
        action="No new paper trade yet. The useful action is to wait for the listed confirmations, not force a weak setup.",
        missing_confirmations=missing,
        blocker_explanations=blockers + explanations,
        watch_triggers=watch,
        next_check="After the next 1H closed candle or when NERO sends an alert.",
    )
