"""
Trade Opportunity Scanner
===========================

Purpose
-------
Explain WHY NERO is or is not paper-trading right now. This is a decision
+ explanation engine, not an order engine — it never touches an exchange
and never promises a profitable outcome.

It aggregates existing NERO signals (quant consensus, GARCH vol regime,
sentiment, ETF flow proxy, gold real-yield proxy, technical snapshot,
current paper-trade state) into a single transparent decision:

    TRADE_ALLOWED | WAIT_FOR_CONFIRMATION | BLOCKED_BY_RISK | DATA_INSUFFICIENT

Design goals
------------
- Pure function core (`scan_trade_opportunity`) — deterministic, no I/O.
- Every condition that passes/fails/near-misses is named and explainable,
  so the dashboard and the user can see the reasoning, not just a number.
- Duplicate-trade protection is a hard BLOCKED_BY_RISK gate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DECISIONS = (
    "TRADE_ALLOWED",
    "WAIT_FOR_CONFIRMATION",
    "BLOCKED_BY_RISK",
    "DATA_INSUFFICIENT",
)

DIRECTIONS = ("LONG_BIAS", "SHORT_BIAS", "NEUTRAL")


@dataclass
class TechnicalSnapshot:
    trend: Optional[str] = None          # "UP" / "DOWN" / "SIDEWAYS"
    rsi: Optional[float] = None
    zscore: Optional[float] = None
    volatility_regime: Optional[str] = None  # "LOW" / "NORMAL" / "HIGH" / "EXTREME"
    price_above_ma20: Optional[bool] = None
    price_above_ma200: Optional[bool] = None


@dataclass
class PaperTradeState:
    has_open_position: bool = False
    has_pending_order: bool = False
    asset: Optional[str] = None


@dataclass
class ScannerInputs:
    asset: str
    quant_consensus_score: Optional[float] = None   # 0-100 (existing NERO output)
    sentiment_score: Optional[float] = None          # -100..100 or 0-100, treated as 0-100 if >=0
    etf_flow_score: Optional[float] = None            # 0-100, BTC only, optional
    real_yield_score: Optional[float] = None           # 0-100, GOLD only, optional
    sharpe_90d: Optional[float] = None
    technical: TechnicalSnapshot = field(default_factory=TechnicalSnapshot)
    paper_trade_state: PaperTradeState = field(default_factory=PaperTradeState)


@dataclass
class ScanResult:
    opportunity_score: float  # 0-100
    decision: str
    asset: str
    direction_bias: str
    passed_conditions: List[str]
    failed_conditions: List[str]
    near_miss_conditions: List[str]
    blocker_reason: Optional[str]
    explanation: str

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Scoring weights (transparent & tunable)
# --------------------------------------------------------------------------- #

WEIGHTS = {
    "quant_consensus": 0.30,
    "sentiment": 0.15,
    "flow_macro": 0.20,      # ETF flow score OR real yield score, whichever applies
    "technical": 0.20,
    "risk": 0.15,             # volatility regime + sharpe
}

# thresholds
CONSENSUS_STRONG = 65
CONSENSUS_WEAK = 40
SENTIMENT_STRONG = 60
NEAR_MISS_MARGIN = 8  # points within a threshold counts as "near miss"


def _technical_subscore(tech: TechnicalSnapshot, passed: List[str], failed: List[str], near: List[str]) -> Optional[float]:
    points = []
    if tech.trend is not None:
        if tech.trend == "UP":
            points.append(70)
            passed.append("trend is UP")
        elif tech.trend == "DOWN":
            points.append(30)
            failed.append("trend is DOWN")
        else:
            points.append(50)
            near.append("trend is SIDEWAYS (no clear direction yet)")

    if tech.rsi is not None:
        if 45 <= tech.rsi <= 65:
            points.append(65)
            passed.append(f"RSI healthy ({tech.rsi:.1f})")
        elif tech.rsi > 70:
            points.append(35)
            failed.append(f"RSI overbought ({tech.rsi:.1f})")
        elif tech.rsi < 30:
            points.append(35)
            failed.append(f"RSI oversold ({tech.rsi:.1f})")
        else:
            points.append(50)
            near.append(f"RSI neutral-ish ({tech.rsi:.1f})")

    if tech.zscore is not None:
        if abs(tech.zscore) >= 2.0:
            points.append(70)
            passed.append(f"z-score extreme ({tech.zscore:.2f})")
        elif abs(tech.zscore) >= 1.0:
            points.append(55)
            near.append(f"z-score positive but not extreme ({tech.zscore:.2f})")
        else:
            points.append(45)

    if tech.price_above_ma200 is True:
        points.append(65)
        passed.append("price above MA200")
    elif tech.price_above_ma200 is False:
        points.append(35)
        failed.append("price below MA200")

    if tech.price_above_ma20 is True:
        points.append(58)
    elif tech.price_above_ma20 is False:
        points.append(42)

    if not points:
        return None
    return float(np.mean(points))


def _risk_subscore(tech: TechnicalSnapshot, sharpe_90d: Optional[float], passed: List[str], failed: List[str], near: List[str]) -> Optional[float]:
    points = []
    if tech.volatility_regime is not None:
        mapping = {"LOW": 60, "NORMAL": 70, "HIGH": 40, "EXTREME": 15}
        pts = mapping.get(tech.volatility_regime, 50)
        points.append(pts)
        if tech.volatility_regime in ("LOW", "NORMAL"):
            passed.append(f"volatility {tech.volatility_regime.lower()}")
        elif tech.volatility_regime == "HIGH":
            near.append("volatility elevated (HIGH regime)")
        else:
            failed.append("volatility EXTREME — risk gate")

    if sharpe_90d is not None:
        if sharpe_90d > 0.5:
            points.append(70)
            passed.append(f"90D Sharpe positive ({sharpe_90d:.2f})")
        elif sharpe_90d > 0:
            points.append(55)
            near.append(f"90D Sharpe marginally positive ({sharpe_90d:.2f})")
        else:
            points.append(25)
            failed.append(f"90D Sharpe negative ({sharpe_90d:.2f})")

    if not points:
        return None
    return float(np.mean(points))


def scan_trade_opportunity(inputs: ScannerInputs) -> ScanResult:
    passed: List[str] = []
    failed: List[str] = []
    near_miss: List[str] = []

    # --- hard risk gate: duplicate trade protection -----------------------
    if inputs.paper_trade_state.has_open_position or inputs.paper_trade_state.has_pending_order:
        reason = (
            f"An open or pending paper trade already exists for {inputs.asset}; "
            "avoiding duplicate entries."
        )
        return ScanResult(
            opportunity_score=0.0,
            decision="BLOCKED_BY_RISK",
            asset=inputs.asset,
            direction_bias="NEUTRAL",
            passed_conditions=[],
            failed_conditions=["duplicate open/pending paper trade exists"],
            near_miss_conditions=[],
            blocker_reason=reason,
            explanation=f"{inputs.asset}: trade blocked — {reason}",
        )

    # --- hard risk gate: extreme volatility --------------------------------
    if inputs.technical.volatility_regime == "EXTREME":
        reason = "Volatility regime is EXTREME — risk controls block new entries."
        return ScanResult(
            opportunity_score=15.0,
            decision="BLOCKED_BY_RISK",
            asset=inputs.asset,
            direction_bias="NEUTRAL",
            passed_conditions=passed,
            failed_conditions=["volatility EXTREME"],
            near_miss_conditions=[],
            blocker_reason=reason,
            explanation=f"{inputs.asset}: trade blocked — {reason}",
        )

    sub_scores: Dict[str, float] = {}

    if inputs.quant_consensus_score is not None:
        sub_scores["quant_consensus"] = float(np.clip(inputs.quant_consensus_score, 0, 100))
        if inputs.quant_consensus_score >= CONSENSUS_STRONG:
            passed.append(f"quant consensus strong ({inputs.quant_consensus_score:.0f})")
        elif inputs.quant_consensus_score >= CONSENSUS_STRONG - NEAR_MISS_MARGIN:
            near_miss.append(f"quant consensus close to strong ({inputs.quant_consensus_score:.0f})")
        elif inputs.quant_consensus_score < CONSENSUS_WEAK:
            failed.append(f"quant consensus weak ({inputs.quant_consensus_score:.0f})")
        else:
            near_miss.append(f"quant consensus mild only ({inputs.quant_consensus_score:.0f})")

    if inputs.sentiment_score is not None:
        sent = inputs.sentiment_score
        sent_0_100 = sent if sent >= 0 and sent <= 100 else float(np.clip((sent + 100) / 2, 0, 100))
        sub_scores["sentiment"] = sent_0_100
        if sent_0_100 >= SENTIMENT_STRONG:
            passed.append("sentiment supportive")
        elif sent_0_100 <= 35:
            failed.append("sentiment negative")
        else:
            near_miss.append("sentiment mixed/neutral")

    flow_macro_score = inputs.etf_flow_score if inputs.etf_flow_score is not None else inputs.real_yield_score
    flow_macro_label = "ETF proxy inflow pressure" if inputs.etf_flow_score is not None else "gold real-yield macro backdrop"
    if flow_macro_score is not None:
        sub_scores["flow_macro"] = float(np.clip(flow_macro_score, 0, 100))
        if flow_macro_score >= 60:
            passed.append(f"{flow_macro_label} moderate-to-strong ({flow_macro_score:.0f})")
        elif flow_macro_score <= 40:
            failed.append(f"{flow_macro_label} weak/unsupportive ({flow_macro_score:.0f})")
        else:
            near_miss.append(f"{flow_macro_label} neutral ({flow_macro_score:.0f})")

    tech_score = _technical_subscore(inputs.technical, passed, failed, near_miss)
    if tech_score is not None:
        sub_scores["technical"] = tech_score

    risk_score = _risk_subscore(inputs.technical, inputs.sharpe_90d, passed, failed, near_miss)
    if risk_score is not None:
        sub_scores["risk"] = risk_score

    if not sub_scores:
        return ScanResult(
            opportunity_score=0.0,
            decision="DATA_INSUFFICIENT",
            asset=inputs.asset,
            direction_bias="NEUTRAL",
            passed_conditions=[],
            failed_conditions=[],
            near_miss_conditions=[],
            blocker_reason="No usable signal inputs were provided.",
            explanation=f"{inputs.asset}: not enough data to evaluate an opportunity right now.",
        )

    total_weight = sum(WEIGHTS[k] for k in sub_scores)
    opportunity_score = sum(sub_scores[k] * WEIGHTS[k] for k in sub_scores) / total_weight
    opportunity_score = float(np.clip(opportunity_score, 0, 100))

    # --- direction bias -----------------------------------------------------
    bias_votes = 0
    if inputs.technical.trend == "UP":
        bias_votes += 1
    elif inputs.technical.trend == "DOWN":
        bias_votes -= 1
    if inputs.technical.price_above_ma200 is True:
        bias_votes += 1
    elif inputs.technical.price_above_ma200 is False:
        bias_votes -= 1
    if inputs.technical.zscore is not None:
        if inputs.technical.zscore > 0.5:
            bias_votes += 1
        elif inputs.technical.zscore < -0.5:
            bias_votes -= 1

    if bias_votes >= 2:
        direction_bias = "LONG_BIAS"
    elif bias_votes <= -2:
        direction_bias = "SHORT_BIAS"
    else:
        direction_bias = "NEUTRAL"

    # --- decision -------------------------------------------------------
    has_hard_failure = any(
        "negative" in f or "EXTREME" in f or "weak" in f for f in failed
    )

    if opportunity_score >= 70 and not failed:
        decision = "TRADE_ALLOWED"
    elif opportunity_score < 35 or (failed and opportunity_score < 50):
        decision = "BLOCKED_BY_RISK" if has_hard_failure and opportunity_score < 35 else "WAIT_FOR_CONFIRMATION"
    else:
        decision = "WAIT_FOR_CONFIRMATION"

    blocker_reason = None
    if decision == "BLOCKED_BY_RISK":
        blocker_reason = "; ".join(failed) if failed else "Composite risk score too low."

    explanation = _build_explanation(inputs.asset, opportunity_score, decision, passed, failed, near_miss)

    return ScanResult(
        opportunity_score=round(opportunity_score, 2),
        decision=decision,
        asset=inputs.asset,
        direction_bias=direction_bias,
        passed_conditions=passed,
        failed_conditions=failed,
        near_miss_conditions=near_miss,
        blocker_reason=blocker_reason,
        explanation=explanation,
    )


def _build_explanation(
    asset: str,
    score: float,
    decision: str,
    passed: List[str],
    failed: List[str],
    near_miss: List[str],
) -> str:
    lines = [f"{asset} Opportunity Score: {score:.0f}/100", f"Decision: {decision}"]
    if passed:
        lines.append("Passed: " + "; ".join(passed))
    if failed:
        lines.append("Failed: " + "; ".join(failed))
    if near_miss:
        lines.append("Near miss: " + "; ".join(near_miss))

    if decision == "TRADE_ALLOWED":
        conclusion = f"{asset} shows strong, aligned evidence — conditions support a paper trade."
    elif decision == "BLOCKED_BY_RISK":
        conclusion = f"{asset} is currently blocked by risk controls — not safe to enter right now."
    elif decision == "DATA_INSUFFICIENT":
        conclusion = f"{asset} cannot be evaluated yet — not enough signal data available."
    else:
        conclusion = f"{asset} is improving, but not enough confirmation yet for a paper trade."
    lines.append("Conclusion: " + conclusion)
    return "\n".join(lines)
