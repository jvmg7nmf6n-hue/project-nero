"""Local command chat for Project Nero.

This module turns Nero's current dashboard readings into plain-language answers.
It is intentionally rule-based and deterministic: no external AI call is needed
for the first version, and it never places or recommends real orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeroChatContext:
    asset: str
    data_status: str
    verdict_direction: str
    verdict_confidence: float
    risk_score: float
    trade_action: str
    trade_bias: str
    trade_confidence: float
    entry_trigger: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    invalidation: str
    consensus_class: str
    consensus_quality: float
    consensus_direction: str
    sentiment: str
    sentiment_score: int | float
    confluence_score: float
    market_regime: str
    volatility_regime: str
    blockers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    test_lab: list[dict[str, Any]] = field(default_factory=list)


SUGGESTED_QUESTIONS = [
    "Kya abhi trade lena chahiye?",
    "BTC ka next trigger kya hai?",
    "Risk kahan se aa raha hai?",
    "Kaunsa algo best perform kar raha hai?",
    "NERO no trade kyun keh raha hai?",
]


def answer_nero_chat(question: str, context: NeroChatContext) -> str:
    query = (question or "").strip().lower()
    if not query:
        return _overview(context)
    if _contains(query, ["trade", "lena", "buy", "sell", "long", "short", "entry", "abhi"]):
        return _trade_answer(context)
    if _contains(query, ["trigger", "level", "entry", "kab"]):
        return _trigger_answer(context)
    if _contains(query, ["risk", "block", "blocker", "danger", "loss"]):
        return _risk_answer(context)
    if _contains(query, ["algo", "test", "lab", "strategy", "best", "perform"]):
        return _algo_answer(context)
    if _contains(query, ["news", "sentiment", "headline"]):
        return _sentiment_answer(context)
    if _contains(query, ["summary", "status", "update", "haal"]):
        return _overview(context)
    return _fallback_answer(context)


def _trade_answer(context: NeroChatContext) -> str:
    lines = [_header(context)]
    if context.trade_action == "NO_TRADE" or context.consensus_class == "NO_TRADE":
        lines.append("Short answer: abhi fresh paper trade open karne ka setup clean nahi hai.")
        if context.blockers:
            lines.append("Main blockers: " + "; ".join(context.blockers[:3]) + ".")
        else:
            lines.append("Reason: signals mixed hain, isliye NERO capital protection prefer kar raha hai.")
    elif context.trade_action == "WAIT_LONG_TRIGGER":
        lines.append("Short answer: LONG idea sirf trigger confirmation ke baad valid hai.")
        lines.append(f"Trigger: {context.entry_trigger}")
        lines.append(f"SL: {context.stop_loss:,.2f} | TP1: {context.take_profit_1:,.2f} | TP2: {context.take_profit_2:,.2f}")
    elif context.trade_action == "WAIT_SHORT_TRIGGER":
        lines.append("Short answer: SHORT idea sirf breakdown confirmation ke baad valid hai.")
        lines.append(f"Trigger: {context.entry_trigger}")
        lines.append(f"SL: {context.stop_loss:,.2f} | TP1: {context.take_profit_1:,.2f} | TP2: {context.take_profit_2:,.2f}")
    else:
        lines.append(f"NERO action: {context.trade_action}. Confirm before using it even for paper-trading.")
    lines.append(_quality_line(context))
    lines.append("Note: yeh decision support hai, financial advice ya profit guarantee nahi.")
    return "\n".join(lines)


def _trigger_answer(context: NeroChatContext) -> str:
    if context.trade_action == "NO_TRADE":
        return "\n".join([
            _header(context),
            "Abhi koi clean trigger active nahi.",
            f"Wait condition: {context.entry_trigger}",
            f"Invalidation/risk note: {context.invalidation}",
        ])
    return "\n".join([
        _header(context),
        f"Active trigger: {context.entry_trigger}",
        f"Direction: {context.trade_bias}",
        f"SL: {context.stop_loss:,.2f}",
        f"TP1/TP2: {context.take_profit_1:,.2f} / {context.take_profit_2:,.2f}",
    ])


def _risk_answer(context: NeroChatContext) -> str:
    lines = [_header(context), f"Risk score: {context.risk_score:.0%}", f"Volatility regime: {context.volatility_regime}"]
    if context.blockers:
        lines.append("Blockers: " + "; ".join(context.blockers[:5]) + ".")
    if context.risk_score >= 0.7:
        lines.append("Interpretation: risk elevated hai; NERO fresh trade ko strict filter karega.")
    elif context.risk_score >= 0.45:
        lines.append("Interpretation: medium risk hai; trigger aur position quality important hai.")
    else:
        lines.append("Interpretation: risk relatively controlled hai, lekin confluence confirm honi chahiye.")
    return "\n".join(lines)


def _algo_answer(context: NeroChatContext) -> str:
    if not context.test_lab:
        return "TEST Lab mein abhi enough algo records nahi. GitHub runs data collect karenge; 30-50 trades per algo ke baad rating meaningful hogi."
    ranked = sorted(context.test_lab, key=lambda row: (float(row.get("rating_score", 0) or 0), int(float(row.get("total_trades", 0) or 0))), reverse=True)
    best = ranked[0]
    lines = [
        "TEST Lab status:",
        f"Best current candidate: {best.get('candidate_id', '-')} | rating={best.get('rating', '-')} | score={float(best.get('rating_score', 0) or 0):.0f}/100.",
        f"Trades so far: {int(float(best.get('total_trades', 0) or 0))} | expectancy={float(best.get('expectancy_r', 0) or 0):.2f}R | profit factor={float(best.get('profit_factor', 0) or 0):.2f}.",
    ]
    if int(float(best.get("total_trades", 0) or 0)) < 30:
        lines.append("Sample warning: abhi early hai; isko final proof na samjho.")
    lines.append("Use: yeh batata hai kaunsa algo evidence collect kar raha hai, blind guessing kam hoti hai.")
    return "\n".join(lines)


def _sentiment_answer(context: NeroChatContext) -> str:
    return "\n".join([
        _header(context),
        f"News/AI sentiment: {context.sentiment} ({context.sentiment_score}/10).",
        "Sentiment akela trade signal nahi; NERO isko quant, readiness, risk aur trigger ke saath combine karta hai.",
    ])


def _overview(context: NeroChatContext) -> str:
    lines = [
        _header(context),
        f"Verdict: {context.verdict_direction.upper()} | confidence {context.verdict_confidence:.0%} | risk {context.risk_score:.0%}.",
        f"Trade Desk: {context.trade_action.replace('_', ' ')} | bias {context.trade_bias} | confidence {context.trade_confidence:.0%}.",
        f"Consensus: {context.consensus_class.replace('_', ' ')} | quality {context.consensus_quality:.0f}/100 | direction {context.consensus_direction}.",
        f"Market: {context.market_regime} / {context.volatility_regime} | technical confluence {context.confluence_score:.0f}/100.",
    ]
    if context.blockers:
        lines.append("Blockers: " + "; ".join(context.blockers[:3]) + ".")
    lines.append(_quality_line(context))
    return "\n".join(lines)


def _fallback_answer(context: NeroChatContext) -> str:
    return "\n".join([
        "Main NERO ke current dashboard ko explain kar sakta hoon.",
        _overview(context),
        "Try asking: 'trade lena chahiye?', 'risk kya hai?', 'best algo kaunsa hai?', ya 'next trigger kya hai?'.",
    ])


def _header(context: NeroChatContext) -> str:
    return f"NERO Chat | {context.asset} | data={context.data_status}"


def _quality_line(context: NeroChatContext) -> str:
    if context.consensus_quality >= 75 and context.trade_action != "NO_TRADE":
        return "Quality read: strong setup, but still trigger-based and paper-test first."
    if context.consensus_quality >= 55:
        return "Quality read: mixed-to-acceptable; wait for confirmation."
    return "Quality read: weak/mixed; no forced trade."


def _contains(query: str, words: list[str]) -> bool:
    return any(word in query for word in words)
