from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nero_app.core.technical_analysis import analyze_technical


@dataclass(frozen=True)
class IntradayTradePlan:
    action: str
    bias: str
    confidence: float
    entry_trigger: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_1: float
    risk_reward_2: float
    invalidation: str
    status: str
    reasons: list[str]


def build_intraday_trade_plan(
    prices: pd.DataFrame,
    asset: str,
    macro_direction: str,
    news_sentiment: str,
    news_score: int,
    risk_score: float,
) -> IntradayTradePlan:
    frame = prices.sort_values("date").copy().reset_index(drop=True)
    if len(frame) < 30:
        return _empty_plan("Need at least 30 intraday candles before Nero can build a plan.")

    technical = analyze_technical(frame, lookback_days=min(60, len(frame)))
    latest = frame.iloc[-1]
    last_close = float(latest["close"])
    atr = _atr(frame)
    if atr <= 0:
        atr = max(last_close * 0.003, 1.0)

    prior = frame.tail(24)
    resistance = float(prior["high"].max())
    support = float(prior["low"].min())
    direction_score = _direction_score(
        macro_direction=macro_direction,
        news_sentiment=news_sentiment,
        news_score=news_score,
        technical_bias=technical.technical_bias_score,
        confluence_score=technical.confluence_score,
    )
    high_risk = risk_score >= 0.72 or technical.volatility_regime == "High-Vol"

    reasons = [
        f"Macro verdict: {macro_direction.upper()}",
        f"News sentiment: {news_sentiment} ({news_score}/10)",
        f"Intraday confluence: {technical.confluence_score:.0f}/100",
        f"Regime: {technical.market_regime} / {technical.volatility_regime}",
        f"RSI {technical.rsi:.1f}, MACD {technical.macd_signal}, BOS {technical.bos_signal}",
    ]

    if direction_score >= 0.34 and not high_risk:
        entry = max(last_close, resistance + atr * 0.05)
        stop = min(support, entry - atr * 1.25)
        tp1 = entry + atr * 1.5
        tp2 = entry + atr * 2.5
        return _plan(
            action="WAIT_LONG_TRIGGER",
            bias="Bullish",
            confidence=direction_score,
            entry_trigger=f"Long only after price holds above {entry:,.2f}",
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            invalidation=f"Cancel long idea if candle closes below {support:,.2f}",
            reasons=reasons,
        )

    if direction_score <= -0.34 and not high_risk:
        entry = min(last_close, support - atr * 0.05)
        stop = max(resistance, entry + atr * 1.25)
        tp1 = entry - atr * 1.5
        tp2 = entry - atr * 2.5
        return _plan(
            action="WAIT_SHORT_TRIGGER",
            bias="Bearish",
            confidence=abs(direction_score),
            entry_trigger=f"Short only after price loses {entry:,.2f}",
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            invalidation=f"Cancel short idea if candle closes above {resistance:,.2f}",
            reasons=reasons,
        )

    if high_risk:
        reasons.append("Risk filter blocked fresh entries: volatility or model risk is elevated.")
    else:
        reasons.append("Signals are mixed, so Nero prefers capital protection over forced entry.")
    return IntradayTradePlan(
        action="NO_TRADE",
        bias="Neutral",
        confidence=round(abs(direction_score), 2),
        entry_trigger="No fresh trade until bias and trigger align.",
        entry_price=last_close,
        stop_loss=0.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
        risk_reward_1=0.0,
        risk_reward_2=0.0,
        invalidation="Wait for a cleaner breakout, breakdown, or liquidity reclaim.",
        status="Stand aside",
        reasons=reasons,
    )


def _direction_score(
    macro_direction: str,
    news_sentiment: str,
    news_score: int,
    technical_bias: float,
    confluence_score: float,
) -> float:
    macro = {"bullish": 0.45, "bearish": -0.45}.get(macro_direction.lower(), 0.0)
    sentiment = max(-1.0, min(1.0, news_score / 10)) * 0.25
    if news_sentiment.lower() == "bearish":
        sentiment = -abs(sentiment)
    elif news_sentiment.lower() == "bullish":
        sentiment = abs(sentiment)
    confluence = ((confluence_score - 50) / 50) * 0.4
    score = macro + sentiment + technical_bias * 0.55 + confluence
    return max(-1.0, min(1.0, score))


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(period).mean())


def _plan(
    action: str,
    bias: str,
    confidence: float,
    entry_trigger: str,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    invalidation: str,
    reasons: list[str],
) -> IntradayTradePlan:
    risk = abs(entry_price - stop_loss)
    reward_1 = abs(take_profit_1 - entry_price)
    reward_2 = abs(take_profit_2 - entry_price)
    return IntradayTradePlan(
        action=action,
        bias=bias,
        confidence=round(max(0.0, min(1.0, confidence)), 2),
        entry_trigger=entry_trigger,
        entry_price=round(entry_price, 4),
        stop_loss=round(stop_loss, 4),
        take_profit_1=round(take_profit_1, 4),
        take_profit_2=round(take_profit_2, 4),
        risk_reward_1=round(reward_1 / risk, 2) if risk else 0.0,
        risk_reward_2=round(reward_2 / risk, 2) if risk else 0.0,
        invalidation=invalidation,
        status="Wait for trigger",
        reasons=reasons,
    )


def _empty_plan(reason: str) -> IntradayTradePlan:
    return IntradayTradePlan(
        action="NO_TRADE",
        bias="Neutral",
        confidence=0.0,
        entry_trigger=reason,
        entry_price=0.0,
        stop_loss=0.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
        risk_reward_1=0.0,
        risk_reward_2=0.0,
        invalidation="No setup.",
        status="Stand aside",
        reasons=[reason],
    )
