from __future__ import annotations

import unittest

from nero_app.core.consensus_engine import build_consensus_decision
from nero_app.core.historical_market_memory import RegimeSimilarity
from nero_app.core.schema import AssessmentOutput, VerdictOutput
from nero_app.core.trade_desk import IntradayTradePlan


class ConsensusEngineTest(unittest.TestCase):
    def test_blocks_no_trade_when_trade_desk_stands_aside(self) -> None:
        decision = build_consensus_decision(
            verdict=VerdictOutput(direction="neutral", confidence=0.4, risk_score=0.7, summary="", drivers=[]),
            assessment=_assessment(confluence_score=55, market_regime="Range"),
            trade_plan=_plan("NO_TRADE", confidence=0.1),
            news_sentiment="Bullish",
            news_score=6,
        )

        self.assertEqual(decision.decision_class, "NO_TRADE")
        self.assertIn("standing aside", " ".join(decision.blockers))

    def test_high_quality_long_when_evidence_aligns(self) -> None:
        decision = build_consensus_decision(
            verdict=VerdictOutput(direction="bullish", confidence=0.82, risk_score=0.28, summary="", drivers=[]),
            assessment=_assessment(confluence_score=86, market_regime="Bull"),
            trade_plan=_plan("WAIT_LONG_TRIGGER", confidence=0.86),
            news_sentiment="Bullish",
            news_score=8,
            market_memory=RegimeSimilarity(
                asset="BTC",
                reference_regime="BTC_120K_RALLY",
                score=88,
                supportive_factors=["etf_inflows"],
                missing_factors=[],
                risk_factors=[],
                matched_events=4,
                verdict="high similarity to historical rally regime",
            ),
        )

        self.assertIn(decision.decision_class, {"NORMAL_TRADE", "HIGH_CONVICTION"})
        self.assertGreaterEqual(decision.trade_quality, 65)
        self.assertEqual(decision.direction, "LONG")


def _assessment(confluence_score: float, market_regime: str) -> AssessmentOutput:
    return AssessmentOutput(
        rsi=55,
        trend_score=0.4,
        fair_value_gap="none",
        liquidity_sweep="none",
        momentum_score=0.3,
        macd_signal="bullish",
        ma_alignment="bullish",
        atr_pct=1.2,
        confluence_score=confluence_score,
        confluence_label="Test confluence",
        market_regime=market_regime,
        volatility_regime="Normal-Vol",
        bos_signal="bullish",
        technical_bias_score=0.55,
    )


def _plan(action: str, confidence: float) -> IntradayTradePlan:
    return IntradayTradePlan(
        action=action,
        bias="Bullish" if action == "WAIT_LONG_TRIGGER" else "Neutral",
        confidence=confidence,
        entry_trigger="test trigger",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        take_profit_2=115.0,
        risk_reward_1=2.0,
        risk_reward_2=3.0,
        invalidation="test invalidation",
        status="Wait for trigger" if action != "NO_TRADE" else "Stand aside",
        reasons=["test"],
    )


if __name__ == "__main__":
    unittest.main()

