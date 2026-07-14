from __future__ import annotations

import unittest

from nero_app.core.nero_chat import NeroChatContext, answer_nero_chat


def sample_context(**overrides) -> NeroChatContext:
    values = {
        "asset": "BTC",
        "data_status": "live",
        "verdict_direction": "neutral",
        "verdict_confidence": 0.4,
        "risk_score": 0.7,
        "trade_action": "NO_TRADE",
        "trade_bias": "Neutral",
        "trade_confidence": 0.25,
        "entry_trigger": "No fresh trade until bias and trigger align.",
        "stop_loss": 0.0,
        "take_profit_1": 0.0,
        "take_profit_2": 0.0,
        "invalidation": "Wait for a cleaner setup.",
        "consensus_class": "NO_TRADE",
        "consensus_quality": 31.0,
        "consensus_direction": "NEUTRAL",
        "sentiment": "Neutral",
        "sentiment_score": 0,
        "confluence_score": 45.0,
        "market_regime": "Range",
        "volatility_regime": "Normal-Vol",
        "blockers": ["Trade Desk is standing aside"],
        "reasons": ["Signals are mixed"],
        "test_lab": [],
    }
    values.update(overrides)
    return NeroChatContext(**values)


class NeroChatTest(unittest.TestCase):
    def test_trade_question_explains_no_trade(self) -> None:
        answer = answer_nero_chat("abhi trade lena chahiye?", sample_context())
        self.assertIn("fresh paper trade", answer)
        self.assertIn("Main blockers", answer)

    def test_trigger_question_shows_long_plan(self) -> None:
        answer = answer_nero_chat(
            "next trigger kya hai?",
            sample_context(
                trade_action="WAIT_LONG_TRIGGER",
                trade_bias="Bullish",
                entry_trigger="Long only after price holds above 65000",
                stop_loss=64000,
                take_profit_1=66000,
                take_profit_2=67000,
                consensus_class="NORMAL_TRADE",
                consensus_quality=76,
                consensus_direction="LONG",
            ),
        )
        self.assertIn("65000", answer)
        self.assertIn("SL", answer)

    def test_algo_question_uses_test_lab_rows(self) -> None:
        answer = answer_nero_chat(
            "best algo kaunsa hai?",
            sample_context(
                test_lab=[
                    {"candidate_id": "A", "rating_score": 40, "total_trades": 10, "rating": "WATCHLIST", "expectancy_r": 0.1, "profit_factor": 1.2},
                    {"candidate_id": "B", "rating_score": 70, "total_trades": 12, "rating": "KEEP_TESTING", "expectancy_r": 0.3, "profit_factor": 1.8},
                ]
            ),
        )
        self.assertIn("B", answer)
        self.assertIn("Sample warning", answer)


if __name__ == "__main__":
    unittest.main()
