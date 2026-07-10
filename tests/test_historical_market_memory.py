from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.historical_market_memory import (
    format_regime_report,
    infer_environment_tags,
    score_regime_similarity,
)


class HistoricalMarketMemoryTest(unittest.TestCase):
    def test_btc_120k_similarity_scores_supportive_environment(self) -> None:
        tags = infer_environment_tags(
            asset="BTC",
            news_text="ETF inflows, institutional adoption, tech stocks strong and stablecoin regulation improves.",
            dxy_trend="weak",
            fed_tone="dovish",
            risk_appetite="risk_on",
            etf_flow="inflows",
            policy_tone="friendly",
        )
        events = pd.DataFrame({"reference_regime": ["BTC_120K_RALLY", "BTC_120K_RALLY"]})

        result = score_regime_similarity("BTC", tags, events)

        self.assertGreaterEqual(result.score, 80)
        self.assertEqual(result.matched_events, 2)
        self.assertIn("etf_inflows", result.supportive_factors)
        self.assertIn("BTC Historical Regime Similarity", format_regime_report(result))

    def test_btc_similarity_flags_pressure_risks(self) -> None:
        tags = infer_environment_tags(
            asset="BTC",
            news_text="Strong dollar, hawkish Fed and liquidation risk hit crypto.",
            dxy_trend="strong",
            fed_tone="hawkish",
            risk_appetite="risk_off",
            etf_flow="outflows",
            policy_tone="hostile",
        )

        result = score_regime_similarity("BTC", tags, pd.DataFrame())

        self.assertLess(result.score, 40)
        self.assertTrue(any("dxy_strong" in risk for risk in result.risk_factors))
        self.assertIn("weak similarity", result.verdict)

    def test_gold_rally_similarity_uses_safe_haven_context(self) -> None:
        tags = infer_environment_tags(
            asset="GOLD",
            news_text="Geopolitical tensions and central bank demand support safe haven gold amid inflation risk.",
            dxy_trend="weak",
            fed_tone="dovish",
            risk_appetite="risk_off",
        )

        result = score_regime_similarity("GOLD", tags, pd.DataFrame({"reference_regime": ["GOLD_RALLY"]}))

        self.assertGreaterEqual(result.score, 80)
        self.assertIn("safe_haven", result.supportive_factors)
        self.assertEqual(result.matched_events, 1)


if __name__ == "__main__":
    unittest.main()
