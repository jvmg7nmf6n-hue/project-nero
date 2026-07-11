from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.social_intelligence import filter_watchlist_for_asset, score_social_post, summarize_social_intel


class SocialIntelligenceTest(unittest.TestCase):
    def test_filters_asset_watchlist(self) -> None:
        watchlist = pd.DataFrame(
            [
                {"name": "BTC Voice", "asset_focus": "BTC|Crypto", "style": "technical", "starting_reliability": 55, "risk_flags": "none"},
                {"name": "Gold Voice", "asset_focus": "Gold|Forex", "style": "macro", "starting_reliability": 50, "risk_flags": "needs_verification"},
            ]
        )

        result = filter_watchlist_for_asset("BTC", watchlist)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "BTC Voice")

    def test_summary_counts_reliability(self) -> None:
        watchlist = pd.DataFrame(
            [
                {"asset_focus": "BTC|Crypto", "style": "technical|macro", "starting_reliability": 60, "risk_flags": "clear_plan"},
                {"asset_focus": "Crypto", "style": "technical", "starting_reliability": 45, "risk_flags": "hype_risk"},
            ]
        )

        summary = summarize_social_intel("BTC", watchlist)

        self.assertEqual(summary.tracked_voices, 2)
        self.assertEqual(summary.high_reliability_voices, 1)
        self.assertIn("technical", summary.dominant_styles)

    def test_scores_social_post_trade_plan(self) -> None:
        score = score_social_post("BTC breakout long entry 64000 stop 62500 target 68000")

        self.assertIn("BTC", score["assets"])
        self.assertEqual(score["sentiment"], "bullish")
        self.assertTrue(score["has_trade_plan"])


if __name__ == "__main__":
    unittest.main()
