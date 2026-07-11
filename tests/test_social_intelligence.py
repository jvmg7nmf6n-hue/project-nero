from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.social_intelligence import (
    build_social_reliability_report,
    evaluate_social_calls,
    filter_watchlist_for_asset,
    score_social_post,
    summarize_social_intel,
)


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

    def test_evaluates_long_call_target_hit(self) -> None:
        calls = pd.DataFrame(
            [
                {
                    "call_id": "c1",
                    "source_name": "Tester",
                    "handle": "@tester",
                    "posted_at": "2026-07-11 00:00:00",
                    "asset": "BTC",
                    "direction": "LONG",
                    "entry": 100.0,
                    "stop": 95.0,
                    "target": 110.0,
                    "status": "pending",
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {"date": "2026-07-11 01:00:00", "open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
                {"date": "2026-07-11 02:00:00", "open": 103.0, "high": 111.0, "low": 102.0, "close": 110.0},
            ]
        )

        evaluated = evaluate_social_calls(calls, prices, horizon_hours=24)

        self.assertEqual(evaluated.iloc[0]["status"], "evaluated")
        self.assertEqual(evaluated.iloc[0]["outcome"], "WIN")
        self.assertAlmostEqual(float(evaluated.iloc[0]["r_multiple"]), 2.0)

    def test_builds_reliability_report(self) -> None:
        calls = pd.DataFrame(
            [
                {"handle": "@a", "source_name": "A", "status": "evaluated", "outcome": "WIN", "r_multiple": 1.5},
                {"handle": "@a", "source_name": "A", "status": "evaluated", "outcome": "LOSS", "r_multiple": -1.0},
                {"handle": "@b", "source_name": "B", "status": "pending", "outcome": "", "r_multiple": ""},
            ]
        )

        report = build_social_reliability_report(calls)

        self.assertIn("reliability_score", report.columns)
        self.assertEqual(int(report[report["handle"] == "@a"].iloc[0]["evaluated_calls"]), 2)


if __name__ == "__main__":
    unittest.main()
