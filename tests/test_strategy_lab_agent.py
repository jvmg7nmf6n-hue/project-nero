from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_lab_agent import CANDIDATES, write_strategy_lab_summary
from tools.nero_strategy_lab_weekly_report import build_strategy_lab_weekly_report


class StrategyLabAgentTest(unittest.TestCase):
    def test_has_old_and_new_lab_candidates(self) -> None:
        self.assertGreaterEqual(len(CANDIDATES), 16)
        self.assertIn("MR_RELAXED_PULLBACK_V1", CANDIDATES)
        self.assertIn("BREAKOUT_MOMENTUM_V1", CANDIDATES)
        self.assertEqual(CANDIDATES["MR_RELAXED_PULLBACK_V1"].display_label, "OLD_MR_RELAXED")
        self.assertEqual(CANDIDATES["NEW_BTC_12H_MR"].bucket, "NEW_TEST")
        self.assertEqual(CANDIDATES["NEW_BTC_12H_MR"].interval, "12h")
        self.assertFalse(CANDIDATES["NEW_BTC_ETH_12H_PAIR"].enabled)
        self.assertEqual(CANDIDATES["V2_BREAKOUT_RETEST"].bucket, "V2_SHADOW")
        self.assertEqual(CANDIDATES["V2_MR_RECOVERY"].display_label, "V2_MR_RECOVERY")
        self.assertTrue(CANDIDATES["V2_MR_REWARD"].require_rsi_recovery)

    def test_summary_rates_candidate_after_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            candidate = CANDIDATES["MR_RELAXED_PULLBACK_V1"]
            pd.DataFrame(
                [
                    {
                        "asset": "COMBINED",
                        "total_trades": 32,
                        "win_rate": 0.56,
                        "expectancy_r": 0.24,
                        "profit_factor": 1.8,
                        "max_drawdown": -0.04,
                        "net_pnl": 740.0,
                        "rating": "KEEP_TESTING",
                        "rating_score": 72.0,
                        "insufficient_sample": False,
                    }
                ]
            ).to_csv(report_dir / f"strategy_lab_{candidate.candidate_id}.csv", index=False)

            summary = write_strategy_lab_summary(report_dir, [candidate])

        self.assertEqual(summary.iloc[0]["candidate_id"], candidate.candidate_id)
        self.assertEqual(summary.iloc[0]["display_label"], "OLD_MR_RELAXED")
        self.assertEqual(summary.iloc[0]["bucket"], "OLD_TEST")
        self.assertEqual(summary.iloc[0]["interval"], "1h")
        self.assertEqual(int(summary.iloc[0]["total_trades"]), 32)
        self.assertEqual(summary.iloc[0]["rating"], "KEEP_TESTING")

    def test_weekly_report_mentions_sample_warning(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": "MR_RELAXED_PULLBACK_V1",
                    "total_trades": 3,
                    "win_rate": 0.33,
                    "expectancy_r": -0.1,
                    "profit_factor": 0.8,
                    "max_drawdown": -0.02,
                    "rating": "INSUFFICIENT_SAMPLE",
                    "rating_score": 41,
                }
            ]
        )
        report = build_strategy_lab_weekly_report(summary)
        self.assertIn("Sample warning", report)
        self.assertIn("MR_RELAXED_PULLBACK_V1", report)


if __name__ == "__main__":
    unittest.main()
