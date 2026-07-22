from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_verification import build_strategy_verification_report, verify_strategy


class StrategyVerificationTests(unittest.TestCase):
    def test_promotes_only_after_adequate_positive_sample(self) -> None:
        row = verify_strategy(
            {
                "candidate_id": "GOOD",
                "display_label": "GOOD",
                "total_trades": 35,
                "win_rate": 0.55,
                "expectancy_r": 0.18,
                "profit_factor": 1.35,
                "max_drawdown": -0.05,
                "net_pnl": 450.0,
            }
        )

        self.assertEqual(row.verdict, "PROMOTE_PAPER")
        self.assertEqual(row.sample_status, "ADEQUATE")

    def test_quarantines_negative_sample_after_twenty_trades(self) -> None:
        row = verify_strategy(
            {
                "candidate_id": "BAD",
                "display_label": "BAD",
                "total_trades": 24,
                "win_rate": 0.33,
                "expectancy_r": -0.22,
                "profit_factor": 0.72,
                "max_drawdown": -0.18,
                "net_pnl": -700.0,
            }
        )

        self.assertEqual(row.verdict, "QUARANTINE")
        self.assertEqual(row.sample_status, "EARLY_BUT_ACTIONABLE")

    def test_watchlists_positive_but_small_sample(self) -> None:
        row = verify_strategy(
            {
                "candidate_id": "EARLY",
                "display_label": "EARLY",
                "total_trades": 8,
                "win_rate": 0.75,
                "expectancy_r": 0.4,
                "profit_factor": 1.5,
                "max_drawdown": 0.0,
                "net_pnl": 140.0,
            }
        )

        self.assertEqual(row.verdict, "WATCHLIST")
        self.assertEqual(row.sample_status, "EARLY")

    def test_build_report_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            summary = base / "strategy_lab_summary.csv"
            out_csv = base / "strategy_verification_report.csv"
            out_json = base / "strategy_verification_report.json"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GOOD",
                        "display_label": "GOOD",
                        "bucket": "NEW_TEST",
                        "family": "Mean Reversion",
                        "interval": "1h",
                        "asset_filter": "BTC",
                        "total_trades": 35,
                        "win_rate": 0.55,
                        "expectancy_r": 0.18,
                        "profit_factor": 1.35,
                        "max_drawdown": -0.05,
                        "net_pnl": 450.0,
                        "rating_score": 80,
                        "rating": "KEEP_TESTING",
                    }
                ]
            ).to_csv(summary, index=False)

            report = build_strategy_verification_report(summary, out_csv, out_json)
            self.assertEqual(report.iloc[0]["verdict"], "PROMOTE_PAPER")
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_json.exists())


if __name__ == "__main__":
    unittest.main()

