from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.sunflower_profit_bridge import build_sunflower_profit_bridge_report


class SunflowerProfitBridgeTests(unittest.TestCase):
    def test_promotes_only_disciplined_positive_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            edge = base / "edge.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GOOD",
                        "display_label": "GOOD_EDGE",
                        "family": "Momentum",
                        "interval": "4h",
                        "asset_filter": "BTC",
                        "total_trades": 35,
                        "win_rate": 0.55,
                        "expectancy_r": 0.22,
                        "profit_factor": 1.35,
                        "max_drawdown": -0.05,
                        "net_pnl": 650.0,
                        "verdict": "PROMOTE_PAPER",
                        "data_status": "OK",
                    }
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame([{"candidate_id": "GOOD", "role": "PROFIT_CANDIDATE"}]).to_csv(edge, index=False)

            report, summary = build_sunflower_profit_bridge_report(verification, edge, base / "out.csv", base / "out.json")

            self.assertEqual(report.iloc[0]["sunflower_gate"], "DISCIPLINED_PROFIT_CANDIDATE")
            self.assertEqual(report.iloc[0]["decision"], "FOCUS_PAPER_CAPITAL")
            self.assertEqual(summary.disciplined_profit_candidates, 1)
            self.assertEqual(summary.status, "DISCIPLINED_EDGE_FOUND")

    def test_positive_small_sample_stays_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "EARLY",
                        "display_label": "EARLY_EDGE",
                        "total_trades": 8,
                        "win_rate": 0.5,
                        "expectancy_r": 0.4,
                        "profit_factor": 1.2,
                        "max_drawdown": 0.0,
                        "net_pnl": 90.0,
                        "verdict": "WATCHLIST",
                        "data_status": "OK",
                    }
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame().to_csv(base / "edge.csv", index=False)

            report, summary = build_sunflower_profit_bridge_report(verification, base / "edge.csv", base / "out.csv", base / "out.json")

            self.assertEqual(report.iloc[0]["sunflower_gate"], "EARLY_PROFIT_WATCHLIST")
            self.assertEqual(summary.early_profit_watchlist, 1)
            self.assertEqual(summary.disciplined_profit_candidates, 0)

    def test_quarantine_blocks_profit_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            edge = base / "edge.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BAD",
                        "display_label": "BAD_EDGE",
                        "total_trades": 40,
                        "win_rate": 0.35,
                        "expectancy_r": -0.25,
                        "profit_factor": 0.75,
                        "max_drawdown": -0.2,
                        "net_pnl": -700.0,
                        "verdict": "QUARANTINE",
                        "data_status": "OK",
                    }
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame([{"candidate_id": "BAD", "role": "CAPITAL_DRAIN"}]).to_csv(edge, index=False)

            report, summary = build_sunflower_profit_bridge_report(verification, edge, base / "out.csv", base / "out.json")

            self.assertEqual(report.iloc[0]["sunflower_gate"], "CAPITAL_DRAIN_BLOCKED")
            self.assertEqual(report.iloc[0]["decision"], "BLOCK_NEW_ENTRIES")
            self.assertEqual(summary.capital_drains_blocked, 1)


if __name__ == "__main__":
    unittest.main()
