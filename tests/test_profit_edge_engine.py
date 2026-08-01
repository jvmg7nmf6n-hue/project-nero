from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.profit_edge_engine import build_profit_edge_report


class ProfitEdgeEngineTests(unittest.TestCase):
    def test_profit_candidates_receive_paper_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            quarantine = base / "quarantine.csv"
            out_csv = base / "edge.csv"
            out_json = base / "edge.json"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GOOD",
                        "display_label": "GOOD_EDGE",
                        "verdict": "WATCHLIST",
                        "total_trades": 12,
                        "win_rate": 0.67,
                        "expectancy_r": 0.16,
                        "profit_factor": 1.6,
                        "max_drawdown": -0.03,
                        "net_pnl": 170.0,
                    },
                    {
                        "candidate_id": "BAD",
                        "display_label": "BAD_EDGE",
                        "verdict": "QUARANTINE",
                        "total_trades": 40,
                        "win_rate": 0.35,
                        "expectancy_r": -0.4,
                        "profit_factor": 0.6,
                        "max_drawdown": -0.1,
                        "net_pnl": -900.0,
                    },
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame([{"candidate_id": "BAD", "display_label": "BAD_EDGE", "net_pnl": -900.0}]).to_csv(quarantine, index=False)

            report, summary = build_profit_edge_report(verification, quarantine, out_csv, out_json)

            good = report[report["candidate_id"].eq("GOOD")].iloc[0]
            bad = report[report["candidate_id"].eq("BAD")].iloc[0]
            self.assertEqual(good["role"], "PROFIT_CANDIDATE")
            self.assertGreater(good["paper_weight"], 0)
            self.assertEqual(bad["role"], "CAPITAL_DRAIN")
            self.assertEqual(bad["paper_weight"], 0)
            self.assertEqual(summary.top_candidate, "GOOD_EDGE")
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_json.exists())

    def test_too_small_winner_is_not_profit_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            quarantine = base / "quarantine.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "TINY",
                        "display_label": "TINY",
                        "verdict": "INSUFFICIENT_SAMPLE",
                        "total_trades": 2,
                        "win_rate": 1.0,
                        "expectancy_r": 1.0,
                        "profit_factor": 10.0,
                        "max_drawdown": 0.0,
                        "net_pnl": 200.0,
                    }
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame().to_csv(quarantine, index=False)

            report, summary = build_profit_edge_report(verification, quarantine, base / "edge.csv", base / "edge.json")

            self.assertEqual(report.iloc[0]["role"], "TOO_EARLY")
            self.assertEqual(summary.profit_candidates, 0)


if __name__ == "__main__":
    unittest.main()
