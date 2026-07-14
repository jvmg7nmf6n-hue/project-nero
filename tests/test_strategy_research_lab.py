from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_research_lab import build_strategy_research_report


class StrategyResearchLabTest(unittest.TestCase):
    def test_research_lab_generates_candidates_from_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "mean_reversion_report.csv"
            closed_path = root / "closed_trades.csv"
            eval_path = root / "evaluations.csv"
            pd.DataFrame(
                [
                    {
                        "asset": "COMBINED",
                        "total_trades": 2,
                        "win_rate": 0.5,
                        "expectancy_r": 0.1,
                        "profit_factor": 1.4,
                        "rejected_setup_counts": '{"CLOSE_NOT_BELOW_LOWER_BB": 80, "RSI_NOT_BELOW_35": 50, "TARGET_NOT_ABOVE_ENTRY": 30}',
                    }
                ]
            ).to_csv(report_path, index=False)
            pd.DataFrame([{"net_pnl": 100, "r_multiple": 1.0}, {"net_pnl": -50, "r_multiple": -0.5}]).to_csv(closed_path, index=False)
            pd.DataFrame().to_csv(eval_path, index=False)

            result = build_strategy_research_report(report_path, closed_path, eval_path)

        self.assertEqual(result.sample_status, "INSUFFICIENT_SAMPLE")
        self.assertTrue(result.candidates)
        candidate_ids = {candidate.candidate_id for candidate in result.candidates}
        self.assertIn("MR_RELAXED_PULLBACK_V1", candidate_ids)
        self.assertTrue(all(candidate.status == "RESEARCH_ONLY" for candidate in result.candidates))

    def test_research_lab_handles_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = build_strategy_research_report(root / "missing.csv", root / "missing_closed.csv", root / "missing_eval.csv")

        self.assertEqual(result.current_edge, "UNPROVEN_NO_TRADES")
        self.assertTrue(result.candidates)
        self.assertEqual(result.label, "RESEARCH_MODE_EARLY")


if __name__ == "__main__":
    unittest.main()
