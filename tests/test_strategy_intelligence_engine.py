from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_intelligence_engine import build_strategy_intelligence_report


class StrategyIntelligenceEngineTests(unittest.TestCase):
    def test_capital_drain_breakout_creates_retest_quality_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            quarantine = base / "quarantine.csv"
            output_csv = base / "brain.csv"
            output_json = base / "brain.json"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BREAKOUT_MOMENTUM_V1",
                        "display_label": "OLD_BREAKOUT",
                        "family": "Momentum",
                        "verdict": "QUARANTINE",
                        "total_trades": 92,
                        "net_pnl": -1549.48,
                        "expectancy_r": -0.31,
                        "profit_factor": 0.61,
                    }
                ]
            ).to_csv(verification, index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BREAKOUT_MOMENTUM_V1",
                        "display_label": "OLD_BREAKOUT",
                        "total_trades": 92,
                        "net_pnl": -1549.48,
                        "reason": "30+ trade sample shows negative or weak risk-adjusted edge.",
                    }
                ]
            ).to_csv(quarantine, index=False)

            report, summary = build_strategy_intelligence_report(
                verification_csv=verification,
                quarantine_csv=quarantine,
                repair_lab_csv=base / "missing_repair.csv",
                output_csv=output_csv,
                output_json=output_json,
            )

        row = report.iloc[0]
        self.assertEqual(summary.status, "TEST_LAB_DESIGNS_READY")
        self.assertEqual(row["failure_class"], "CAPITAL_DRAIN")
        self.assertIn("RETEST", row["proposed_label"])
        self.assertEqual(int(row["rule_change_count"]), 2)
        self.assertEqual(row["decision"], "READY_FOR_TEST_LAB_DESIGN")

    def test_proposal_contract_contains_anti_overfit_fresh_data_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "MR_RELAXED_PULLBACK_V1",
                        "display_label": "OLD_MR_RELAXED",
                        "family": "Mean Reversion",
                        "verdict": "QUARANTINE",
                        "total_trades": 41,
                        "net_pnl": -671.40,
                        "expectancy_r": -0.16,
                        "profit_factor": 0.74,
                    }
                ]
            ).to_csv(verification, index=False)

            report, _summary = build_strategy_intelligence_report(
                verification_csv=verification,
                quarantine_csv=base / "missing_quarantine.csv",
                repair_lab_csv=base / "missing_repair.csv",
                output_csv=base / "brain.csv",
                output_json=base / "brain.json",
            )

        row = report.iloc[0]
        contract = json.loads(row["proposal_contract"])
        self.assertEqual(contract["fresh_data_requirement"], "forward paper from next run or documented non-overlapping historical window")
        self.assertIn("same_window_retest_promotion", contract["blocked"])
        self.assertTrue(contract["promotion_gates"]["beats_random_baseline"])
        self.assertLessEqual(int(row["rule_change_count"]), 2)

    def test_small_sample_failure_stays_in_research_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "SHORT_SOL_4H",
                        "display_label": "SHORT_SOL_4H",
                        "family": "Short Momentum",
                        "verdict": "INSUFFICIENT_SAMPLE",
                        "total_trades": 7,
                        "net_pnl": -70.78,
                        "expectancy_r": -0.14,
                        "profit_factor": 0.72,
                    }
                ]
            ).to_csv(verification, index=False)

            report, summary = build_strategy_intelligence_report(
                verification_csv=verification,
                quarantine_csv=base / "missing_quarantine.csv",
                repair_lab_csv=base / "missing_repair.csv",
                output_csv=base / "brain.csv",
                output_json=base / "brain.json",
            )

        self.assertEqual(summary.test_lab_ready, 0)
        self.assertEqual(report.iloc[0]["decision"], "RESEARCH_QUEUE_SMALL_SAMPLE")


if __name__ == "__main__":
    unittest.main()
