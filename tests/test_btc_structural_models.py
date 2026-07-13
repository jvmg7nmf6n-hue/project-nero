from __future__ import annotations

import unittest

from nero_app.core.btc_structural_models import (
    block_subsidy_for_height,
    build_btc_structural_report,
    issued_supply_for_height,
)


class BtcStructuralModelsTest(unittest.TestCase):
    def test_halving_subsidy_schedule(self) -> None:
        self.assertEqual(block_subsidy_for_height(0), 50.0)
        self.assertEqual(block_subsidy_for_height(210_000), 25.0)
        self.assertEqual(block_subsidy_for_height(840_000), 3.125)

    def test_issued_supply_progresses_by_era(self) -> None:
        self.assertAlmostEqual(issued_supply_for_height(210_000), 10_500_000.0)
        self.assertAlmostEqual(issued_supply_for_height(420_000), 15_750_000.0)

    def test_structural_report_scores_current_scarcity(self) -> None:
        report = build_btc_structural_report(current_price=120_000, block_height=900_000, miner_cost_floor=55_000)

        self.assertGreaterEqual(report.structural_score, 70)
        self.assertEqual(report.structural_label, "STRUCTURAL_SUPPORTIVE")
        self.assertGreater(report.stock_to_flow, 100)
        self.assertGreater(report.miner_floor_ratio, 2)

    def test_miner_floor_stress_penalizes_score(self) -> None:
        report = build_btc_structural_report(current_price=45_000, block_height=900_000, miner_cost_floor=55_000)

        self.assertLess(report.miner_floor_ratio, 1)
        self.assertIn("miner stress", " ".join(report.notes).lower())


if __name__ == "__main__":
    unittest.main()
