from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_evolution import build_strategy_evolution_report, write_strategy_evolution_report


class StrategyEvolutionTest(unittest.TestCase):
    def test_evolution_report_reworks_negative_edge_and_proposes_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            lab_dir = root / "strategy_lab"
            report_dir.mkdir()
            trade_dir = lab_dir / "BREAKOUT_MOMENTUM_V1" / "trades"
            trade_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BREAKOUT_MOMENTUM_V1",
                        "family": "Momentum",
                        "title": "20-bar breakout momentum",
                        "total_trades": 40,
                        "win_rate": 0.45,
                        "expectancy_r": -0.2,
                        "profit_factor": 0.7,
                        "max_drawdown": -0.06,
                        "net_pnl": -300,
                        "rating_score": 44,
                        "rating": "REJECT_OR_REWORK",
                        "insufficient_sample": False,
                    }
                ]
            ).to_csv(report_dir / "strategy_lab_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BREAKOUT_MOMENTUM_V1",
                        "asset": "BTC",
                        "exit_reason": "SL",
                        "r_multiple": -1.1,
                        "entry_rsi": 61,
                        "planned_reward_r": 1.1,
                    }
                ]
            ).to_csv(trade_dir / "closed_trades.csv", index=False)

            report = build_strategy_evolution_report(lab_dir=lab_dir, report_dir=report_dir)

        self.assertEqual(report.total_trades, 40)
        self.assertEqual(report.total_losses, 1)
        self.assertEqual(report.recommendation_rows[0]["Action"], "REWORK")
        self.assertEqual(report.variant_rows[0]["Proposed Variant"], "BREAKOUT_MOMENTUM_V2")
        self.assertIn("Stop-loss", report.autopsy_rows[0]["Likely Mistake"])


    def test_asset_failure_correction_flags_promising_and_quarantine_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            lab_dir = root / "strategy_lab"
            report_dir.mkdir()
            trade_dir = lab_dir / "MIXED_TEST" / "trades"
            trade_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"candidate_id": "MIXED_TEST", "family": "Mixed", "total_trades": 10, "win_rate": 0.5, "expectancy_r": 0.1, "profit_factor": 1.2, "max_drawdown": -0.03},
                ]
            ).to_csv(report_dir / "strategy_lab_summary.csv", index=False)
            pd.DataFrame(
                [
                    *[{"candidate_id": "MIXED_TEST", "asset": "OIL_FUT", "exit_reason": "TARGET", "r_multiple": 0.8, "net_pnl": 40} for _ in range(5)],
                    *[{"candidate_id": "MIXED_TEST", "asset": "GBPUSD", "exit_reason": "SL", "r_multiple": -1.0, "net_pnl": -25} for _ in range(5)],
                ]
            ).to_csv(trade_dir / "closed_trades.csv", index=False)

            report = build_strategy_evolution_report(lab_dir=lab_dir, report_dir=report_dir)

        actions = {row["Asset"]: row["Action"] for row in report.asset_action_rows}
        self.assertEqual(actions["OIL_FUT"], "PROMISING_WATCH")
        self.assertEqual(actions["GBPUSD"], "QUARANTINE")

    def test_write_strategy_evolution_report_outputs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            report = build_strategy_evolution_report(lab_dir=report_dir / "missing", report_dir=report_dir)
            write_strategy_evolution_report(report, report_dir=report_dir)

            self.assertTrue((report_dir / "strategy_evolution_report.json").exists())
            self.assertTrue((report_dir / "strategy_evolution_recommendations.csv").exists())


if __name__ == "__main__":
    unittest.main()
