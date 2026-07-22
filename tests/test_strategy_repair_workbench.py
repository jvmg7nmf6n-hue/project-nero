from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_repair_workbench import build_strategy_repair_workbench


class StrategyRepairWorkbenchTests(unittest.TestCase):
    def test_maps_quarantined_strategy_to_registered_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            quarantine = base / "quarantine.csv"
            summary = base / "summary.csv"
            out_csv = base / "repair.csv"
            out_json = base / "repair.json"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BREAKOUT_MOMENTUM_V1",
                        "display_label": "OLD_BREAKOUT",
                        "total_trades": 92,
                        "net_pnl": -1549.48,
                        "reason": "Weak edge.",
                    }
                ]
            ).to_csv(quarantine, index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "REPAIR_BREAKOUT_QUALITY_V1",
                        "total_trades": 12,
                        "net_pnl": 173.96,
                    }
                ]
            ).to_csv(summary, index=False)

            report = build_strategy_repair_workbench(quarantine, summary, out_csv, out_json)

            self.assertEqual(len(report), 1)
            row = report.iloc[0]
            self.assertEqual(row["repair_candidate"], "REPAIR_BREAKOUT_QUALITY_V1")
            self.assertEqual(row["repair_label"], "FIX_BREAKOUT_QUALITY")
            self.assertEqual(row["repair_status"], "DEPLOYED_COLLECTING_EVIDENCE")
            self.assertEqual(row["repair_trades"], 12)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_json.exists())

    def test_marks_unmapped_strategy_as_design_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            quarantine = base / "quarantine.csv"
            summary = base / "summary.csv"
            out_csv = base / "repair.csv"
            out_json = base / "repair.json"
            pd.DataFrame(
                [{"candidate_id": "UNKNOWN_BAD", "display_label": "UNKNOWN", "total_trades": 30, "net_pnl": -100}]
            ).to_csv(quarantine, index=False)
            pd.DataFrame().to_csv(summary, index=False)

            report = build_strategy_repair_workbench(quarantine, summary, out_csv, out_json)

        self.assertEqual(report.iloc[0]["repair_candidate"], "NEEDS_DESIGN")
        self.assertEqual(report.iloc[0]["repair_status"], "DESIGN_REQUIRED")


if __name__ == "__main__":
    unittest.main()
