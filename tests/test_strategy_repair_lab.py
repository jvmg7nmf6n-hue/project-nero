from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_repair_lab import (
    MAX_REPAIR_ATTEMPTS,
    build_strategy_repair_lab_report,
    validate_fresh_data_window,
)


class StrategyRepairLabTests(unittest.TestCase):
    def test_validate_fresh_data_window_blocks_overlap(self) -> None:
        result = validate_fresh_data_window("2026-01-01", "2026-03-01", "2026-02-01", "2026-04-01")

        self.assertFalse(result.valid)
        self.assertTrue(result.overlaps_original)
        self.assertEqual(result.reason, "OVERLAPS_ORIGINAL_FAILED_WINDOW")

    def test_validate_fresh_data_window_accepts_unseen_window(self) -> None:
        result = validate_fresh_data_window("2026-01-01", "2026-03-01", "2026-03-02", "2026-04-01")

        self.assertTrue(result.valid)
        self.assertFalse(result.overlaps_original)
        self.assertEqual(result.reason, "FRESH_UNSEEN_WINDOW")

    def test_repair_lab_uses_forward_paper_when_original_window_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            attempts = base / "attempts.csv"
            out_json = base / "attempts.json"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "BREAKOUT_MOMENTUM_V1",
                        "quarantined_label": "OLD_BREAKOUT",
                        "repair_candidate": "REPAIR_BREAKOUT_QUALITY_V1",
                        "repair_label": "FIX_BREAKOUT_QUALITY",
                        "repair_trades": 12,
                        "repair_net_pnl": 173.96,
                    }
                ]
            ).to_csv(workbench, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=attempts,
                attempts_json=out_json,
                original_windows_csv=base / "missing_windows.csv",
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["status"], "FORWARD_REPAIR_TRACKING")
        self.assertEqual(row["fresh_data_mode"], "FORWARD_PAPER_FROM_TODAY")
        self.assertEqual(row["anti_overfit_guard"], "PASS_FORWARD_PAPER_ACCEPTED")
        self.assertEqual(row["forward_start"], "2026-08-01")

    def test_repair_lab_blocks_fifth_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            attempts = base / "attempts.csv"
            out_json = base / "attempts.json"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "BREAKOUT_MOMENTUM_V1",
                        "quarantined_label": "OLD_BREAKOUT",
                        "repair_candidate": "REPAIR_BREAKOUT_QUALITY_V1",
                        "repair_label": "FIX_BREAKOUT_QUALITY",
                    }
                ]
            ).to_csv(workbench, index=False)
            pd.DataFrame(
                [
                    {"repair_candidate": "REPAIR_BREAKOUT_QUALITY_V1", "attempt_number": MAX_REPAIR_ATTEMPTS}
                ]
            ).to_csv(attempts, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=attempts,
                attempts_json=out_json,
                original_windows_csv=base / "missing_windows.csv",
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["status"], "ATTEMPT_CAP_REACHED")
        self.assertEqual(row["anti_overfit_guard"], "BLOCKED_MAX_4_ATTEMPTS")
        self.assertEqual(int(row["attempt_number"]), MAX_REPAIR_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
