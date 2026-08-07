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
            lineage_csv = base / "lineage.csv"
            lineage_json = base / "lineage.json"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "BREAKOUT_MOMENTUM_V1",
                        "quarantined_label": "OLD_BREAKOUT",
                        "quarantined_trades": 92,
                        "quarantined_net_pnl": -1549.48,
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
                lineage_csv=lineage_csv,
                lineage_json=lineage_json,
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["status"], "FORWARD_REPAIR_TRACKING")
        self.assertEqual(row["fresh_data_mode"], "FORWARD_PAPER_FROM_TODAY")
        self.assertEqual(row["anti_overfit_guard"], "PASS_FORWARD_PAPER_ACCEPTED")
        self.assertEqual(row["forward_start"], "2026-08-01")
        self.assertEqual(row["failure_reason_code"], "CAPITAL_DRAIN")
        self.assertEqual(row["sample_milestone"], "UNDER_30_COLLECTING")
        self.assertEqual(row["promotion_decision"], "COLLECT_FRESH_DATA_ACCOUNTING_ONLY")
        self.assertEqual(row["repair_quality_label"], "ACCOUNTING_PROFIT_ONLY")

    def test_active_forward_rows_do_not_increment_attempt_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            attempts = base / "attempts.csv"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "MR_REGIME_FILTER_V1",
                        "quarantined_label": "OLD_MR_REGIME",
                        "repair_candidate": "REPAIR_MR_REGIME_LATE_V1",
                        "repair_label": "FIX_MR_LATE",
                    }
                ]
            ).to_csv(workbench, index=False)
            pd.DataFrame(
                [
                    {
                        "repair_candidate": "REPAIR_MR_REGIME_LATE_V1",
                        "attempt_number": 1,
                        "status": "FORWARD_REPAIR_TRACKING",
                    }
                ]
            ).to_csv(attempts, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=attempts,
                attempts_json=base / "attempts.json",
                original_windows_csv=base / "missing_windows.csv",
                lineage_csv=base / "lineage.csv",
                lineage_json=base / "lineage.json",
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        self.assertEqual(int(report.iloc[0]["attempt_number"]), 1)

    def test_repair_lab_blocks_fifth_attempt_as_permanently_dead(self) -> None:
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
                    {"repair_candidate": "REPAIR_BREAKOUT_QUALITY_V1", "attempt_number": MAX_REPAIR_ATTEMPTS, "status": "REJECTED"}
                ]
            ).to_csv(attempts, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=attempts,
                attempts_json=out_json,
                original_windows_csv=base / "missing_windows.csv",
                lineage_csv=base / "lineage.csv",
                lineage_json=base / "lineage.json",
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["status"], "ATTEMPT_CAP_REACHED")
        self.assertEqual(row["anti_overfit_guard"], "BLOCKED_MAX_4_ATTEMPTS")
        self.assertEqual(row["promotion_decision"], "PERMANENTLY_DEAD")
        self.assertEqual(row["lineage_status"], "DEAD_AFTER_4")
        self.assertEqual(int(row["attempt_number"]), MAX_REPAIR_ATTEMPTS)


    def test_accounting_profit_only_repair_is_not_promoted_as_r_plus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            summary = base / "summary.csv"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "OLD_BREAKOUT",
                        "quarantined_label": "OLD_BREAKOUT",
                        "repair_candidate": "FIX_BREAKOUT_QUALITY",
                        "repair_label": "FIX_BREAKOUT_QUALITY",
                    }
                ]
            ).to_csv(workbench, index=False)
            pd.DataFrame(
                [
                    {"candidate_id": "OLD_BREAKOUT", "total_trades": 92, "net_pnl": -1549.48, "expectancy_r": -0.32, "profit_factor": 0.55},
                    {"candidate_id": "FIX_BREAKOUT_QUALITY", "total_trades": 19, "net_pnl": 169.55, "expectancy_r": -0.3586, "profit_factor": 1.37},
                ]
            ).to_csv(summary, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=base / "attempts.csv",
                attempts_json=base / "attempts.json",
                original_windows_csv=base / "missing_windows.csv",
                summary_csv=summary,
                lineage_csv=base / "lineage.csv",
                lineage_json=base / "lineage.json",
                now=datetime(2026, 8, 7, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["repair_quality_label"], "ACCOUNTING_PROFIT_ONLY")
        self.assertEqual(row["promotion_decision"], "COLLECT_FRESH_DATA_ACCOUNTING_ONLY")
        self.assertIn("R is not", row["next_action"])

    def test_r_plus_repair_gets_early_label_but_still_needs_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            summary = base / "summary.csv"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "OLD_BAD",
                        "quarantined_label": "OLD_BAD",
                        "repair_candidate": "FIX_R_PLUS",
                        "repair_label": "FIX_R_PLUS",
                    }
                ]
            ).to_csv(workbench, index=False)
            pd.DataFrame(
                [
                    {"candidate_id": "OLD_BAD", "total_trades": 40, "net_pnl": -500.0, "expectancy_r": -0.2, "profit_factor": 0.6},
                    {"candidate_id": "FIX_R_PLUS", "total_trades": 12, "net_pnl": 120.0, "expectancy_r": 0.12, "profit_factor": 1.45},
                ]
            ).to_csv(summary, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=base / "attempts.csv",
                attempts_json=base / "attempts.json",
                original_windows_csv=base / "missing_windows.csv",
                summary_csv=summary,
                lineage_csv=base / "lineage.csv",
                lineage_json=base / "lineage.json",
                now=datetime(2026, 8, 7, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["repair_quality_label"], "R_PLUS_EARLY")
        self.assertEqual(row["promotion_decision"], "COLLECT_FRESH_DATA")

    def test_repair_can_reach_shadow_promotion_only_after_baseline_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workbench = base / "workbench.csv"
            summary = base / "summary.csv"
            baseline = base / "baseline.csv"
            pd.DataFrame(
                [
                    {
                        "quarantined_strategy": "PARENT_BAD",
                        "quarantined_label": "OLD_BAD",
                        "quarantined_trades": 60,
                        "quarantined_net_pnl": -600.0,
                        "repair_candidate": "REPAIR_GOOD",
                        "repair_label": "FIX_GOOD",
                    }
                ]
            ).to_csv(workbench, index=False)
            pd.DataFrame(
                [
                    {"candidate_id": "PARENT_BAD", "total_trades": 60, "net_pnl": -600.0, "expectancy_r": -0.1, "profit_factor": 0.7},
                    {"candidate_id": "REPAIR_GOOD", "total_trades": 55, "net_pnl": 250.0, "expectancy_r": 0.08, "profit_factor": 1.35},
                ]
            ).to_csv(summary, index=False)
            pd.DataFrame([{"candidate_id": "REPAIR_GOOD", "expectancy_r": 0.01}]).to_csv(baseline, index=False)

            report = build_strategy_repair_lab_report(
                workbench_csv=workbench,
                attempts_csv=base / "attempts.csv",
                attempts_json=base / "attempts.json",
                original_windows_csv=base / "missing_windows.csv",
                summary_csv=summary,
                random_baseline_csv=baseline,
                lineage_csv=base / "lineage.csv",
                lineage_json=base / "lineage.json",
                now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

        row = report.iloc[0]
        self.assertEqual(row["sample_milestone"], "REVIEW_50")
        self.assertEqual(row["random_baseline_status"], "BASELINE_AVAILABLE")
        self.assertTrue(bool(row["beats_parent"]))
        self.assertTrue(bool(row["beats_random_baseline"]))
        self.assertEqual(row["promotion_decision"], "PROMOTE_SHADOW")


if __name__ == "__main__":
    unittest.main()
