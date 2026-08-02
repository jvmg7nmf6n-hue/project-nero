from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.daily_hypothesis_report import build_daily_hypothesis_report, format_daily_hypothesis_message


class DailyHypothesisReportTests(unittest.TestCase):
    def test_build_report_detects_new_items_and_prevents_duplicate_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            state = base / "state.json"
            now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
            pd.DataFrame(
                [
                    {
                        "parent_strategy": "OLD_BREAKOUT",
                        "repair_candidate": "REPAIR_BREAKOUT_QUALITY_V1",
                        "repair_label": "FIX_BREAKOUT_QUALITY",
                        "status": "FORWARD_REPAIR_TRACKING",
                        "forward_start": "2026-08-02",
                    },
                    {
                        "parent_strategy": "OLD_MR_DEEP",
                        "repair_candidate": "NEEDS_DESIGN",
                        "repair_label": "NEEDS_DESIGN",
                        "status": "DESIGN_REQUIRED",
                        "forward_start": "2026-08-02",
                    },
                ]
            ).to_csv(base / "strategy_repair_lab_attempts.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "RMR_CONFIRMATION_BTC_1D",
                        "display_label": "RMR_CONFIRM_BTC_1D",
                        "verdict": "WATCHLIST",
                        "net_pnl": 106.58,
                        "expectancy_r": 0.5315,
                    }
                ]
            ).to_csv(base / "strategy_verification_report.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "RMR_CONFIRMATION_BTC_1D",
                        "display_label": "RMR_CONFIRM_BTC_1D",
                        "role": "PROFIT_CANDIDATE",
                        "total_trades": 8,
                        "net_pnl": 106.58,
                        "expectancy_r": 0.5315,
                        "edge_score": 100,
                    }
                ]
            ).to_csv(base / "profit_edge_report.csv", index=False)

            payload, summary = build_daily_hypothesis_report(report_dir=base, state_path=state, now=now)
            self.assertTrue(summary.notification_due)
            self.assertEqual(summary.active_new_hypotheses, 1)
            self.assertEqual(summary.design_required, 1)
            self.assertEqual(summary.new_watchlist_additions, 1)
            self.assertEqual(summary.successful_strategies, 1)
            self.assertEqual(summary.top_strategy, "RMR_CONFIRM_BTC_1D")
            self.assertIn("RMR_CONFIRM_BTC_1D", format_daily_hypothesis_message(summary, payload))

            _, second = build_daily_hypothesis_report(report_dir=base, state_path=state, now=now)
            self.assertFalse(second.notification_due)
            self.assertEqual(second.new_watchlist_additions, 0)

    def test_missing_reports_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = build_daily_hypothesis_report(
                report_dir=Path(tmp),
                state_path=Path(tmp) / "state.json",
                now=datetime(2026, 8, 2, tzinfo=timezone.utc),
            )
        self.assertEqual(summary.successful_strategies, 0)
        self.assertEqual(summary.top_strategy, "NONE")


if __name__ == "__main__":
    unittest.main()

