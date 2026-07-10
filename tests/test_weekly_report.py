from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from tools.nero_weekly_report import WeeklyReportPaths, build_weekly_report, read_csv_if_exists


class WeeklyReportTest(unittest.TestCase):
    def test_read_csv_if_exists_returns_empty_for_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            frame = read_csv_if_exists(Path(directory) / "missing.csv")
        self.assertTrue(frame.empty)

    def test_build_weekly_report_summarizes_prediction_and_trade_records(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_log = root / "prediction_log.csv"
            demo_trades = root / "demo_trades.csv"
            prediction_report = root / "prediction_report.csv"
            mean_reversion_report = root / "mean_reversion_report.csv"
            output = root / "weekly.txt"

            prediction_log.write_text(
                "timestamp,asset,evaluation_status,outcome\n"
                "2026-07-10T12:00:00,BTC,pending,\n",
                encoding="utf-8",
            )
            prediction_report.write_text(
                "agent,asset,total,evaluated,wins,misses,win_rate,pending\n"
                "NERO_CORE,BTC,3,2,1,1,0.5,1\n",
                encoding="utf-8",
            )
            demo_trades.write_text(
                "asset,status,result,r_multiple\n"
                "BTC,closed,win,0.5\n"
                "GOLD,closed,loss,-1.0\n",
                encoding="utf-8",
            )
            mean_reversion_report.write_text(
                "asset,total_trades,win_rate,net_pnl,expectancy_r,max_drawdown,insufficient_sample\n"
                "BTC,1,1.0,156.66,1.57,0.0,True\n"
                "COMBINED,1,1.0,156.66,1.57,0.0,True\n",
                encoding="utf-8",
            )

            report = build_weekly_report(
                WeeklyReportPaths(
                    prediction_log=prediction_log,
                    demo_trades=demo_trades,
                    prediction_report=prediction_report,
                    mean_reversion_report=mean_reversion_report,
                    output_report=output,
                ),
                now=datetime(2026, 7, 13, 10, 0, tzinfo=ZoneInfo("Asia/Karachi")),
            )

        self.assertIn("NERO Weekly Performance Report", report)
        self.assertIn("Prediction Lab: 3 total predictions", report)
        self.assertIn("- BTC: total=3, evaluated=2", report)
        self.assertIn("Demo trades: 2 closed trades", report)
        self.assertIn("Mean-Reversion Agent", report)
        self.assertIn("insufficient sample", report)


if __name__ == "__main__":
    unittest.main()
