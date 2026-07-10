from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nero_app.core.prediction_lab import _already_recorded_today, write_prediction_lab_report
from tools.nero_prediction_lab import format_prediction_lab_alert


class PredictionLabTest(unittest.TestCase):
    def test_already_recorded_today_detects_prediction_lab_row(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_log.csv"
            path.write_text(
                "timestamp,asset,headline,direction,confidence,risk_score,thematic_score,momentum_score,rsi,fair_value_gap,liquidity_sweep,data_source,entry_date,entry_close,horizon_days,target_date,evaluation_status,exit_date,exit_close,actual_return,outcome\n"
                "2026-07-10T10:00:00,BTC,Auto,bullish,0.7,0.3,0.1,0.2,40,none,none,live | Prediction Lab,2026-07-10,100,1,2026-07-11,pending,,,,\n",
                encoding="utf-8",
            )
            prices = pd.DataFrame({"date": [pd.Timestamp("2026-07-10")], "close": [100.0]})

            result = _already_recorded_today("BTC", 1, prices, path)

        self.assertTrue(result)

    def test_prediction_lab_report_summarizes_nero_core_outcomes(self) -> None:
        frame = pd.DataFrame(
            [
                {"asset": "BTC", "evaluation_status": "evaluated", "outcome": "win"},
                {"asset": "BTC", "evaluation_status": "evaluated", "outcome": "miss"},
                {"asset": "GOLD", "evaluation_status": "pending", "outcome": ""},
            ]
        )
        with TemporaryDirectory() as directory:
            report = write_prediction_lab_report(frame, Path(directory) / "report.csv")

        btc = report[report["asset"] == "BTC"].iloc[0]
        gold = report[report["asset"] == "GOLD"].iloc[0]
        self.assertEqual(btc["agent"], "NERO_CORE")
        self.assertEqual(int(btc["evaluated"]), 2)
        self.assertEqual(float(btc["win_rate"]), 0.5)
        self.assertEqual(int(gold["pending"]), 1)

    def test_prediction_lab_alert_includes_mobile_summary(self) -> None:
        message = format_prediction_lab_alert(
            assets=["BTC", "GOLD"],
            horizon_days=1,
            recorded=2,
            evaluated=1,
            report_path=Path("reports") / "prediction_lab_report.csv",
        )

        self.assertIn("NERO Prediction Lab", message)
        self.assertIn("Assets: BTC,GOLD", message)
        self.assertIn("Recorded: 2", message)
        self.assertIn("Evaluated: 1", message)
        self.assertIn("reports", message)


if __name__ == "__main__":
    unittest.main()
