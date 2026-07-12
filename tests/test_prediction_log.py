from __future__ import annotations

import unittest

import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory

from nero_app.core.prediction_log import build_prediction_truth_report, evaluate_prediction_log


class PredictionTruthReportTest(unittest.TestCase):
    def test_truth_report_handles_empty_frame(self) -> None:
        report = build_prediction_truth_report(pd.DataFrame())

        self.assertEqual(report["total"], 0)
        self.assertEqual(report["win_rate"], 0.0)
        self.assertIn("No prediction records", report["notes"][0])

    def test_truth_report_scores_evaluated_predictions(self) -> None:
        frame = pd.DataFrame(
            [
                {"asset": "BTC", "evaluation_status": "evaluated", "outcome": "win", "actual_return": 0.05, "confidence": 0.7},
                {"asset": "BTC", "evaluation_status": "evaluated", "outcome": "miss", "actual_return": -0.02, "confidence": 0.8},
                {"asset": "GOLD", "evaluation_status": "pending", "outcome": "", "actual_return": "", "confidence": 0.5},
            ]
        )

        report = build_prediction_truth_report(frame)

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["evaluated"], 2)
        self.assertEqual(report["pending"], 1)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["misses"], 1)
        self.assertAlmostEqual(report["win_rate"], 0.5)
        self.assertEqual(len(report["rows"]), 2)


    def test_evaluator_only_scores_selected_asset(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2026-07-11", "close": 4100.0},
                {"date": "2026-07-12", "close": 4110.0},
            ]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_log.csv"
            path.write_text(
                "timestamp,asset,headline,direction,confidence,risk_score,thematic_score,momentum_score,rsi,fair_value_gap,liquidity_sweep,data_source,entry_date,entry_close,horizon_days,target_date,evaluation_status,exit_date,exit_close,actual_return,outcome\n"
                "2026-07-10T10:00:00,BTC,test,neutral,0.35,0.5,0.1,0.1,50,none,none,test,2026-07-10,64000,1,2026-07-11,pending,,,,\n"
                "2026-07-10T10:00:00,GOLD,test,neutral,0.35,0.5,0.1,0.1,50,none,none,test,2026-07-10,4000,1,2026-07-11,pending,,,,\n",
                encoding="utf-8",
            )

            frame = evaluate_prediction_log(prices, path=path, asset="GOLD")

        btc = frame[frame["asset"] == "BTC"].iloc[0]
        gold = frame[frame["asset"] == "GOLD"].iloc[0]
        self.assertEqual(btc["evaluation_status"], "pending")
        self.assertEqual(gold["evaluation_status"], "evaluated")
if __name__ == "__main__":
    unittest.main()


