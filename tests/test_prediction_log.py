from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.prediction_log import build_prediction_truth_report


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


if __name__ == "__main__":
    unittest.main()
