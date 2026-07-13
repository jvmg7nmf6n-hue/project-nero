from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nero_app.core.strategy_performance_auditor import build_strategy_performance_audit


class StrategyPerformanceAuditorTest(unittest.TestCase):
    def test_auditor_marks_small_sample_insufficient(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mr_report = root / "mean_reversion_report.csv"
            closed = root / "closed_trades.csv"
            evaluations = root / "evaluations.csv"
            prediction_log = root / "prediction_log.csv"

            pd.DataFrame(
                [
                    {"asset": "BTC", "total_trades": 1, "win_rate": 1.0, "net_pnl": 150.0, "expectancy_r": 1.5, "rejected_setup_counts": '{"RSI_NOT_BELOW_35": 4}'},
                    {"asset": "COMBINED", "total_trades": 1, "win_rate": 1.0, "net_pnl": 150.0, "expectancy_r": 1.5, "rejected_setup_counts": '{"RSI_NOT_BELOW_35": 4}'},
                ]
            ).to_csv(mr_report, index=False)
            pd.DataFrame([{"asset": "BTC", "net_pnl": 150.0, "r_multiple": 1.5}]).to_csv(closed, index=False)
            pd.DataFrame([{"asset": "BTC", "rejection_reasons": "RSI_NOT_BELOW_35"}]).to_csv(evaluations, index=False)
            pd.DataFrame(
                [
                    {"asset": "BTC", "evaluation_status": "evaluated", "outcome": "win", "actual_return": 0.02, "confidence": 0.7},
                    {"asset": "BTC", "evaluation_status": "pending", "outcome": "", "actual_return": "", "confidence": 0.5},
                ]
            ).to_csv(prediction_log, index=False)

            report = build_strategy_performance_audit(
                mean_reversion_report_path=mr_report,
                closed_trades_path=closed,
                evaluations_path=evaluations,
                prediction_log_path=prediction_log,
            )

        self.assertEqual(report.grade, "INSUFFICIENT_SAMPLE")
        self.assertTrue(report.insufficient_sample)
        self.assertEqual(report.best_asset, "BTC")
        self.assertIn("RSI_NOT_BELOW_35", report.top_blocker)
        self.assertGreater(report.score, 0)

    def test_auditor_handles_missing_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_strategy_performance_audit(
                mean_reversion_report_path=root / "missing_report.csv",
                closed_trades_path=root / "missing_closed.csv",
                evaluations_path=root / "missing_eval.csv",
                prediction_log_path=root / "missing_predictions.csv",
            )

        self.assertEqual(report.total_closed_trades, 0)
        self.assertEqual(report.total_saved_signals, 0)
        self.assertEqual(report.grade, "INSUFFICIENT_SAMPLE")


if __name__ == "__main__":
    unittest.main()
