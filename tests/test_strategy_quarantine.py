from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nero_app.core.strategy_lab_agent import run_strategy_lab
from nero_app.core.strategy_quarantine import build_strategy_quarantine_report, load_quarantined_strategy_ids


class StrategyQuarantineTests(unittest.TestCase):
    def test_build_report_keeps_only_quarantine_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            verification = base / "verification.csv"
            out_csv = base / "quarantine.csv"
            out_json = base / "quarantine.json"
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BAD",
                        "display_label": "BAD_STRAT",
                        "verdict": "QUARANTINE",
                        "total_trades": 31,
                        "net_pnl": -500,
                        "expectancy_r": -0.4,
                        "profit_factor": 0.7,
                        "primary_reason": "Weak edge.",
                        "action": "Pause.",
                    },
                    {"candidate_id": "GOOD", "display_label": "GOOD", "verdict": "WATCHLIST"},
                ]
            ).to_csv(verification, index=False)

            report = build_strategy_quarantine_report(verification, out_csv, out_json)
            loaded = load_quarantined_strategy_ids(out_csv)

        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]["candidate_id"], "BAD")
        self.assertEqual(loaded, {"BAD"})

    def test_strategy_lab_skips_quarantined_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            quarantine = base / "strategy_quarantine_report.csv"
            pd.DataFrame(
                [{"candidate_id": "MR_RELAXED_PULLBACK_V1", "display_label": "OLD_MR_RELAXED", "blocked": True}]
            ).to_csv(quarantine, index=False)
            env = {
                "STRATEGY_LAB_CANDIDATES": "MR_RELAXED_PULLBACK_V1",
                "STRATEGY_QUARANTINE_REPORT": str(quarantine),
                "STRATEGY_LAB_DATA_DIR": str(base / "data"),
                "STRATEGY_LAB_REPORT_DIR": str(base / "reports"),
            }
            with patch.dict(os.environ, env, clear=False):
                summary = run_strategy_lab(assets={"BTC": "BTCUSDT"})

        self.assertEqual(summary.candidate_count, 1)
        self.assertEqual(summary.evaluated, 0)
        self.assertEqual(summary.entries, 0)
        self.assertTrue(any("QUARANTINED_BY_VERIFICATION" in alert for alert in summary.alerts))


if __name__ == "__main__":
    unittest.main()
