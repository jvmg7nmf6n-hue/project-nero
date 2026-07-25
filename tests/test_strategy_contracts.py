from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_contracts import (
    build_data_quality_manifest,
    strategy_contract_row,
    write_strategy_architecture_manifests,
)


@dataclass(frozen=True)
class DummySpec:
    candidate_id: str = "TEST_MR"
    display_label: str = "TEST_MR_LABEL"
    bucket: str = "NEW_TEST"
    family: str = "Mean Reversion"
    title: str = "Test mean reversion"
    enabled: bool = True
    interval: str = "4h"
    asset_filter: tuple[str, ...] = ("BTC",)
    asset_exclude: tuple[str, ...] = ()
    evidence_note: str = "unit test"
    entry_side: str = "LONG"
    rsi_entry_below: float = 35.0
    atr_stop_multiple: float = 1.5
    target_mode: str = "FROZEN_MA20"


class StrategyContractTests(unittest.TestCase):
    def test_strategy_contract_has_explicit_rules(self) -> None:
        row = strategy_contract_row(DummySpec())

        self.assertEqual(row.candidate_id, "TEST_MR")
        self.assertEqual(row.asset_scope, "BTC")
        self.assertEqual(row.direction, "LONG")
        self.assertIn("RSI below 35.0", row.entry_rule)
        self.assertIn("Frozen MA20", row.target_rule)
        self.assertIn("Paper only", row.risk_rule)

    def test_data_quality_manifest_blocks_runtime_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report_dir = base / "reports"
            lab_dir = base / "lab"
            report_dir.mkdir()
            error_dir = lab_dir / "TEST_MR" / "trades"
            error_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"asset": "BTC", "total_trades": 5, "net_pnl": 12.5, "rating": "INSUFFICIENT_SAMPLE"}]
            ).to_csv(report_dir / "strategy_lab_TEST_MR.csv", index=False)
            pd.DataFrame([{"asset": "BTC", "error": "STALE_FEED"}]).to_csv(error_dir / "runtime_errors.csv", index=False)
            pd.DataFrame([{"asset": "BTC", "passed": False}]).to_csv(error_dir / "evaluations.csv", index=False)

            manifest = build_data_quality_manifest([DummySpec()], report_dir=report_dir, lab_dir=lab_dir)

            self.assertEqual(manifest.iloc[0]["quality_status"], "CHECK")
            self.assertFalse(bool(manifest.iloc[0]["trusted_for_promotion"]))

    def test_write_strategy_architecture_manifests_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report_dir = base / "reports"
            lab_dir = base / "lab"

            paths = write_strategy_architecture_manifests(
                [DummySpec()],
                assets={"BTC": "BTCUSDT"},
                report_dir=report_dir,
                lab_dir=lab_dir,
                now=datetime(2026, 7, 25, tzinfo=timezone.utc),
                workflow_name="unit-test",
            )

            for path in paths.values():
                self.assertTrue(path.exists(), str(path))
            manifest_text = (report_dir / "strategy_run_manifest.json").read_text(encoding="utf-8")
            self.assertIn("unit-test", manifest_text)
            self.assertIn("paper_only", manifest_text)


if __name__ == "__main__":
    unittest.main()
