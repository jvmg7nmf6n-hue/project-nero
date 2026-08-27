from __future__ import annotations

from datetime import datetime, timedelta
import unittest

import pandas as pd

from nero_app.core.cycle_intelligence import (
    GOLD_PROXY_NOTE,
    build_cycle_intelligence_report,
    build_cycle_intelligence_row,
)


def _daily_frame(count: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    begin = datetime(2025, 1, 1)
    closes = [start + index * step for index in range(count)]
    return pd.DataFrame(
        {
            "date": [begin + timedelta(days=index) for index in range(count)],
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1000.0] * count,
        }
    )


class CycleIntelligenceTests(unittest.TestCase):
    def test_mayer_multiple_requires_200_clean_daily_closes(self) -> None:
        row, _ = build_cycle_intelligence_row("BTC", _daily_frame(199), "test", "provided")

        self.assertEqual(row["status"], "UNAVAILABLE")
        self.assertIn("needs at least 200", str(row["unavailable_reason"]))
        self.assertIsNone(row["mayer_multiple"])

    def test_builds_price_only_cycle_fields_from_daily_closes(self) -> None:
        row, _ = build_cycle_intelligence_row("BTC", _daily_frame(240), "test", "provided")

        self.assertEqual(row["status"], "OK")
        self.assertEqual(row["total_daily_closes"], 240)
        self.assertAlmostEqual(float(row["latest_close"]), 339.0)
        self.assertGreater(float(row["mayer_multiple"]), 1.0)
        self.assertEqual(row["sma200_slope_label"], "SMA200_RISING")
        self.assertEqual(row["consumer"], "dashboard Cycle Intelligence tab; reports/cycle_intelligence_report.*")

    def test_gold_maps_to_paxg_and_keeps_proxy_caveat(self) -> None:
        row, _ = build_cycle_intelligence_row("GOLD", _daily_frame(240), "test", "provided")

        self.assertEqual(row["cycle_asset"], "PAXG")
        self.assertEqual(row["symbol"], "PAXGUSDT")
        self.assertEqual(row["paxg_caveat"], GOLD_PROXY_NOTE)

    def test_missing_data_is_unavailable_not_zero(self) -> None:
        row, _ = build_cycle_intelligence_row("SOL", pd.DataFrame(), "test", "provided")

        self.assertEqual(row["status"], "UNAVAILABLE")
        self.assertIsNone(row["latest_close"])
        self.assertIsNone(row["sma_200"])
        self.assertIsNone(row["drawdown_from_high_pct"])

    def test_fallback_source_is_not_used_for_cycle_fields(self) -> None:
        row, _ = build_cycle_intelligence_row("BTC", _daily_frame(240), "Generated sample candles", "fallback")

        self.assertEqual(row["status"], "UNAVAILABLE")
        self.assertIn("fallback/sample", str(row["unavailable_reason"]))
        self.assertIsNone(row["mayer_multiple"])

    def test_report_includes_availability_and_redundancy_check(self) -> None:
        report = build_cycle_intelligence_report(
            assets=["BTC"],
            price_frames={"BTC": _daily_frame(260)},
            prefer_live=False,
        )

        self.assertEqual(len(report.rows), 1)
        self.assertTrue(report.availability_rows)
        self.assertTrue(report.correlation_rows)
        self.assertFalse(any("cycle_intelligence_score" in row for row in report.rows))


if __name__ == "__main__":
    unittest.main()
