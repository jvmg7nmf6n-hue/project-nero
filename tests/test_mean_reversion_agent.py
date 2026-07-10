from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nero_app.core.mean_reversion_agent import MeanReversionAgent, MeanReversionConfig, report_row


class MeanReversionAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MeanReversionConfig(assets={"BTC": "BTCUSDT"}, initial_equity=10000.0)
        self.now = datetime(2026, 7, 10, 12, tzinfo=timezone.utc)

    def candle(self, close_time: int = 3600000, **overrides):
        data = {
            "date": pd.Timestamp("2026-07-10T01:00:00Z"),
            "open_time": close_time - 3600000,
            "close_time": close_time,
            "open": 101.0,
            "high": 102.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "rsi": 30.0,
            "ma20": 105.0,
            "bb_lower": 101.0,
            "ma200": 95.0,
            "atr": 2.0,
        }
        data.update(overrides)
        return pd.Series(data)

    def test_entry_conditions_pass_and_size_is_capped_by_notional(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MeanReversionAgent(config=self.config, data_dir=Path(directory) / "data", report_dir=Path(directory) / "reports", now=self.now)
            state = {"equity": 10000.0, "daily_r": 0.0, "open_trade": None}
            candle = self.candle()

            evaluation = agent._evaluate_entry("BTC", "BTCUSDT", candle, state)
            trade = agent._enter_trade("BTC", "BTCUSDT", candle, state)

        self.assertTrue(evaluation["passed"])
        self.assertIsNotNone(trade)
        self.assertEqual(trade["target_mode"], "FROZEN_MA20")
        self.assertLessEqual(trade["notional"], 10000.0)
        self.assertGreater(trade["risk_dollars"], 0.0)

    def test_rejects_when_daily_loss_guard_is_hit(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MeanReversionAgent(config=self.config, data_dir=Path(directory) / "data", report_dir=Path(directory) / "reports", now=self.now)
            state = {"equity": 10000.0, "daily_r": -3.0, "open_trade": None}

            evaluation = agent._evaluate_entry("BTC", "BTCUSDT", self.candle(), state)

        self.assertFalse(evaluation["passed"])
        self.assertIn("DAILY_LOSS_GUARD", evaluation["rejection_reasons"])

    def test_exit_uses_stop_first_when_stop_and_target_hit_same_candle(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MeanReversionAgent(config=self.config, data_dir=Path(directory) / "data", report_dir=Path(directory) / "reports", now=self.now)
            state = {"equity": 10000.0, "daily_r": 0.0, "open_trade": None}
            entry = agent._enter_trade("BTC", "BTCUSDT", self.candle(close_time=3600000), state)
            self.assertIsNotNone(entry)

            exit_event = agent._maybe_exit(
                "BTC",
                "BTCUSDT",
                self.candle(close_time=7200000, high=110.0, low=90.0, close=104.0),
                state,
            )

        self.assertIsNotNone(exit_event)
        self.assertEqual(exit_event["exit_reason"], "SL")
        self.assertLess(exit_event["net_pnl"], 0.0)
        self.assertIsNone(state["open_trade"])

    def test_time_exit_after_24_hours(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MeanReversionAgent(config=self.config, data_dir=Path(directory) / "data", report_dir=Path(directory) / "reports", now=self.now)
            state = {"equity": 10000.0, "daily_r": 0.0, "open_trade": None}
            agent._enter_trade("BTC", "BTCUSDT", self.candle(close_time=3600000), state)

            exit_event = agent._maybe_exit(
                "BTC",
                "BTCUSDT",
                self.candle(close_time=3600000 + 24 * 3600000, high=103.0, low=98.0, close=101.0),
                state,
            )

        self.assertIsNotNone(exit_event)
        self.assertEqual(exit_event["exit_reason"], "TIME")

    def test_report_marks_insufficient_sample_under_20_trades(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "asset": "BTC",
                    "net_pnl": 50.0,
                    "r_multiple": 0.5,
                    "equity_after": 10050.0,
                    "fees": 20.0,
                    "notional": 10000.0,
                    "slippage_bps": 2.0,
                    "holding_hours": 3.0,
                }
            ]
        )

        row = report_row("BTC", trades, pd.DataFrame())

        self.assertTrue(row["insufficient_sample"])
        self.assertEqual(row["total_trades"], 1)
        self.assertGreater(row["profit_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
