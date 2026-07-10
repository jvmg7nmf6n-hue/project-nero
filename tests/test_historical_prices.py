from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from nero_app.core.historical_prices import (
    fetch_and_write_standard_histories,
    fetch_binance_daily_history,
    fetch_twelve_data_daily_history,
    write_price_history,
)


def _binance_candle(open_time: int, close: float) -> list[object]:
    return [open_time, "100", "110", "90", str(close), "123", open_time + 86399999, "0", 1, "0", "0", "0"]


class HistoricalPricesTest(unittest.TestCase):
    def test_fetch_binance_daily_history_parses_candles(self) -> None:
        response = Mock()
        response.json.return_value = [_binance_candle(1609459200000, 101), _binance_candle(1609545600000, 102)]
        response.raise_for_status.return_value = None

        with patch("nero_app.core.historical_prices.requests.get", return_value=response):
            frame = fetch_binance_daily_history("BTCUSDT", start="2021-01-01", end="2021-01-03")

        self.assertEqual(len(frame), 2)
        self.assertEqual(float(frame.iloc[-1]["close"]), 102.0)
        self.assertIn("date", frame.columns)

    def test_fetch_twelve_data_daily_history_requires_key_and_parses_values(self) -> None:
        response = Mock()
        response.json.return_value = {
            "values": [
                {"datetime": "2021-01-02", "open": "1900", "high": "1910", "low": "1880", "close": "1905"},
                {"datetime": "2021-01-01", "open": "1880", "high": "1900", "low": "1870", "close": "1890"},
            ]
        }
        response.raise_for_status.return_value = None

        with patch("nero_app.core.historical_prices.requests.get", return_value=response):
            frame = fetch_twelve_data_daily_history("XAU/USD", start="2021-01-01", api_key="key")

        self.assertEqual(len(frame), 2)
        self.assertEqual(float(frame.iloc[0]["close"]), 1890.0)
        self.assertEqual(float(frame.iloc[-1]["close"]), 1905.0)

    def test_write_price_history_dedupes_and_sorts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "btc.csv"
            frame = pd.DataFrame(
                [
                    {"date": "2021-01-02", "open": 1, "high": 1, "low": 1, "close": 2, "volume": 1},
                    {"date": "2021-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                    {"date": "2021-01-02", "open": 1, "high": 1, "low": 1, "close": 3, "volume": 1},
                ]
            )

            write_price_history(frame, path)
            saved = pd.read_csv(path)

        self.assertEqual(len(saved), 2)
        self.assertEqual(float(saved.iloc[-1]["close"]), 3.0)

    def test_fetch_and_write_standard_histories_records_partial_errors(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            btc = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=2), "open": [1, 2], "high": [2, 3], "low": [1, 2], "close": [1, 2], "volume": [10, 20]})
            with patch("nero_app.core.historical_prices.fetch_binance_daily_history", return_value=btc), patch(
                "nero_app.core.historical_prices.fetch_twelve_data_daily_history", side_effect=ValueError("missing")
            ):
                results = fetch_and_write_standard_histories(output_dir=output_dir)

            btc_result = next(result for result in results if result.asset == "BTC")
            gold_result = next(result for result in results if result.asset == "GOLD")
            self.assertEqual(btc_result.status, "ok")
            self.assertTrue(btc_result.path.exists())
            self.assertIn("error", gold_result.status)


if __name__ == "__main__":
    unittest.main()

