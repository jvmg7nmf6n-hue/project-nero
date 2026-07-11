from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.quant_intelligence import (
    build_quant_snapshot,
    information_coefficient,
    log_returns,
    quant_driver_rows,
    rolling_beta,
    rolling_correlation,
    zscore,
)


class QuantIntelligenceTest(unittest.TestCase):
    def test_log_returns_uses_price_ratio(self) -> None:
        returns = log_returns(pd.Series([100.0, 110.0, 121.0]))

        self.assertAlmostEqual(float(returns.iloc[0]), 0.0953101798, places=6)
        self.assertAlmostEqual(float(returns.iloc[1]), 0.0953101798, places=6)

    def test_correlation_and_beta_detect_relationship(self) -> None:
        frame = pd.DataFrame(
            {
                "asset": [0.01, 0.02, 0.03, 0.04, 0.05],
                "driver": [0.005, 0.010, 0.015, 0.020, 0.025],
            }
        )

        corr = rolling_correlation(frame, "asset", "driver", window=5).iloc[-1]
        beta = rolling_beta(frame, "asset", "driver", window=5).iloc[-1]

        self.assertAlmostEqual(float(corr), 1.0, places=6)
        self.assertAlmostEqual(float(beta), 2.0, places=6)

    def test_zscore_flags_stretch(self) -> None:
        series = pd.Series([100.0] * 20 + [110.0])

        latest = zscore(series, window=20).iloc[-1]

        self.assertGreater(float(latest), 2.0)

    def test_quant_snapshot_returns_regime_and_rows(self) -> None:
        dates = pd.date_range("2026-01-01", periods=120, freq="D")
        closes = [100.0 + index * 0.5 for index in range(120)]
        prices = pd.DataFrame({"date": dates, "close": closes})

        snapshot = build_quant_snapshot(prices, asset="BTC", source="test")
        rows = quant_driver_rows(snapshot)

        self.assertEqual(snapshot.asset, "BTC")
        self.assertEqual(snapshot.observation_count, 120)
        self.assertIn("/", snapshot.regime)
        self.assertGreaterEqual(len(rows), 8)

    def test_information_coefficient_uses_rank_correlation(self) -> None:
        ic = information_coefficient(pd.Series([1, 2, 3, 4]), pd.Series([2, 4, 6, 8]))

        self.assertAlmostEqual(ic, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
