import unittest

import pandas as pd

from nero_app.core.range_mean_reversion import (
    RangeMRConfig,
    _entry_rejection_reasons,
    _maybe_exit,
    _open_trade,
    average_directional_index,
    range_mr_hypothesis_configs,
    run_range_mean_reversion_backtest,
)


class RangeMeanReversionTests(unittest.TestCase):
    def test_regime_gate_allows_range_and_blocks_trend(self):
        cfg = RangeMRConfig()
        range_candle = pd.Series({"adx": 18.0, "close": 95.0, "bb_lower": 96.0, "bb_upper": 104.0, "atr": 2.0})
        trend_candle = pd.Series({"adx": 31.0, "close": 95.0, "bb_lower": 96.0, "bb_upper": 104.0, "atr": 2.0})
        self.assertEqual(_entry_rejection_reasons(range_candle, None, cfg), [])
        self.assertIn("ADX_NOT_RANGE", _entry_rejection_reasons(trend_candle, None, cfg))

    def test_short_exit_uses_short_accounting(self):
        cfg = RangeMRConfig(slippage_bps=0.0, fee_bps_crypto=0.0)
        entry = pd.Series({"date": "2026-01-01", "close": 110.0, "atr": 5.0, "adx": 18.0, "bb_width_pct": 4.0})
        trade = _open_trade(entry, 0, "SHORT", cfg)
        exit_candle = pd.Series({"date": "2026-01-02", "high": 111.0, "low": 99.0, "close": 100.0, "ma20": 101.0, "adx": 18.0})
        result = _maybe_exit(trade, exit_candle, 1, "BTC", "1h", 10000.0, cfg, 0)
        self.assertTrue(result["closed"])
        self.assertEqual(result["trade"].exit_reason, "TARGET")
        self.assertGreater(result["trade"].net_pnl, 0)

    def test_long_disaster_stop(self):
        cfg = RangeMRConfig(slippage_bps=0.0, fee_bps_crypto=0.0)
        entry = pd.Series({"date": "2026-01-01", "close": 100.0, "atr": 4.0, "adx": 18.0, "bb_width_pct": 4.0})
        trade = _open_trade(entry, 0, "LONG", cfg)
        exit_candle = pd.Series({"date": "2026-01-02", "high": 100.0, "low": 91.5, "close": 92.0, "ma20": 105.0, "adx": 18.0})
        result = _maybe_exit(trade, exit_candle, 1, "BTC", "1h", 10000.0, cfg, 0)
        self.assertTrue(result["closed"])
        self.assertEqual(result["trade"].exit_reason, "SL")
        self.assertLess(result["trade"].r_multiple, 0)

    def test_adx_hysteresis_requires_two_exit_bars(self):
        cfg = RangeMRConfig(slippage_bps=0.0, fee_bps_crypto=0.0)
        entry = pd.Series({"date": "2026-01-01", "close": 100.0, "atr": 4.0, "adx": 18.0, "bb_width_pct": 4.0})
        trade = _open_trade(entry, 0, "LONG", cfg)
        first = pd.Series({"date": "2026-01-02", "high": 102.0, "low": 98.0, "close": 99.0, "ma20": 105.0, "adx": 28.5})
        second = pd.Series({"date": "2026-01-03", "high": 102.0, "low": 98.0, "close": 99.0, "ma20": 105.0, "adx": 29.0})
        result1 = _maybe_exit(trade, first, 1, "BTC", "1h", 10000.0, cfg, 0)
        self.assertFalse(result1["closed"])
        result2 = _maybe_exit(trade, second, 2, "BTC", "1h", 10000.0, cfg, result1["adx_break_count"])
        self.assertTrue(result2["closed"])
        self.assertEqual(result2["trade"].exit_reason, "REGIME_BREAK")

    def test_public_backtest_records_band_extreme_trade(self):
        cfg = RangeMRConfig(adx_period=3, atr_period=3, bollinger_period=5, slippage_bps=0.0, fee_bps_crypto=0.0)
        rows = []
        base = pd.Timestamp("2026-01-01", tz="UTC")
        prices = [100, 101, 99, 100, 101, 100, 99, 100, 98, 96, 97, 99, 101, 100]
        for i, close in enumerate(prices):
            rows.append({"date": base + pd.Timedelta(hours=i), "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100})
        frame = pd.DataFrame(rows)
        trades, evaluations = run_range_mean_reversion_backtest(frame, "BTC", "1h", cfg)
        self.assertFalse(evaluations.empty)
        self.assertIn("passed", evaluations.columns)


    def test_hypothesis_configs_are_named_and_distinct(self):
        configs = range_mr_hypothesis_configs()
        names = [cfg.hypothesis_id for cfg in configs]
        self.assertIn("RMR_RANGE_GATE_ONLY", names)
        self.assertIn("RMR_DEEP_BAND", names)
        self.assertIn("RMR_ADX_FALLING", names)
        self.assertIn("RMR_CONFIRMATION_ENTRY", names)
        self.assertIn("RMR_LONG_ONLY", names)
        self.assertEqual(len(names), len(set(names)))

    def test_deep_band_rejects_shallow_band_touch(self):
        cfg = RangeMRConfig(min_band_atr=0.5)
        candle = pd.Series({"adx": 18.0, "close": 95.9, "bb_lower": 96.0, "bb_upper": 104.0, "ma20": 100.0, "atr": 2.0, "band_distance_atr": 0.05})
        self.assertIn("BAND_EXTREME_NOT_DEEP", _entry_rejection_reasons(candle, None, cfg))

    def test_long_only_rejects_short_band_setup(self):
        cfg = RangeMRConfig(long_only=True)
        candle = pd.Series({"adx": 18.0, "close": 105.0, "bb_lower": 96.0, "bb_upper": 104.0, "ma20": 100.0, "atr": 2.0, "band_distance_atr": 0.5})
        self.assertIn("SHORT_DISABLED", _entry_rejection_reasons(candle, None, cfg))

    def test_adx_calculates_without_crashing(self):
        rows = []
        base = pd.Timestamp("2026-01-01", tz="UTC")
        for i in range(40):
            close = 100 + i
            rows.append({"date": base + pd.Timedelta(hours=i), "open": close, "high": close + 2, "low": close - 1, "close": close, "volume": 100})
        adx = average_directional_index(pd.DataFrame(rows), period=5)
        self.assertGreater(adx.dropna().iloc[-1], 50)


if __name__ == "__main__":
    unittest.main()
