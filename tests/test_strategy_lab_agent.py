from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.strategy_lab_agent import CANDIDATES, STRATEGY_LAB_DEFAULT_ASSETS, SignalValidator, write_strategy_lab_summary
from tools.nero_strategy_lab_weekly_report import build_strategy_lab_weekly_report


class StrategyLabAgentTest(unittest.TestCase):
    def test_has_old_and_new_lab_candidates(self) -> None:
        self.assertGreaterEqual(len(CANDIDATES), 18)
        self.assertIn("MR_RELAXED_PULLBACK_V1", CANDIDATES)
        self.assertIn("BREAKOUT_MOMENTUM_V1", CANDIDATES)
        self.assertEqual(CANDIDATES["MR_RELAXED_PULLBACK_V1"].display_label, "OLD_MR_RELAXED")
        self.assertEqual(CANDIDATES["NEW_BTC_12H_MR"].bucket, "NEW_TEST")
        self.assertEqual(CANDIDATES["NEW_BTC_12H_MR"].interval, "12h")
        self.assertFalse(CANDIDATES["NEW_BTC_ETH_12H_PAIR"].enabled)
        self.assertEqual(CANDIDATES["V2_BREAKOUT_RETEST"].bucket, "V2_SHADOW")
        self.assertEqual(CANDIDATES["V2_MR_RECOVERY"].display_label, "V2_MR_RECOVERY")
        self.assertTrue(CANDIDATES["V2_MR_REWARD"].require_rsi_recovery)
        self.assertEqual(CANDIDATES["HYP_OIL_TREND_V1"].bucket, "HYPOTHESIS_TEST")
        self.assertEqual(CANDIDATES["HYP_OIL_TREND_V1"].asset_filter, ("OIL_FUT", "BRENT_FUT"))
        self.assertEqual(CANDIDATES["HYP_OIL_MR_V1"].display_label, "HYP_OIL_MR")



    def test_strategy_lab_default_assets_include_stocks_and_currencies(self) -> None:
        self.assertIn("GOLD", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("SILVER", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("OIL", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("GOLD_FUT", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("COPPER_FUT", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("BRENT_FUT", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("SPY", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("QQQ", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("MSTR", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("DXY", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("EURUSD", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertIn("USDJPY", STRATEGY_LAB_DEFAULT_ASSETS)
        self.assertEqual(len(STRATEGY_LAB_DEFAULT_ASSETS), 33)


    def test_hypothesis_assets_bypass_default_quarantine(self) -> None:
        from nero_app.core.strategy_lab_agent import _candidate_assets

        assets = {"ETH": "ETHUSDT", "OIL_FUT": "CL=F", "BRENT_FUT": "BZ=F"}
        broad = _candidate_assets(CANDIDATES["MR_RELAXED_PULLBACK_V1"], assets)
        oil_hypothesis = _candidate_assets(CANDIDATES["HYP_OIL_TREND_V1"], assets)

        self.assertNotIn("ETH", broad)
        self.assertEqual(set(oil_hypothesis), {"OIL_FUT", "BRENT_FUT"})

    def test_signal_validator_keeps_family_rules_separate(self) -> None:
        mr_spec = CANDIDATES["V2_MR_RECOVERY"]
        momentum_spec = CANDIDATES["V2_BREAKOUT_RETEST"]
        candle = pd.Series(
            {
                "close": 95.0,
                "low": 94.0,
                "high": 99.0,
                "atr": 2.0,
                "rsi": 34.0,
                "rsi_prev": 33.0,
                "close_prev": 94.0,
                "ma20": 100.0,
                "ma200": 90.0,
                "bb_lower": 96.0,
                "breakout_high": 110.0,
            }
        )

        mr_reasons, _ = SignalValidator(mr_spec).validate(candle, {}, -3.0)
        momentum_reasons, _ = SignalValidator(momentum_spec).validate(candle, {}, -3.0)

        self.assertNotIn("CLOSE_NOT_ABOVE_BREAKOUT_HIGH", mr_reasons)
        self.assertIn("CLOSE_NOT_ABOVE_BREAKOUT_HIGH", momentum_reasons)
        self.assertNotIn("CLOSE_NOT_NEAR_OR_BELOW_LOWER_BB", momentum_reasons)

    def test_signal_validator_reward_gate_uses_spec_target(self) -> None:
        spec = CANDIDATES["V2_MR_REWARD"]
        candle = pd.Series(
            {
                "close": 95.0,
                "low": 94.0,
                "high": 99.0,
                "atr": 2.0,
                "rsi": 34.0,
                "rsi_prev": 33.0,
                "close_prev": 94.0,
                "ma20": 100.0,
                "ma200": 90.0,
                "bb_lower": 96.0,
                "breakout_high": 110.0,
            }
        )

        reasons, planned_reward_r = SignalValidator(spec).validate(candle, {}, -3.0)

        self.assertGreaterEqual(planned_reward_r, 1.2)
        self.assertNotIn("PLANNED_REWARD_TOO_LOW", reasons)

    def test_summary_rates_candidate_after_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            candidate = CANDIDATES["MR_RELAXED_PULLBACK_V1"]
            pd.DataFrame(
                [
                    {
                        "asset": "COMBINED",
                        "total_trades": 32,
                        "win_rate": 0.56,
                        "expectancy_r": 0.24,
                        "profit_factor": 1.8,
                        "max_drawdown": -0.04,
                        "net_pnl": 740.0,
                        "rating": "KEEP_TESTING",
                        "rating_score": 72.0,
                        "insufficient_sample": False,
                    }
                ]
            ).to_csv(report_dir / f"strategy_lab_{candidate.candidate_id}.csv", index=False)

            summary = write_strategy_lab_summary(report_dir, [candidate])

        self.assertEqual(summary.iloc[0]["candidate_id"], candidate.candidate_id)
        self.assertEqual(summary.iloc[0]["display_label"], "OLD_MR_RELAXED")
        self.assertEqual(summary.iloc[0]["bucket"], "OLD_TEST")
        self.assertEqual(summary.iloc[0]["interval"], "1h")
        self.assertEqual(int(summary.iloc[0]["total_trades"]), 32)
        self.assertEqual(summary.iloc[0]["rating"], "KEEP_TESTING")

    def test_weekly_report_mentions_sample_warning(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": "MR_RELAXED_PULLBACK_V1",
                    "total_trades": 3,
                    "win_rate": 0.33,
                    "expectancy_r": -0.1,
                    "profit_factor": 0.8,
                    "max_drawdown": -0.02,
                    "rating": "INSUFFICIENT_SAMPLE",
                    "rating_score": 41,
                }
            ]
        )
        report = build_strategy_lab_weekly_report(summary)
        self.assertIn("Sample warning", report)
        self.assertIn("MR_RELAXED_PULLBACK_V1", report)


if __name__ == "__main__":
    unittest.main()
