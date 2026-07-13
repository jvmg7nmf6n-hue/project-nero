from __future__ import annotations

import unittest

from nero_app.core.trade_path import TradePathInput, build_trade_path_report


class TradePathTest(unittest.TestCase):
    def test_trade_path_explains_weak_btc_setup(self) -> None:
        report = build_trade_path_report(
            TradePathInput(
                asset="BTC",
                readiness_label="NO_TRADE_RISK",
                readiness_score=0,
                opportunity_decision="BLOCKED_BY_RISK",
                opportunity_score=24,
                direction_bias="NEUTRAL",
                quant_score=36,
                external_score=24,
                external_label="OUTFLOW_PRESSURE",
                sentiment_score=40,
                volatility_regime="VOL_NORMAL",
                failed_conditions=["ETF proxy inflow pressure weak/unsupportive (24)"],
            )
        )

        self.assertEqual(report.path_label, "NO_TRADE_PATH")
        self.assertTrue(any("Quant consensus is weak" in item for item in report.missing_confirmations))
        self.assertTrue(any("ETF flow" in item for item in report.watch_triggers))

    def test_trade_path_blocks_duplicate_trade(self) -> None:
        report = build_trade_path_report(
            TradePathInput(
                asset="BTC",
                readiness_label="NO_TRADE_RISK",
                readiness_score=5,
                opportunity_decision="BLOCKED_BY_RISK",
                opportunity_score=0,
                direction_bias="NEUTRAL",
                blockers=["active paper trade already exists"],
                has_active_paper_trade=True,
            )
        )

        self.assertEqual(report.path_label, "WAIT_EXISTING_TRADE")
        self.assertIn("let that trade resolve", report.action)


if __name__ == "__main__":
    unittest.main()
