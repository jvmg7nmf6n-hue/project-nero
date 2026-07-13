from __future__ import annotations

import unittest

import pandas as pd

from nero_app.core.etf_flow_intelligence import compute_actual_etf_flow_score, compute_etf_flow_score
from nero_app.core.gold_real_yield import compute_gold_real_yield_score
from nero_app.core.trade_opportunity_scanner import (
    PaperTradeState,
    ScannerInputs,
    TechnicalSnapshot,
    scan_trade_opportunity,
)
from nero_app.core.trade_readiness import ReadinessInputs, build_trade_readiness_report


def _df(closes, volumes=None):
    index = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    data = {"Close": closes}
    if volumes is not None:
        data["Volume"] = volumes
    return pd.DataFrame(data, index=index)


def _series(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="D"))


class NewIntelligenceModulesTest(unittest.TestCase):
    def test_etf_flow_marks_inflow_proxy(self) -> None:
        closes = [50.0] * 25 + [51.0, 52.0, 53.0, 54.0, 56.0]
        volumes = [1_000_000.0] * 29 + [5_000_000.0]
        btc = _df([60000.0 + i * 100 for i in range(30)])

        report = compute_etf_flow_score({"IBIT": _df(closes, volumes)}, btc, etf_tickers=["IBIT"])

        self.assertGreaterEqual(report.etf_flow_score, 60.0)
        self.assertIn(report.etf_flow_label, {"MODERATE_INFLOW_PRESSURE", "STRONG_INFLOW_PRESSURE"})
        self.assertEqual(report.dominant_etf, "IBIT")


    def test_actual_etf_flow_scores_real_inflows(self) -> None:
        flows = pd.DataFrame(
            [
                {"date": "2026-07-10", "ticker": "IBIT", "flow_musd": 350},
                {"date": "2026-07-10", "ticker": "FBTC", "flow_musd": 150},
                {"date": "2026-07-10", "ticker": "GBTC", "flow_musd": -50},
            ]
        )

        report = compute_actual_etf_flow_score(flows, lookback_days=1)

        self.assertFalse(report.is_proxy)
        self.assertEqual(report.etf_flow_label, "MODERATE_INFLOW_PRESSURE")
        self.assertEqual(report.dominant_etf, "IBIT")
        self.assertGreater(report.etf_flow_score, 60)
        self.assertIn("actual", report.evidence_frame()["data_type"].unique())

    def test_etf_flow_handles_missing_data(self) -> None:
        report = compute_etf_flow_score({}, None, etf_tickers=["IBIT"])

        self.assertEqual(report.etf_flow_label, "DATA_INSUFFICIENT")
        self.assertEqual(report.etf_flow_score, 0.0)

    def test_gold_real_yield_supportive_when_real_yield_negative(self) -> None:
        nominal = _series([1.0] * 30)
        breakeven = _series([3.0] * 30)

        report = compute_gold_real_yield_score(nominal_yield_series=nominal, breakeven_inflation_series=breakeven)

        self.assertEqual(report.real_yield_label, "GOLD_MACRO_SUPPORTIVE")
        self.assertLess(report.estimated_real_yield, 0)

    def test_gold_real_yield_pressure_when_real_yield_high(self) -> None:
        nominal = _series([5.0] * 30)
        breakeven = _series([1.0] * 30)

        report = compute_gold_real_yield_score(nominal_yield_series=nominal, breakeven_inflation_series=breakeven)

        self.assertEqual(report.real_yield_label, "GOLD_MACRO_PRESSURE")
        self.assertLess(report.real_yield_score, 40.0)

    def test_trade_scanner_allows_strong_aligned_paper_setup(self) -> None:
        result = scan_trade_opportunity(
            ScannerInputs(
                asset="BTC",
                quant_consensus_score=85,
                sentiment_score=80,
                etf_flow_score=80,
                sharpe_90d=1.2,
                technical=TechnicalSnapshot(
                    trend="UP",
                    rsi=55,
                    zscore=2.1,
                    volatility_regime="NORMAL",
                    price_above_ma20=True,
                    price_above_ma200=True,
                ),
            )
        )

        self.assertEqual(result.decision, "TRADE_ALLOWED")
        self.assertEqual(result.direction_bias, "LONG_BIAS")

    def test_trade_scanner_blocks_duplicate_trade(self) -> None:
        result = scan_trade_opportunity(
            ScannerInputs(
                asset="GOLD",
                quant_consensus_score=90,
                paper_trade_state=PaperTradeState(has_pending_order=True, asset="GOLD"),
            )
        )

        self.assertEqual(result.decision, "BLOCKED_BY_RISK")
        self.assertTrue(result.blocker_reason)

    def test_trade_readiness_allows_only_aligned_paper_setup(self) -> None:
        report = build_trade_readiness_report(
            ReadinessInputs(
                asset="BTC",
                opportunity_decision="TRADE_ALLOWED",
                opportunity_score=82,
                quant_score=72,
                volatility_regime="VOL_NORMAL",
                sentiment_score=70,
            )
        )

        self.assertEqual(report.label, "TRADE_READY")
        self.assertGreaterEqual(report.readiness_score, 75)

    def test_trade_readiness_blocks_active_paper_trade(self) -> None:
        report = build_trade_readiness_report(
            ReadinessInputs(
                asset="BTC",
                opportunity_decision="TRADE_ALLOWED",
                opportunity_score=90,
                quant_score=80,
                volatility_regime="VOL_NORMAL",
                sentiment_score=80,
                has_active_paper_trade=True,
            )
        )

        self.assertEqual(report.label, "NO_TRADE_RISK")
        self.assertIn("active paper trade already exists", report.blockers)


if __name__ == "__main__":
    unittest.main()
