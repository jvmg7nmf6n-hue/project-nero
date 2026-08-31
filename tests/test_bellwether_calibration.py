from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nero_app.core.bellwether_calibration import (
    SCHEMA_VERSION,
    append_calibration_forecast,
    build_calibration_report,
    intended_resolution_at,
    load_calibration_ledger,
    mark_legacy_predictions,
    resolve_calibration_ledger,
    source_id_for,
    validate_price_frame,
)
from nero_app.core.market_data import MarketDataResult
from nero_app.core.schema import AnalysisRequest, AssessmentOutput, AssetSymbol, BrainOutput, NeroResult, VerdictOutput


def _result(asset: AssetSymbol = AssetSymbol.BTC, direction: str = "bullish", confidence: float = 0.62) -> NeroResult:
    return NeroResult(
        request=AnalysisRequest(asset=asset, headline="Liquidity improves for digital assets.", lookback_days=30),
        brain=BrainOutput(matches=[], thematic_score=0.2, dominant_tags=["liquidity"]),
        assessment=AssessmentOutput(
            rsi=55,
            trend_score=0.2,
            fair_value_gap="none",
            liquidity_sweep="none",
            momentum_score=0.1,
            confluence_score=57,
            market_regime="Bull",
            volatility_regime="Normal-Vol",
            technical_bias_score=0.1,
        ),
        verdict=VerdictOutput(direction=direction, confidence=confidence, risk_score=0.4, summary="test", drivers=[]),
    )


def _prices(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date) for date, _ in rows],
            "open": [price for _, price in rows],
            "high": [price * 1.01 for _, price in rows],
            "low": [price * 0.99 for _, price in rows],
            "close": [price for _, price in rows],
            "volume": [100.0 for _ in rows],
        }
    )


class _MarketClient:
    def __init__(self, data: MarketDataResult) -> None:
        self.data = data

    def load(self, **kwargs) -> MarketDataResult:
        return self.data


class BellwetherCalibrationTest(unittest.TestCase):
    def test_source_id_is_stable_for_live_provider_label(self) -> None:
        source_id = source_id_for("BTC", "Binance BTCUSDT daily candles", "live")

        self.assertEqual(source_id, "binance:BTCUSDT:daily")

    def test_append_is_idempotent_and_marks_confidence_not_probability(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.csv"
            market = MarketDataResult(
                prices=_prices([("2026-01-01", 100.0)]),
                source="Binance BTCUSDT daily candles",
                status="live",
            )
            issued = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

            first = append_calibration_forecast(_result(), market, ledger_path=ledger, issued_at=issued)
            second = append_calibration_forecast(_result(), market, ledger_path=ledger, issued_at=issued)
            frame = load_calibration_ledger(ledger)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(frame.iloc[0]["probability_status"], "NOT_A_PROBABILITY")
        self.assertEqual(frame.iloc[0]["status"], "PENDING")

    def test_resolver_records_outcome_but_refuses_brier_for_non_probability(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.csv"
            report_csv = Path(directory) / "report.csv"
            report_json = Path(directory) / "report.json"
            issue_market = MarketDataResult(
                prices=_prices([("2026-01-01", 100.0)]),
                source="Binance BTCUSDT daily candles",
                status="live",
            )
            issued = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
            append_calibration_forecast(_result(), issue_market, ledger_path=ledger, issued_at=issued)
            resolution_market = MarketDataResult(
                prices=_prices([("2026-01-01", 100.0), ("2026-01-02", 102.0)]),
                source="Binance BTCUSDT daily candles",
                status="live",
            )

            summary = resolve_calibration_ledger(
                ledger_path=ledger,
                report_csv=report_csv,
                report_json=report_json,
                market_client=_MarketClient(resolution_market),
                now=datetime(2026, 1, 2, 13, tzinfo=timezone.utc),
            )
            frame = load_calibration_ledger(ledger)

        self.assertEqual(summary.resolved, 1)
        self.assertEqual(frame.iloc[0]["status"], "NOT_A_PROBABILITY")
        self.assertEqual(frame.iloc[0]["reason_code"], "CONFIDENCE_FIELD_NOT_PROBABILITY")
        self.assertEqual(frame.iloc[0]["outcome_no_deadband"], "up")
        self.assertTrue(pd.isna(frame.iloc[0]["brier_error"]) or str(frame.iloc[0]["brier_error"]) == "")

    def test_resolver_blocks_source_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.csv"
            issue_market = MarketDataResult(
                prices=_prices([("2026-01-01", 100.0)]),
                source="Binance BTCUSDT daily candles",
                status="live",
            )
            append_calibration_forecast(_result(), issue_market, ledger_path=ledger, issued_at=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
            resolution_market = MarketDataResult(
                prices=_prices([("2026-01-02", 102.0)]),
                source="Coinbase BTC-USD daily candles",
                status="live",
            )

            resolve_calibration_ledger(
                ledger_path=ledger,
                report_csv=Path(directory) / "report.csv",
                report_json=Path(directory) / "report.json",
                market_client=_MarketClient(resolution_market),
                now=datetime(2026, 1, 2, 13, tzinfo=timezone.utc),
            )
            frame = load_calibration_ledger(ledger)

        self.assertEqual(frame.iloc[0]["status"], "SOURCE_MISMATCH")
        self.assertEqual(frame.iloc[0]["reason_code"], "ISSUE_RESOLUTION_SOURCE_MISMATCH")

    def test_gold_resolution_skips_weekend(self) -> None:
        issued = datetime(2026, 1, 2, 18, tzinfo=timezone.utc)

        intended = intended_resolution_at("GOLD", issued)

        self.assertEqual(intended.weekday(), 0)
        self.assertEqual(intended.hour, 17)

    def test_price_validation_rejects_fallback_and_future_data(self) -> None:
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

        fallback = validate_price_frame(_prices([("2026-01-01", 100.0)]), "Generated sample candles", "sample", now=now)
        future = validate_price_frame(_prices([("2026-01-02", 100.0)]), "Binance BTCUSDT daily candles", "live", now=now)

        self.assertEqual(fallback, "NON_LIVE_SOURCE")
        self.assertEqual(future, "FUTURE_TIMESTAMP")

    def test_legacy_predictions_are_explicitly_unscored(self) -> None:
        legacy = pd.DataFrame([{"asset": "BTC", "confidence": 0.5}])

        marked = mark_legacy_predictions(legacy)

        self.assertEqual(marked.iloc[0]["calibration_status"], "LEGACY_UNSCORED")

    def test_report_shows_insufficient_probability_sample(self) -> None:
        ledger = pd.DataFrame(
            [
                {"asset": "BTC", "status": "NOT_A_PROBABILITY", "probability_status": "NOT_A_PROBABILITY"},
                {"asset": "GOLD", "status": "PENDING", "probability_status": "NOT_A_PROBABILITY"},
            ]
        )

        report, summary = build_calibration_report(ledger, legacy_prediction_log=pd.DataFrame([{"asset": "BTC"}]))

        self.assertEqual(summary["headline_status"], "INSUFFICIENT_PROBABILITY_DATA")
        self.assertEqual(summary["legacy_prediction_rows"], 1)
        self.assertIn("calibration_status", report.columns)


if __name__ == "__main__":
    unittest.main()
