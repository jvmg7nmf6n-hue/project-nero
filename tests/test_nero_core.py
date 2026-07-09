from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from nero_app.core.ai_sentiment import analyze_news_sentiment
from nero_app.core.backtester import run_event_backtest
from nero_app.core.data_loader import load_macro_events, load_price_history
from nero_app.core.demo_trader import accountability_scorecard, load_demo_trades, run_demo_trader
from nero_app.core.market_data import MarketDataClient
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert, send_ntfy_alert
from nero_app.core.news_feed import NewsFeedClient, NewsItem, _rank_for_asset
from nero_app.core.orchestrator import NeroOrchestrator
from nero_app.core.prediction_log import append_prediction, evaluate_prediction_log, load_prediction_log
from nero_app.core.schema import AnalysisRequest, AssetSymbol
from nero_app.core.settings import load_settings, save_settings
from nero_app.core.technical_analysis import analyze_technical
from nero_app.core.trade_desk import IntradayTradePlan, build_intraday_trade_plan


class NeroCoreTest(unittest.TestCase):
    def test_orchestrator_returns_structured_verdict(self) -> None:
        orchestrator = NeroOrchestrator(load_macro_events())
        result = orchestrator.run(
            AnalysisRequest(
                asset=AssetSymbol.BTC,
                headline="Fed turns dovish and liquidity improves for crypto risk assets.",
                lookback_days=30,
            ),
            load_price_history(),
        )

        self.assertGreater(len(result.brain.matches), 0)
        self.assertIn(result.verdict.direction, {"bullish", "bearish", "neutral"})
        self.assertGreaterEqual(result.verdict.confidence, 0)
        self.assertLessEqual(result.verdict.confidence, 1)
        self.assertTrue(hasattr(result.assessment, "confluence_score"))

    def test_backtester_handles_matches(self) -> None:
        orchestrator = NeroOrchestrator(load_macro_events())
        result = orchestrator.run(
            AnalysisRequest(
                asset=AssetSymbol.SPY,
                headline="Inflation is cooling and rate cut hopes lift equities.",
                lookback_days=30,
            ),
            load_price_history(),
        )
        backtest = run_event_backtest(result.brain.matches, load_price_history())

        self.assertGreaterEqual(backtest.sample_count, 1)
        self.assertGreaterEqual(backtest.win_rate, 0)
        self.assertLessEqual(backtest.win_rate, 1)


    def test_technical_analysis_returns_confluence_and_regime(self) -> None:
        technical = analyze_technical(load_price_history(), lookback_days=30)

        self.assertGreaterEqual(technical.confluence_score, 0)
        self.assertLessEqual(technical.confluence_score, 100)
        self.assertIn(technical.market_regime, {"Bull", "Bear", "Range"})
        self.assertIn(technical.volatility_regime, {"High-Vol", "Normal-Vol", "Low-Vol"})
        self.assertIn(technical.macd_signal, {"bullish", "bearish", "neutral"})


    def test_market_data_sample_mode_is_offline_safe(self) -> None:
        result = MarketDataClient().load(asset="BTC", prefer_live=False)

        self.assertEqual(result.status, "sample")
        self.assertGreaterEqual(len(result.prices), 30)
        self.assertTrue({"date", "open", "high", "low", "close", "volume"}.issubset(result.prices.columns))


    def test_twelve_data_live_mode_requires_key_and_falls_back(self) -> None:
        result = MarketDataClient().load(asset="GOLD", prefer_live=True, twelve_data_api_key="")

        self.assertIn("missing Twelve Data API key", result.status)
        self.assertGreaterEqual(len(result.prices), 30)


    def test_intraday_sample_mode_is_available(self) -> None:
        result = MarketDataClient().load_intraday(asset="BTC", prefer_live=False)

        self.assertEqual(result.status, "sample")
        self.assertGreaterEqual(len(result.prices), 30)
        self.assertTrue({"date", "open", "high", "low", "close", "volume"}.issubset(result.prices.columns))


    def test_btc_daily_falls_back_to_coinbase_when_binance_fails(self) -> None:
        binance_response = Mock()
        binance_response.raise_for_status.side_effect = requests.HTTPError("blocked")
        coinbase_response = Mock()
        coinbase_response.raise_for_status.return_value = None
        coinbase_response.json.return_value = [
            [1700000000 + i * 86400, 40000 + i, 40200 + i, 40100 + i, 40150 + i, 10 + i]
            for i in range(35)
        ]

        with patch("nero_app.core.market_data.requests.get", side_effect=[binance_response, coinbase_response]):
            result = MarketDataClient().load(asset="BTC", prefer_live=True, days=35)

        self.assertEqual(result.status, "live")
        self.assertIn("Coinbase BTC-USD", result.source)
        self.assertGreaterEqual(len(result.prices), 30)


    def test_btc_intraday_falls_back_to_coinbase_when_binance_fails(self) -> None:
        binance_response = Mock()
        binance_response.raise_for_status.side_effect = requests.HTTPError("blocked")
        coinbase_response = Mock()
        coinbase_response.raise_for_status.return_value = None
        coinbase_response.json.return_value = [
            [1700000000 + i * 3600, 40000 + i, 40200 + i, 40100 + i, 40150 + i, 10 + i]
            for i in range(35)
        ]

        with patch("nero_app.core.market_data.requests.get", side_effect=[binance_response, coinbase_response]):
            result = MarketDataClient().load_intraday(asset="BTC", prefer_live=True, interval="1h", candles=35)

        self.assertEqual(result.status, "live")
        self.assertIn("Coinbase BTC-USD", result.source)
        self.assertGreaterEqual(len(result.prices), 30)


    def test_btc_intraday_falls_back_to_kraken_when_binance_and_coinbase_fail(self) -> None:
        binance_response = Mock()
        binance_response.raise_for_status.side_effect = requests.HTTPError("blocked")
        coinbase_response = Mock()
        coinbase_response.raise_for_status.side_effect = requests.HTTPError("blocked")
        kraken_response = Mock()
        kraken_response.raise_for_status.return_value = None
        kraken_response.json.return_value = {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    [1700000000 + i * 3600, str(40100 + i), str(40200 + i), str(40000 + i), str(40150 + i), "40125", str(10 + i), 20]
                    for i in range(35)
                ],
                "last": 1700000000 + 35 * 3600,
            },
        }

        with patch("nero_app.core.market_data.requests.get", side_effect=[binance_response, coinbase_response, kraken_response]):
            result = MarketDataClient().load_intraday(asset="BTC", prefer_live=True, interval="1h", candles=35)

        self.assertEqual(result.status, "live")
        self.assertIn("Kraken XBTUSD", result.source)
        self.assertGreaterEqual(len(result.prices), 30)


    def test_twelve_data_parser_handles_missing_volume(self) -> None:
        payload = {
            "values": [
                {"datetime": "2026-07-05", "open": "3300", "high": "3310", "low": "3290", "close": "3305"},
                {"datetime": "2026-07-06", "open": "3305", "high": "3320", "low": "3300", "close": "3318"},
            ]
        }
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with patch("nero_app.core.market_data.requests.get", return_value=response):
            frame = MarketDataClient()._load_twelve_data_daily("XAU/USD", days=30, api_key="test")

        self.assertTrue("volume" in frame.columns)
        self.assertEqual(frame["volume"].sum(), 0.0)
        self.assertEqual(float(frame.iloc[-1]["close"]), 3318.0)


    def test_mobile_alert_requires_email_credentials(self) -> None:
        result = send_email_alert("", 465, "", "", "", "hello", "hello")

        self.assertFalse(result.ok)
        self.assertIn("required", result.message)


    def test_ntfy_alert_requires_topic(self) -> None:
        result = send_ntfy_alert("https://ntfy.sh", "", "hello", "hello")

        self.assertFalse(result.ok)
        self.assertIn("topic", result.message.lower())


    def test_ntfy_alert_posts_to_topic(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None

        with patch("nero_app.core.mobile_alerts.requests.post", return_value=response) as post:
            result = send_ntfy_alert("https://ntfy.sh", "nero-test", "Nero", "message")

        self.assertTrue(result.ok)
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "https://ntfy.sh/nero-test")


    def test_trade_alert_formatter_includes_plan_levels(self) -> None:
        plan = build_intraday_trade_plan(
            load_price_history().tail(120),
            asset="BTC",
            macro_direction="bullish",
            news_sentiment="Bullish",
            news_score=5,
            risk_score=0.35,
        )

        message = format_trade_alert("BTC", plan)

        self.assertIn("NERO ALERT | BTC", message)
        self.assertIn("Action:", message)
        self.assertIn("Entry:", message)
        self.assertIn("SL:", message)


    def test_trade_desk_builds_structured_plan(self) -> None:
        plan = build_intraday_trade_plan(
            load_price_history().tail(120),
            asset="BTC",
            macro_direction="bullish",
            news_sentiment="Bullish",
            news_score=5,
            risk_score=0.35,
        )

        self.assertIn(plan.action, {"WAIT_LONG_TRIGGER", "WAIT_SHORT_TRIGGER", "NO_TRADE"})
        self.assertGreaterEqual(plan.confidence, 0)
        self.assertLessEqual(plan.confidence, 1)
        self.assertGreater(len(plan.reasons), 0)


    def test_demo_trader_opens_triggered_paper_trade(self) -> None:
        prices = load_price_history().tail(40).copy()
        latest_close = float(prices.iloc[-1]["close"])
        plan = IntradayTradePlan(
            action="WAIT_LONG_TRIGGER",
            bias="Bullish",
            confidence=0.72,
            entry_trigger="Long only after trigger confirmation.",
            entry_price=latest_close,
            stop_loss=latest_close * 0.99,
            take_profit_1=latest_close * 1.01,
            take_profit_2=latest_close * 1.02,
            risk_reward_1=1.0,
            risk_reward_2=2.0,
            invalidation="Cancel below stop.",
            status="Wait for trigger",
            reasons=["test setup"],
        )
        prices.loc[prices.index[-1], "high"] = plan.entry_price * 1.001

        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo_trades.csv"
            summary = run_demo_trader("BTC", plan, prices, source="test", path=path)
            frame = load_demo_trades(path)

        self.assertEqual(summary.opened, 1)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["status"], "open")


    def test_demo_trader_closes_and_scores_paper_trade(self) -> None:
        prices = load_price_history().tail(40).copy()
        latest_close = float(prices.iloc[-1]["close"])
        plan = IntradayTradePlan(
            action="WAIT_LONG_TRIGGER",
            bias="Bullish",
            confidence=0.72,
            entry_trigger="Long only after trigger confirmation.",
            entry_price=latest_close,
            stop_loss=latest_close * 0.99,
            take_profit_1=latest_close * 1.01,
            take_profit_2=latest_close * 1.02,
            risk_reward_1=1.0,
            risk_reward_2=2.0,
            invalidation="Cancel below stop.",
            status="Wait for trigger",
            reasons=["test setup"],
        )
        prices.loc[prices.index[-1], "high"] = plan.entry_price * 1.001

        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo_trades.csv"
            run_demo_trader("BTC", plan, prices, source="test", path=path)
            future = prices.copy()
            future.loc[future.index[-1], "date"] = pd.to_datetime(future.iloc[-1]["date"]) + pd.Timedelta(hours=1)
            future.loc[future.index[-1], "high"] = plan.take_profit_1 * 1.001
            summary = run_demo_trader("BTC", plan, future, source="test", path=path)
            frame = load_demo_trades(path)
            scorecard = accountability_scorecard(frame)

        self.assertEqual(summary.closed, 1)
        self.assertEqual(frame.iloc[0]["status"], "closed")
        self.assertEqual(scorecard["wins"], 1)


    def test_prediction_log_round_trip(self) -> None:
        orchestrator = NeroOrchestrator(load_macro_events())
        result = orchestrator.run(
            AnalysisRequest(
                asset=AssetSymbol.BTC,
                headline="Liquidity improves for digital assets after dovish policy signals.",
                lookback_days=30,
            ),
            load_price_history(),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_log.csv"
            prices = load_price_history()
            append_prediction(result, data_source="test source", prices=prices, path=path)
            frame = load_prediction_log(path)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["asset"], "BTC")
        self.assertEqual(frame.iloc[0]["evaluation_status"], "pending")
        self.assertTrue(frame.iloc[0]["target_date"])


    def test_prediction_log_loads_mixed_schema_csv(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_log.csv"
            path.write_text(
                "timestamp,asset,headline,direction,confidence,risk_score,thematic_score,momentum_score,rsi,fair_value_gap,liquidity_sweep,data_source\n"
                "2026-07-01T10:00:00,BTC,Old headline,bullish,0.4,0.5,0.1,0.2,55,none,none,sample\n"
                "2026-07-02T10:00:00,GOLD,New headline,bearish,0.5,0.6,0.2,-0.1,48,none,none,live,2026-07-02,3300,7,2026-07-09,pending,,,,\n",
                encoding="utf-8",
            )

            frame = load_prediction_log(path)

        self.assertEqual(len(frame), 2)
        self.assertIn("evaluation_status", frame.columns)
        self.assertEqual(frame.iloc[0]["asset"], "BTC")
        self.assertEqual(frame.iloc[1]["evaluation_status"], "pending")


    def test_prediction_log_evaluates_directional_outcome(self) -> None:
        orchestrator = NeroOrchestrator(load_macro_events())
        prices = load_price_history()
        result = orchestrator.run(
            AnalysisRequest(
                asset=AssetSymbol.BTC,
                headline="Liquidity improves for digital assets after dovish policy signals.",
                lookback_days=30,
            ),
            prices,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction_log.csv"
            append_prediction(result, data_source="test source", prices=prices.head(60), horizon_days=7, path=path)
            frame = evaluate_prediction_log(prices, path=path)

        self.assertEqual(frame.iloc[0]["evaluation_status"], "evaluated")
        self.assertIn(frame.iloc[0]["outcome"], {"win", "miss", "neutral"})


    def test_news_ranker_filters_asset_headlines(self) -> None:
        items = [
            NewsItem(title="Oil prices rise after supply shock", source="test", link="", published="", tags=["Commodities", "Sentiment"]),
            NewsItem(title="Fed keeps rates steady", source="test", link="", published="", tags=["Central Banks"]),
        ]

        ranked = _rank_for_asset(items, "OIL")

        self.assertEqual(ranked[0].title, "Oil prices rise after supply shock")

    def test_ai_sentiment_local_fallback_scores_headlines(self) -> None:
        items = [
            NewsItem(title="Gold rally gains as dollar weakens", source="test", link="", published="", tags=["Gold", "Forex"]),
            NewsItem(title="Fed caution raises volatility", source="test", link="", published="", tags=["Central Banks", "Sentiment"]),
        ]

        result = analyze_news_sentiment(items, asset="GOLD")

        self.assertIn(result.overall_sentiment, {"Bullish", "Bearish", "Neutral"})
        self.assertGreaterEqual(result.sentiment_score, -10)
        self.assertLessEqual(result.sentiment_score, 10)


    def test_news_feed_fallback_is_available(self) -> None:
        client = NewsFeedClient(timeout_seconds=1)
        result = client.load("UNKNOWN")

        self.assertGreater(len(result.headlines), 0)
        self.assertIn("fallback", result.status)


    def test_settings_loads_environment_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with patch.dict(
                "os.environ",
                {
                    "TWELVE_DATA_API_KEY": "td",
                    "GEMINI_API_KEY": "gm",
                    "SMTP_PORT": "465",
                    "RECEIVER_EMAIL": "alerts@example.com",
                },
                clear=False,
            ):
                settings = load_settings(path=path)

        self.assertEqual(settings["twelve_data_api_key"], "td")
        self.assertEqual(settings["gemini_api_key"], "gm")
        self.assertEqual(settings["smtp_port"], 465)
        self.assertEqual(settings["receiver_email"], "alerts@example.com")
        self.assertTrue(settings["prefer_live"])


    def test_settings_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_settings({"prefer_live": True, "use_latest_news": True, "twelve_data_api_key": "abc", "gemini_api_key": "xyz"}, path=path)
            settings = load_settings(path=path)

        self.assertTrue(settings["prefer_live"])
        self.assertTrue(settings["use_latest_news"])
        self.assertEqual(settings["twelve_data_api_key"], "abc")


if __name__ == "__main__":
    unittest.main()
