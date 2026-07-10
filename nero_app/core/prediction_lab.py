from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nero_app.core.ai_sentiment import analyze_news_sentiment
from nero_app.core.data_loader import load_macro_events
from nero_app.core.market_data import MarketDataClient
from nero_app.core.news_feed import NewsFeedClient
from nero_app.core.orchestrator import NeroOrchestrator
from nero_app.core.prediction_log import DEFAULT_LOG_PATH, append_prediction, evaluate_prediction_log, load_prediction_log
from nero_app.core.schema import AnalysisRequest, AssetSymbol


DEFAULT_PREDICTION_LAB_REPORT = Path("reports") / "prediction_lab_report.csv"


@dataclass(frozen=True)
class PredictionLabSummary:
    recorded: int
    evaluated: int
    report_path: Path


def run_nero_core_prediction_lab(
    assets: list[str],
    horizon_days: int = 1,
    prediction_log_path: Path = DEFAULT_LOG_PATH,
    report_path: Path = DEFAULT_PREDICTION_LAB_REPORT,
    twelve_data_api_key: str = "",
    gemini_api_key: str = "",
) -> PredictionLabSummary:
    market_client = MarketDataClient(timeout_seconds=15)
    news_client = NewsFeedClient(timeout_seconds=12)
    orchestrator = NeroOrchestrator(load_macro_events())
    recorded = 0
    evaluated_total = 0

    for asset in [item.strip().upper() for item in assets if item.strip()]:
        market_data = market_client.load(asset=asset, prefer_live=True, days=365, twelve_data_api_key=twelve_data_api_key)
        evaluate_prediction_log(market_data.prices, path=prediction_log_path)
        evaluated_total = _evaluated_count(prediction_log_path)
        if market_data.prices.empty or _already_recorded_today(asset, horizon_days, market_data.prices, prediction_log_path):
            continue

        news_result = news_client.load(asset)
        sentiment = analyze_news_sentiment(news_result.headlines, asset=asset, gemini_api_key=gemini_api_key)
        headline = (
            f"Auto Prediction Lab | {asset} | News sentiment {sentiment.overall_sentiment} "
            f"({sentiment.sentiment_score}/10). {sentiment.summary}"
        )
        try:
            request = AnalysisRequest(asset=AssetSymbol(asset), headline=headline, lookback_days=30)
        except ValueError:
            continue
        result = orchestrator.run(request, market_data.prices)
        append_prediction(
            result,
            data_source=f"{market_data.source} ({market_data.status}) | Prediction Lab",
            prices=market_data.prices,
            horizon_days=horizon_days,
            path=prediction_log_path,
        )
        recorded += 1

    frame = load_prediction_log(prediction_log_path)
    write_prediction_lab_report(frame, report_path)
    return PredictionLabSummary(recorded=recorded, evaluated=evaluated_total, report_path=report_path)


def write_prediction_lab_report(frame: pd.DataFrame, report_path: Path = DEFAULT_PREDICTION_LAB_REPORT) -> pd.DataFrame:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if frame.empty:
        report = pd.DataFrame(columns=["agent", "asset", "total", "evaluated", "wins", "misses", "win_rate", "pending"])
        report.to_csv(report_path, index=False)
        return report

    for asset, group in frame.groupby("asset", dropna=False):
        evaluated = group[group["evaluation_status"] == "evaluated"]
        wins = int((evaluated["outcome"] == "win").sum()) if not evaluated.empty else 0
        misses = int((evaluated["outcome"] == "miss").sum()) if not evaluated.empty else 0
        pending = int((group["evaluation_status"] == "pending").sum()) if "evaluation_status" in group else 0
        rows.append(
            {
                "agent": "NERO_CORE",
                "asset": asset,
                "total": int(len(group)),
                "evaluated": int(len(evaluated)),
                "wins": wins,
                "misses": misses,
                "win_rate": wins / len(evaluated) if len(evaluated) else 0.0,
                "pending": pending,
            }
        )
    report = pd.DataFrame(rows).sort_values(["agent", "asset"])
    report.to_csv(report_path, index=False)
    return report


def _already_recorded_today(asset: str, horizon_days: int, prices: pd.DataFrame, path: Path) -> bool:
    frame = load_prediction_log(path)
    if frame.empty or prices.empty:
        return False
    latest = prices.sort_values("date").iloc[-1]
    entry_date = pd.to_datetime(latest["date"]).date().isoformat()
    matches = frame[
        (frame["asset"].astype(str).str.upper() == asset.upper())
        & (frame["entry_date"].astype(str) == entry_date)
        & (frame["horizon_days"].astype(str) == str(horizon_days))
        & (frame["data_source"].astype(str).str.contains("Prediction Lab", na=False))
    ]
    return not matches.empty


def _evaluated_count(path: Path) -> int:
    frame = load_prediction_log(path)
    if frame.empty:
        return 0
    return int((frame["evaluation_status"] == "evaluated").sum())
