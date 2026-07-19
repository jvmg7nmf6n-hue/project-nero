from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.market_data import MarketDataClient
from nero_app.core.range_mean_reversion import (
    RangeMRConfig,
    classify_range_mr,
    range_mr_hypothesis_configs,
    run_random_range_baseline,
    run_range_mean_reversion_backtest,
    split_train_test,
    summarize_range_mr_result,
)

REPORT_DIR = Path("reports")

TIERS: dict[str, dict[str, list[str]]] = {
    "TIER_1_RANGE_PRONE": {
        "EURUSD": ["1h", "4h", "1d"],
        "USDJPY": ["1h", "4h", "1d"],
        "GBPUSD": ["1h", "4h", "1d"],
        "USDCHF": ["1h", "4h", "1d"],
        "GOLD": ["4h", "1d", "1wk"],
        "SILVER": ["4h", "1d", "1wk"],
    },
    "TIER_2_CONDITIONAL": {
        "BTC": ["4h", "12h", "1d"],
        "ETH": ["4h", "12h", "1d"],
    },
    "TIER_3_STRESS": {
        "SOL": ["4h", "12h"],
        "NEAR": ["4h", "12h"],
    },
}

MIN_ROWS = {
    "1h": 500,
    "4h": 250,
    "12h": 140,
    "1d": 180,
    "1wk": 40,
}

CANDLES = {
    "1h": 9000,
    "4h": 2200,
    "12h": 800,
    "1d": 370,
    "1wk": 370,
}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = range_mr_hypothesis_configs()
    client = MarketDataClient(timeout_seconds=15)
    audit_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for tier, assets in TIERS.items():
        for asset, timeframes in assets.items():
            for timeframe in timeframes:
                candles, source, status = _fetch_candles(client, asset, timeframe)
                audit = _audit_row(tier, asset, timeframe, candles, source, status)
                audit_rows.append(audit)
                if audit["test_status"] != "OK":
                    for cfg in configs:
                        result_rows.append(_skipped_row(audit, reason=audit["test_status"], cfg=cfg))
                    continue
                for cfg in configs:
                    result_rows.append(_test_config(tier, asset, timeframe, candles, source, cfg))

    audit_frame = pd.DataFrame(audit_rows)
    results_frame = pd.DataFrame(result_rows)
    audit_frame.to_csv(REPORT_DIR / "range_mr_data_audit.csv", index=False)
    results_frame.to_csv(REPORT_DIR / "range_mr_backtest_results.csv", index=False)
    summary = _summary_payload(results_frame, audit_frame)
    (REPORT_DIR / "range_mr_backtest_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "range_mr_summary.txt").write_text(_summary_text(summary), encoding="utf-8")
    print(
        "Range MR backtest complete. "
        f"configs={len(results_frame)} ok={int((audit_frame['test_status'] == 'OK').sum())} "
        f"survived={int((results_frame['classification'] == 'SURVIVED').sum())} "
        f"watchlist={int((results_frame['classification'] == 'PROMISING_WATCHLIST').sum())} "
        f"report=reports/range_mr_backtest_results.csv"
    )


def _fetch_candles(client: MarketDataClient, asset: str, timeframe: str) -> tuple[pd.DataFrame, str, str]:
    try:
        if timeframe == "1wk":
            result = client.load(asset=asset, prefer_live=True, days=370, twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""))
            frame = _weekly(result.prices)
        elif timeframe == "1d":
            result = client.load(asset=asset, prefer_live=True, days=370, twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""))
            frame = result.prices
        else:
            result = client.load_intraday(
                asset=asset,
                prefer_live=True,
                interval=timeframe,
                candles=CANDLES[timeframe],
                twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
            )
            frame = result.prices
        return _last_year(frame), result.source, result.status
    except Exception as exc:  # noqa: BLE001 - report data failures instead of hiding them.
        return pd.DataFrame(), "ERROR", f"ERROR {exc.__class__.__name__}: {exc}"


def _audit_row(tier: str, asset: str, timeframe: str, candles: pd.DataFrame, source: str, status: str) -> dict[str, Any]:
    rows = int(len(candles)) if candles is not None else 0
    start = pd.to_datetime(candles["date"]).min().isoformat() if rows and "date" in candles else ""
    end = pd.to_datetime(candles["date"]).max().isoformat() if rows and "date" in candles else ""
    needed = MIN_ROWS[timeframe]
    test_status = "OK"
    if status != "live":
        test_status = "SKIPPED_NOT_LIVE"
    elif rows < needed:
        test_status = "SKIPPED_LOW_SAMPLE"
    return {
        "tier": tier,
        "asset": asset,
        "timeframe": timeframe,
        "rows": rows,
        "min_rows": needed,
        "start": start,
        "end": end,
        "source": source,
        "status": status,
        "test_status": test_status,
    }


def _test_config(tier: str, asset: str, timeframe: str, candles: pd.DataFrame, source: str, cfg: RangeMRConfig) -> dict[str, Any]:
    train_prices, test_prices = split_train_test(candles)
    train_trades, train_evals = run_range_mean_reversion_backtest(train_prices, asset, timeframe, cfg)
    test_trades, test_evals = run_range_mean_reversion_backtest(test_prices, asset, timeframe, cfg)
    all_trades, all_evals = run_range_mean_reversion_backtest(candles, asset, timeframe, cfg)
    random_trades = run_random_range_baseline(test_prices, asset, timeframe, len(test_trades), cfg)

    train_summary = summarize_range_mr_result(train_trades, train_evals)
    test_summary = summarize_range_mr_result(test_trades, test_evals)
    full_summary = summarize_range_mr_result(all_trades, all_evals)
    random_summary = summarize_range_mr_result(random_trades)
    grid_status = _grid_status(timeframe, train_summary, test_summary, cfg)
    classification = classify_range_mr(train_summary, test_summary, random_summary, grid_status, cfg.min_train_test_trades)
    return {
        "tier": tier,
        "hypothesis_id": cfg.hypothesis_id,
        "asset": asset,
        "timeframe": timeframe,
        "source": source,
        "rows": int(len(candles)),
        "train_trades": train_summary["trades"],
        "train_expectancy_r": round(train_summary["expectancy_r"], 4),
        "train_ci_low": round(train_summary["ci_low"], 4),
        "test_trades": test_summary["trades"],
        "test_expectancy_r": round(test_summary["expectancy_r"], 4),
        "test_ci_low": round(test_summary["ci_low"], 4),
        "full_trades": full_summary["trades"],
        "full_win_rate": round(full_summary["win_rate"], 4),
        "full_expectancy_r": round(full_summary["expectancy_r"], 4),
        "full_profit_factor": round(full_summary["profit_factor"], 4),
        "random_test_trades": random_summary["trades"],
        "random_test_expectancy_r": round(random_summary["expectancy_r"], 4),
        "edge_over_random_r": round(test_summary["expectancy_r"] - random_summary["expectancy_r"], 4),
        "top_rejection": full_summary["top_rejection"],
        "grid_status": grid_status,
        "classification": classification,
        "notes": _notes(classification, grid_status, train_summary, test_summary, random_summary),
    }


def _skipped_row(audit: dict[str, Any], reason: str, cfg: RangeMRConfig) -> dict[str, Any]:
    return {
        "tier": audit["tier"],
        "hypothesis_id": cfg.hypothesis_id,
        "asset": audit["asset"],
        "timeframe": audit["timeframe"],
        "source": audit["source"],
        "rows": audit["rows"],
        "train_trades": 0,
        "train_expectancy_r": 0.0,
        "train_ci_low": 0.0,
        "test_trades": 0,
        "test_expectancy_r": 0.0,
        "test_ci_low": 0.0,
        "full_trades": 0,
        "full_win_rate": 0.0,
        "full_expectancy_r": 0.0,
        "full_profit_factor": 0.0,
        "random_test_trades": 0,
        "random_test_expectancy_r": 0.0,
        "edge_over_random_r": 0.0,
        "top_rejection": "none",
        "grid_status": "NOT_RUN",
        "classification": "SKIPPED",
        "notes": reason,
    }


def _grid_status(timeframe: str, train: dict[str, Any], test: dict[str, Any], cfg: RangeMRConfig) -> str:
    if timeframe == "1h":
        return "NOT_APPLICABLE_NATIVE_1H"
    if timeframe in {"1d", "1wk"}:
        return "STRUCTURALLY_UNTESTED_DAILY_WEEKLY"
    if train["trades"] < cfg.min_train_test_trades or test["trades"] < cfg.min_train_test_trades:
        return "NOT_RUN_INSUFFICIENT_SAMPLE"
    if train["expectancy_r"] > 0 and test["expectancy_r"] > 0:
        return "PENDING_SHIFT_VERIFICATION"
    return "NOT_RUN_NEGATIVE_BASE"


def _notes(classification: str, grid_status: str, train: dict[str, Any], test: dict[str, Any], random_summary: dict[str, Any]) -> str:
    edge = test["expectancy_r"] - random_summary.get("expectancy_r", 0.0)
    if classification == "SURVIVED":
        return "Positive both halves, CI cleared zero, beat random, and grid shift passed."
    if classification == "PROMISING_WATCHLIST":
        return f"Positive but not fully verified; grid={grid_status}; edge_over_random={edge:.4f}R."
    return f"Not enough edge; train={train['expectancy_r']:.4f}R test={test['expectancy_r']:.4f}R edge_over_random={edge:.4f}R."


def _summary_payload(results: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
    valid = results[results["classification"] != "SKIPPED"] if not results.empty else pd.DataFrame()
    tier_expectancy = {}
    if not valid.empty:
        tier_expectancy = valid.groupby("tier")["full_expectancy_r"].mean().round(4).to_dict()
    return {
        "strategy": "RANGE_MEAN_REVERSION hypothesis family v1.0.0",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_configs": int(len(results)),
        "data_ok_configs": int((audit["test_status"] == "OK").sum()) if not audit.empty else 0,
        "survived": int((results["classification"] == "SURVIVED").sum()) if not results.empty else 0,
        "watchlist": int((results["classification"] == "PROMISING_WATCHLIST").sum()) if not results.empty else 0,
        "skipped": int((results["classification"] == "SKIPPED").sum()) if not results.empty else 0,
        "hypotheses": sorted(results["hypothesis_id"].dropna().unique().tolist()) if "hypothesis_id" in results else [],
        "tier_average_expectancy_r": tier_expectancy,
        "tier_1_beats_tier_3": _tier_1_beats_tier_3(tier_expectancy),
        "best_configs": _best_configs(results),
        "data_rule": "No clean live data means skipped, not guessed.",
    }


def _summary_text(summary: dict[str, Any]) -> str:
    best = summary.get("best_configs", [])
    lines = [
        f"{summary['strategy']}",
        f"Hypotheses: {', '.join(summary.get('hypotheses', []))}",
        f"Data OK configs: {summary['data_ok_configs']} / {summary['total_configs']}",
        f"SURVIVED: {summary['survived']} | WATCHLIST: {summary['watchlist']} | SKIPPED: {summary['skipped']}",
        f"Tier 1 beats Tier 3: {summary['tier_1_beats_tier_3']}",
        "Best configs:",
    ]
    for row in best:
        lines.append(f"- {row['hypothesis_id']} {row['asset']} {row['timeframe']} {row['classification']} test={row['test_expectancy_r']}R edge_random={row['edge_over_random_r']}R")
    return "\n".join(lines) + "\n"


def _tier_1_beats_tier_3(tiers: dict[str, float]) -> str:
    if "TIER_1_RANGE_PRONE" not in tiers or "TIER_3_STRESS" not in tiers:
        return "UNKNOWN_INSUFFICIENT_DATA"
    return "YES" if tiers["TIER_1_RANGE_PRONE"] > tiers["TIER_3_STRESS"] else "NO"


def _best_configs(results: pd.DataFrame) -> list[dict[str, Any]]:
    if results.empty:
        return []
    frame = results[results["classification"] != "SKIPPED"].copy()
    if frame.empty:
        return []
    cols = ["hypothesis_id", "asset", "timeframe", "classification", "test_expectancy_r", "edge_over_random_r", "full_trades"]
    return frame.sort_values(["test_expectancy_r", "edge_over_random_r"], ascending=[False, False])[cols].head(8).to_dict("records")


def _weekly(frame: pd.DataFrame) -> pd.DataFrame:
    prices = frame.copy()
    if prices.empty:
        return prices
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["date"]).set_index("date").sort_index()
    weekly = prices.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return weekly.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _last_year(frame: pd.DataFrame) -> pd.DataFrame:
    prices = frame.copy()
    if prices.empty or "date" not in prices:
        return prices
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["date"]).sort_values("date")
    cutoff = prices["date"].max() - pd.Timedelta(days=370)
    return prices[prices["date"] >= cutoff].reset_index(drop=True)


if __name__ == "__main__":
    main()



