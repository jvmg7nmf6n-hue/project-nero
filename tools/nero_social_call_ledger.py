from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.market_data import MarketDataClient
from nero_app.core.social_intelligence import (
    build_social_reliability_report,
    evaluate_social_calls,
    load_social_call_ledger,
    save_social_call_ledger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NERO social call ledger")
    parser.add_argument("--asset", default="BTC", help="Asset price feed to use for evaluation")
    parser.add_argument("--horizon-hours", type=int, default=24, help="Evaluation window")
    parser.add_argument("--prefer-live", action="store_true", help="Use live market feed if available")
    args = parser.parse_args()

    ledger = load_social_call_ledger()
    if ledger.empty:
        print("No social calls found.")
        return

    client = MarketDataClient()
    market = client.load_intraday(args.asset.upper(), prefer_live=args.prefer_live, interval="1h", candles=max(48, args.horizon_hours + 24))
    asset_calls = ledger[ledger["asset"].astype(str).str.upper() == args.asset.upper()].copy()
    other_calls = ledger[ledger["asset"].astype(str).str.upper() != args.asset.upper()].copy()
    evaluated_asset = evaluate_social_calls(asset_calls, market.prices, horizon_hours=args.horizon_hours)
    updated = evaluated_asset if other_calls.empty else __import__("pandas").concat([other_calls, evaluated_asset], ignore_index=True)
    save_social_call_ledger(updated)
    report = build_social_reliability_report(updated)
    print(f"Social call ledger evaluated for {args.asset.upper()} using {market.source} ({market.status}).")
    print(report.to_string(index=False) if not report.empty else "No reliability report yet.")


if __name__ == "__main__":
    main()
