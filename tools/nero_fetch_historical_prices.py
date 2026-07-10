from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.historical_prices import DEFAULT_HISTORY_DIR, fetch_and_write_standard_histories
from nero_app.core.white_house_dataset_builder import build_white_house_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch BTC/Gold historical prices for NERO event studies")
    parser.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Optional end date YYYY-MM-DD")
    parser.add_argument("--output-dir", default=str(DEFAULT_HISTORY_DIR), help="Output directory for price CSV files")
    parser.add_argument("--twelve-data-api-key", default="", help="Optional Twelve Data key; otherwise uses env TWELVE_DATA_API_KEY")
    parser.add_argument("--rebuild-white-house", action="store_true", help="Rebuild White House event-study outputs after fetching prices")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = fetch_and_write_standard_histories(
        start=args.start,
        end=args.end or None,
        output_dir=output_dir,
        twelve_data_api_key=args.twelve_data_api_key or None,
    )
    for result in results:
        print(f"{result.asset}: {result.status} rows={result.rows} source={result.source} path={result.path}")

    if args.rebuild_white_house:
        btc_path = output_dir / "btc_daily.csv"
        gold_path = output_dir / "gold_daily.csv"
        built = build_white_house_dataset(
            btc_price_path=btc_path if btc_path.exists() else None,
            gold_price_path=gold_path if gold_path.exists() else None,
        )
        print(
            "White House dataset rebuilt. "
            f"events={built.events} btc_enriched={built.btc_enriched} gold_enriched={built.gold_enriched} "
            f"output={built.enriched_path} summary={built.summary_path}"
        )


if __name__ == "__main__":
    main()
