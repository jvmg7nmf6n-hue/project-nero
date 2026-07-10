from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.white_house_dataset_builder import build_white_house_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NERO White House BTC/Gold event-study dataset")
    parser.add_argument("--input", default="nero_app/data/white_house_market_events.csv", help="Input White House event memory CSV")
    parser.add_argument("--output", default="reports/white_house_market_events_enriched.csv", help="Output enriched event CSV")
    parser.add_argument("--summary", default="reports/white_house_impact_summary.csv", help="Output grouped impact summary CSV")
    parser.add_argument("--btc-prices", default="", help="Optional BTC daily price CSV with date,close")
    parser.add_argument("--gold-prices", default="", help="Optional Gold daily price CSV with date,close")
    args = parser.parse_args()

    result = build_white_house_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        btc_price_path=Path(args.btc_prices) if args.btc_prices else None,
        gold_price_path=Path(args.gold_prices) if args.gold_prices else None,
    )
    print(
        "White House dataset built. "
        f"events={result.events} btc_enriched={result.btc_enriched} gold_enriched={result.gold_enriched} "
        f"output={result.enriched_path} summary={result.summary_path}"
    )


if __name__ == "__main__":
    main()
