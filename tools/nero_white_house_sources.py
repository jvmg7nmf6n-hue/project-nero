from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.white_house_sources import fetch_source_snapshot, list_official_sources, write_source_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="NERO official White House source ingestion helper")
    parser.add_argument("--list-sources", action="store_true", help="Print configured official source list")
    parser.add_argument("--fetch", action="store_true", help="Fetch a lightweight source snapshot")
    parser.add_argument("--output", default="reports/white_house_source_snapshot.csv", help="Snapshot output CSV path")
    args = parser.parse_args()

    if args.list_sources:
        for source in list_official_sources():
            print(f"{source.name} | {source.authority} | {source.url}")
        return

    if args.fetch:
        frame = fetch_source_snapshot()
        output = write_source_snapshot(frame, Path(args.output))
        print(f"White House source snapshot written: {output}")
        print(frame[["source", "status", "links_found", "relevant_links", "error"]].to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
