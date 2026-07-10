from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.white_house_impact import format_white_house_impact_report, score_white_house_impact


def main() -> None:
    parser = argparse.ArgumentParser(description="NERO White House market impact scorer")
    parser.add_argument("--text", required=True, help="White House headline, press-release text, or press-conference summary")
    parser.add_argument("--top", type=int, default=3, help="Number of historical matches to show")
    args = parser.parse_args()

    result = score_white_house_impact(args.text, top_n=args.top)
    print(format_white_house_impact_report(result))


if __name__ == "__main__":
    main()
