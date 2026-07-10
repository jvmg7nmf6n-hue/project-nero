from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.historical_market_memory import (
    format_regime_report,
    infer_environment_tags,
    load_historical_events,
    score_regime_similarity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="NERO historical market memory regime scorer")
    parser.add_argument("--asset", default="BTC", choices=["BTC", "GOLD"], help="Asset to score")
    parser.add_argument("--news", default="", help="Current news/context text")
    parser.add_argument("--dxy", default="neutral", help="DXY trend: weak/strong/neutral")
    parser.add_argument("--fed", default="neutral", help="Fed tone: dovish/hawkish/neutral")
    parser.add_argument("--risk", default="neutral", help="Risk appetite: risk_on/risk_off/neutral")
    parser.add_argument("--etf", default="neutral", help="ETF flow: inflows/outflows/neutral")
    parser.add_argument("--policy", default="neutral", help="Policy tone: friendly/hostile/neutral")
    args = parser.parse_args()

    events = load_historical_events()
    tags = infer_environment_tags(
        asset=args.asset,
        news_text=args.news,
        dxy_trend=args.dxy,
        fed_tone=args.fed,
        risk_appetite=args.risk,
        etf_flow=args.etf,
        policy_tone=args.policy,
    )
    result = score_regime_similarity(args.asset, tags, events)
    print(format_regime_report(result))


if __name__ == "__main__":
    main()
