from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.sunflower_profit_bridge import build_sunflower_profit_bridge_report


def main() -> None:
    report, summary = build_sunflower_profit_bridge_report()
    print(
        "Sunflower profit bridge complete. "
        f"strategies={summary.strategies_reviewed} "
        f"disciplined={summary.disciplined_profit_candidates} "
        f"early_watchlist={summary.early_profit_watchlist} "
        f"blocked={summary.capital_drains_blocked} "
        f"top={summary.top_candidate}"
    )
    for row in report.head(5).to_dict("records"):
        print(
            f"{row['display_label']}: gate={row['sunflower_gate']} "
            f"trades={row['total_trades']} net={row['net_pnl']} "
            f"score={row['discipline_score']}"
        )


if __name__ == "__main__":
    main()
