from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.profit_edge_engine import build_profit_edge_report


def main() -> None:
    report, summary = build_profit_edge_report()
    print(
        "Profit edge engine complete. "
        f"strategies={summary.total_strategies} "
        f"profit_candidates={summary.profit_candidates} "
        f"capital_drains={summary.capital_drains} "
        f"top={summary.top_candidate} "
        f"recovery_ratio={summary.recovery_ratio:.2%}"
    )
    if report.empty:
        return
    top = report[report["role"].eq("PROFIT_CANDIDATE")].sort_values(["edge_score", "net_pnl"], ascending=False).head(5)
    for _, row in top.iterrows():
        print(
            f"{row['display_label']}: score={row['edge_score']:.0f} "
            f"trades={int(row['total_trades'])} net={row['net_pnl']:.2f} "
            f"weight={row['paper_weight']:.2%}"
        )


if __name__ == "__main__":
    main()
