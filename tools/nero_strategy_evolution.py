from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_evolution import build_strategy_evolution_report, write_strategy_evolution_report


def main() -> None:
    report = build_strategy_evolution_report()
    write_strategy_evolution_report(report)
    print(
        "Strategy evolution complete. "
        f"label={report.label} maturity={report.maturity_score:.0f}/100 "
        f"trades={report.total_trades} losses={report.total_losses} "
        f"variants={len(report.variant_rows)}"
    )
    for row in report.recommendation_rows:
        print(f"{row['Candidate']}: {row['Action']} - {row['Recommendation']}")


if __name__ == "__main__":
    main()
