from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_intelligence_engine import build_strategy_intelligence_report


def main() -> None:
    report, summary = build_strategy_intelligence_report()
    print(
        "Strategy intelligence engine complete. "
        f"failures={summary.failures_read} "
        f"proposals={summary.proposals} "
        f"test_lab_ready={summary.test_lab_ready} "
        f"research_queue={summary.research_queue} "
        f"top={summary.top_proposal} "
        f"status={summary.status}"
    )
    if report.empty:
        return
    for _, row in report.sort_values("intelligence_score", ascending=False).head(5).iterrows():
        print(
            f"{row['parent_label']} -> {row['proposed_label']} "
            f"decision={row['decision']} score={float(row['intelligence_score']):.0f} "
            f"failure={row['failure_class']} changes={int(row['rule_change_count'])}"
        )


if __name__ == "__main__":
    main()
