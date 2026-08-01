from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_repair_lab import build_strategy_repair_lab_report


def main() -> None:
    report = build_strategy_repair_lab_report()
    print(f"Strategy Repair Lab complete. attempts={len(report)}")
    if report.empty:
        return
    decisions = report["promotion_decision"].value_counts().to_dict() if "promotion_decision" in report else {}
    print("Repair decisions: " + ", ".join(f"{key}={value}" for key, value in sorted(decisions.items())))
    for _, row in report.iterrows():
        print(
            f"{row['parent_label']} -> {row['repair_label']} "
            f"attempt={int(row['attempt_number'])}/{int(row['max_attempts'])} "
            f"status={row['status']} decision={row.get('promotion_decision', 'UNKNOWN')} "
            f"reason={row.get('failure_reason_code', 'UNKNOWN')} "
            f"repair_trades={int(row.get('repair_trades', 0))} "
            f"delta={float(row.get('repair_vs_parent_net_delta', 0.0)):.2f}"
        )


if __name__ == "__main__":
    main()
