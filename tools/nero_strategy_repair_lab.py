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
    for _, row in report.iterrows():
        print(
            f"{row['parent_label']} -> {row['repair_label']} "
            f"attempt={int(row['attempt_number'])}/{int(row['max_attempts'])} "
            f"status={row['status']} mode={row['fresh_data_mode']}"
        )


if __name__ == "__main__":
    main()
