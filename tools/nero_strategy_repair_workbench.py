from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_repair_workbench import build_strategy_repair_workbench


def main() -> None:
    report = build_strategy_repair_workbench()
    print(f"Strategy repair workbench complete. repairs={len(report)}")
    for _, row in report.iterrows():
        print(f"{row['quarantined_label']} -> {row['repair_label']} status={row['repair_status']} repair_trades={int(row['repair_trades'])}")


if __name__ == "__main__":
    main()
