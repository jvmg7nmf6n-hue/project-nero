from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_quarantine import build_strategy_quarantine_report


def main() -> None:
    report = build_strategy_quarantine_report()
    print(f"Strategy quarantine complete. blocked={len(report)}")
    for _, row in report.head(10).iterrows():
        print(f"{row['display_label']}: blocked net={row['net_pnl']:.2f} trades={int(row['total_trades'])}")


if __name__ == "__main__":
    main()
