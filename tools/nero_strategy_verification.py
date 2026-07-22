from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.strategy_verification import build_strategy_verification_report


def main() -> None:
    report = build_strategy_verification_report()
    if report.empty:
        print("Strategy verification complete. strategies=0")
        return
    counts = report["verdict"].value_counts().to_dict()
    summary = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"Strategy verification complete. strategies={len(report)} {summary}")
    top = report.sort_values(["evidence_score", "total_trades"], ascending=False).head(5)
    for _, row in top.iterrows():
        print(f"{row['display_label']}: {row['verdict']} score={row['evidence_score']:.0f} trades={int(row['total_trades'])} net={row['net_pnl']:.2f}")


if __name__ == "__main__":
    main()
