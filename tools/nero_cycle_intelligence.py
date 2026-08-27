from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nero_app.core.cycle_intelligence import build_cycle_intelligence_report


REPORT_DIR = ROOT / "reports"


def main() -> None:
    report = build_cycle_intelligence_report()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_csv = REPORT_DIR / "cycle_intelligence_report.csv"
    report_json = REPORT_DIR / "cycle_intelligence_report.json"
    availability_csv = REPORT_DIR / "cycle_intelligence_availability.csv"
    availability_json = REPORT_DIR / "cycle_intelligence_availability.json"
    correlation_csv = REPORT_DIR / "cycle_intelligence_correlation.csv"
    correlation_json = REPORT_DIR / "cycle_intelligence_correlation.json"

    pd.DataFrame(report.rows).to_csv(report_csv, index=False)
    report_json.write_text(json.dumps(report.rows, indent=2, default=str), encoding="utf-8")

    pd.DataFrame(report.availability_rows).to_csv(availability_csv, index=False)
    availability_json.write_text(json.dumps(report.availability_rows, indent=2, default=str), encoding="utf-8")

    pd.DataFrame(report.correlation_rows).to_csv(correlation_csv, index=False)
    correlation_json.write_text(json.dumps(report.correlation_rows, indent=2, default=str), encoding="utf-8")

    ok_count = sum(1 for row in report.rows if row.get("status") == "OK")
    unavailable_count = sum(1 for row in report.rows if row.get("status") == "UNAVAILABLE")
    print(
        "Cycle Intelligence complete. "
        f"assets={len(report.rows)} ok={ok_count} unavailable={unavailable_count} "
        f"report={report_csv}"
    )
    for row in report.rows:
        if row.get("status") == "OK":
            print(
                f"{row['asset']}: MM={float(row['mayer_multiple']):.3f} "
                f"percentile={float(row['mm_percentile_rank']):.1f}% "
                f"slope={row['sma200_slope_label']} "
                f"drawdown={float(row['drawdown_from_high_pct']):.1f}% "
                f"source={row['source']}"
            )
        else:
            print(f"{row['asset']}: UNAVAILABLE reason={row.get('unavailable_reason')}")


if __name__ == "__main__":
    main()
