from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.bellwether_calibration import (
    DEFAULT_HEARTBEAT_PATH,
    DEFAULT_LEDGER_PATH,
    DEFAULT_REPORT_CSV,
    DEFAULT_REPORT_JSON,
    load_calibration_ledger,
    load_heartbeats,
    record_calibration_heartbeat,
    resolve_calibration_ledger,
    write_calibration_report,
)
from nero_app.core.mobile_alerts import send_ntfy_alert
from nero_app.core.prediction_log import load_prediction_log


def format_calibration_alert(recorded: int, resolved: int, status_counts: dict[str, int], report_path: Path) -> str:
    return "\n".join(
        [
            "NERO Bellwether Calibration",
            f"Records: {recorded}",
            f"Resolved this run: {resolved}",
            f"Statuses: {status_counts}",
            f"Report: {report_path}",
            "Health only: calibration accuracy does not equal tradable profitability.",
        ]
    )


def _send_ntfy_summary(message: str) -> None:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic or os.getenv("BELLWETHER_CALIBRATION_NTFY_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    result = send_ntfy_alert(
        server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        topic=topic,
        title="NERO Calibration",
        message=message,
        priority="default",
        tags="bar_chart",
    )
    print(result.message)


def main() -> None:
    try:
        summary = resolve_calibration_ledger(
            ledger_path=DEFAULT_LEDGER_PATH,
            report_csv=DEFAULT_REPORT_CSV,
            report_json=DEFAULT_REPORT_JSON,
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        )
        record_calibration_heartbeat(DEFAULT_HEARTBEAT_PATH, status="OK")
        write_calibration_report(
            load_calibration_ledger(DEFAULT_LEDGER_PATH),
            DEFAULT_REPORT_CSV,
            DEFAULT_REPORT_JSON,
            heartbeats=load_heartbeats(DEFAULT_HEARTBEAT_PATH),
            legacy_prediction_log=load_prediction_log(),
        )
    except Exception as exc:
        record_calibration_heartbeat(DEFAULT_HEARTBEAT_PATH, status="ERROR", reason_code=exc.__class__.__name__)
        raise
    print(
        "Bellwether calibration complete. "
        f"records={summary.recorded} resolved={summary.resolved} report={summary.report_path}"
    )
    _send_ntfy_summary(
        format_calibration_alert(
            recorded=summary.recorded,
            resolved=summary.resolved,
            status_counts=summary.status_counts,
            report_path=summary.report_path,
        )
    )


if __name__ == "__main__":
    main()
