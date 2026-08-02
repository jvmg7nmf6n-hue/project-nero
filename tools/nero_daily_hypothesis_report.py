from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.daily_hypothesis_report import build_daily_hypothesis_report, format_daily_hypothesis_message


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    payload, summary = build_daily_hypothesis_report(update_state=False)
    message = format_daily_hypothesis_message(summary, payload)
    print(message)

    if not summary.notification_due:
        print("Daily hypothesis notification skipped: already sent and no new watchlist/hypothesis items.")
        return
    if not _truthy(os.getenv("MOBILE_ALERTS_ENABLED"), default=True):
        print("Daily hypothesis notification skipped: MOBILE_ALERTS_ENABLED is false.")
        return

    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        print("Daily hypothesis notification skipped: NTFY_TOPIC is missing.")
        return

    from nero_app.core.mobile_alerts import send_ntfy_alert

    result = send_ntfy_alert(
        server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        topic=topic,
        title="NERO Daily Hypothesis Report",
        message=message,
        priority="default",
        tags="chart_with_upwards_trend",
    )
    print(result.message)
    if result.ok:
        build_daily_hypothesis_report(update_state=True)


if __name__ == "__main__":
    main()
