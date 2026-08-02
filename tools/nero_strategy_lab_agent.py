from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mean_reversion_agent import load_assets_from_env
from nero_app.core.mobile_alerts import send_ntfy_alert
from nero_app.core.strategy_lab_agent import STRATEGY_LAB_DEFAULT_ASSETS, run_strategy_lab


def main() -> None:
    assets = load_assets_from_env(default=STRATEGY_LAB_DEFAULT_ASSETS)
    summary = run_strategy_lab(assets=assets, now=datetime.now(timezone.utc))
    print(
        "Strategy lab complete. "
        f"candidates={summary.candidate_count} evaluated={summary.evaluated} "
        f"entries={summary.entries} exits={summary.exits} alerts={len(summary.alerts)}"
    )
    for alert in summary.alerts:
        print(alert)
        _send_ntfy(alert)


def _send_ntfy(message: str) -> None:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
    lowered = message.lower()
    alert_errors = os.getenv("STRATEGY_LAB_ALERT_ERRORS", "false").strip().lower() in {"1", "true", "yes", "on"}
    if ("error" in lowered or "stale feed" in lowered) and not alert_errors:
        print("Ntfy skipped for Strategy Lab error/stale-feed alert; set STRATEGY_LAB_ALERT_ERRORS=true to enable.")
        return
    priority = "high" if any(token in message for token in ["PAPER_ENTRY", "TARGET", "SL"]) else "default"
    result = send_ntfy_alert(
        server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        topic=topic,
        title="Nero Strategy TEST Lab",
        message=message,
        priority=priority,
        tags="test_tube",
    )
    print(result.message)


if __name__ == "__main__":
    main()
