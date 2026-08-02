from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mean_reversion_agent import MeanReversionAgent, MeanReversionConfig, load_assets_from_env
from nero_app.core.mobile_alerts import send_ntfy_alert


def main() -> None:
    assets = load_assets_from_env()
    config = MeanReversionConfig(
        assets=assets,
        fee_bps=float(os.getenv("MR_FEE_BPS", "10")),
        slippage_bps=float(os.getenv("MR_SLIPPAGE_BPS", "2")),
        risk_per_trade=float(os.getenv("MR_RISK_PER_TRADE", "0.01")),
        daily_loss_guard_r=float(os.getenv("MR_DAILY_LOSS_GUARD_R", "-3")),
        max_notional_pct=float(os.getenv("MR_MAX_NOTIONAL_PCT", "1")),
    )
    agent = MeanReversionAgent(config=config, now=datetime.now(timezone.utc))
    summary = agent.run(list(assets.keys()))
    print(
        "Mean reversion complete. "
        f"evaluated={summary.evaluated} entries={summary.entries} exits={summary.exits} "
        f"missed_runs={summary.missed_runs} alerts={len(summary.alerts)}"
    )
    for alert in summary.alerts:
        print(alert)
        _send_ntfy(alert)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _send_ntfy(message: str) -> None:
    if "ERROR" in message and not _truthy(os.getenv("MR_ALERT_ERRORS"), default=False):
        print("Ntfy skipped for Mean Reversion error; set MR_ALERT_ERRORS=true to enable.")
        return
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
    result = send_ntfy_alert(
        server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        topic=topic,
        title="Nero mean-reversion agent",
        message=message,
        priority="high" if "ENTRY" in message or "ERROR" in message else "default",
        tags="chart_with_upwards_trend",
    )
    print(result.message)


if __name__ == "__main__":
    main()
