from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mobile_alerts import send_ntfy_alert
from nero_app.core.prediction_lab import run_nero_core_prediction_lab


def format_prediction_lab_alert(assets: list[str], horizon_days: int, recorded: int, evaluated: int, report_path: Path) -> str:
    return "\n".join(
        [
            "NERO Prediction Lab",
            f"Assets: {','.join(assets)}",
            f"Horizon: {horizon_days} day(s)",
            f"Recorded: {recorded}",
            f"Evaluated: {evaluated}",
            f"Report: {report_path}",
            "Decision support only. Track sample size before trusting accuracy.",
        ]
    )


def _send_ntfy_summary(message: str) -> None:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
    result = send_ntfy_alert(
        server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        topic=topic,
        title="Nero Prediction Lab",
        message=message,
        priority="default",
        tags="chart_with_upwards_trend",
    )
    print(result.message)


def main() -> None:
    assets = [asset.strip().upper() for asset in os.getenv("PREDICTION_LAB_ASSETS", "BTC,GOLD").split(",") if asset.strip()]
    horizon_days = int(os.getenv("PREDICTION_LAB_HORIZON_DAYS", "1"))
    summary = run_nero_core_prediction_lab(
        assets=assets,
        horizon_days=horizon_days,
        twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    )
    print(
        "Prediction Lab complete. "
        f"assets={','.join(assets)} horizon_days={horizon_days} "
        f"recorded={summary.recorded} evaluated={summary.evaluated} report={summary.report_path}"
    )
    _send_ntfy_summary(
        format_prediction_lab_alert(
            assets=assets,
            horizon_days=horizon_days,
            recorded=summary.recorded,
            evaluated=summary.evaluated,
            report_path=summary.report_path,
        )
    )


if __name__ == "__main__":
    main()
