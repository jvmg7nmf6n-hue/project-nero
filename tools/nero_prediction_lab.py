from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.prediction_lab import run_nero_core_prediction_lab


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


if __name__ == "__main__":
    main()
