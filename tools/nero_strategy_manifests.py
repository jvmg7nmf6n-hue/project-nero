from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mean_reversion_agent import load_assets_from_env
from nero_app.core.strategy_contracts import write_strategy_architecture_manifests
from nero_app.core.strategy_lab_agent import CANDIDATES, STRATEGY_LAB_DEFAULT_ASSETS


def main() -> None:
    assets = load_assets_from_env(default=STRATEGY_LAB_DEFAULT_ASSETS)
    paths = write_strategy_architecture_manifests(
        CANDIDATES.values(),
        assets=assets,
        now=datetime.now(timezone.utc),
        workflow_name="manual-strategy-manifest-refresh",
    )
    print("Strategy manifests complete.")
    for label, path in paths.items():
        print(f"{label}={path}")


if __name__ == "__main__":
    main()
