from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.live_trade_status import build_live_trade_status_report


def main() -> None:
    report, summary = build_live_trade_status_report()
    print(
        "Live trade status complete. "
        f"state_open={summary.state_open_trades} trusted_live={summary.trusted_live_trades} "
        f"stale_or_blocked={summary.stale_or_blocked_trades} mismatches={summary.heartbeat_mismatches}"
    )
    if not report.empty:
        for row in report.to_dict("records"):
            if not row.get("trusted_live"):
                print(f"{row.get('strategy_id')} {row.get('asset')}: {row.get('issue')}")


if __name__ == "__main__":
    main()
