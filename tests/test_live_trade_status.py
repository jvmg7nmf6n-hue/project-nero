from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nero_app.core.live_trade_status import build_live_trade_status_report


class LiveTradeStatusTests(unittest.TestCase):
    def test_quarantined_strategy_open_state_is_not_trusted_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lab = base / "lab"
            state_dir = lab / "OLD_BREAKOUT" / "state"
            hb_dir = lab / "OLD_BREAKOUT" / "heartbeats"
            state_dir.mkdir(parents=True)
            hb_dir.mkdir(parents=True)
            (state_dir / "BTC.json").write_text(json.dumps({"open_trade": _trade("OLD_BREAKOUT", "BTC")}), encoding="utf-8")
            pd.DataFrame([{"timestamp": "2026-08-13T09:00:00+00:00", "asset": "BTC", "open_trade": True}]).to_csv(hb_dir / "heartbeats.csv", index=False)
            quarantine = base / "quarantine.csv"
            pd.DataFrame([{"candidate_id": "OLD_BREAKOUT", "blocked": True}]).to_csv(quarantine, index=False)

            report, summary = build_live_trade_status_report(
                strategy_lab_dir=lab,
                mean_reversion_dir=base / "mr",
                quarantine_csv=quarantine,
                output_csv=base / "live.csv",
                output_json=base / "live.json",
                now=datetime(2026, 8, 13, 9, 5, tzinfo=timezone.utc),
            )

        self.assertEqual(summary.state_open_trades, 1)
        self.assertEqual(summary.trusted_live_trades, 0)
        self.assertEqual(report.iloc[0]["issue"], "STRATEGY_QUARANTINED")

    def test_heartbeat_state_mismatch_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lab = base / "lab"
            state_dir = lab / "RMR" / "state"
            hb_dir = lab / "RMR" / "heartbeats"
            state_dir.mkdir(parents=True)
            hb_dir.mkdir(parents=True)
            (state_dir / "ETH.json").write_text(json.dumps({"open_trade": _trade("RMR", "ETH")}), encoding="utf-8")
            pd.DataFrame([{"timestamp": "2026-08-13T09:00:00+00:00", "asset": "ETH", "open_trade": False}]).to_csv(hb_dir / "heartbeats.csv", index=False)

            report, summary = build_live_trade_status_report(
                strategy_lab_dir=lab,
                mean_reversion_dir=base / "mr",
                quarantine_csv=base / "missing.csv",
                output_csv=base / "live.csv",
                output_json=base / "live.json",
                now=datetime(2026, 8, 13, 9, 5, tzinfo=timezone.utc),
            )

        self.assertEqual(summary.trusted_live_trades, 0)
        self.assertEqual(summary.heartbeat_mismatches, 1)
        self.assertEqual(report.iloc[0]["issue"], "HEARTBEAT_STATE_MISMATCH")

    def test_fresh_matching_heartbeat_counts_as_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lab = base / "lab"
            state_dir = lab / "RMR" / "state"
            hb_dir = lab / "RMR" / "heartbeats"
            state_dir.mkdir(parents=True)
            hb_dir.mkdir(parents=True)
            (state_dir / "ETH.json").write_text(json.dumps({"open_trade": _trade("RMR", "ETH")}), encoding="utf-8")
            pd.DataFrame([{"timestamp": "2026-08-13T09:00:00+00:00", "asset": "ETH", "open_trade": True}]).to_csv(hb_dir / "heartbeats.csv", index=False)

            _, summary = build_live_trade_status_report(
                strategy_lab_dir=lab,
                mean_reversion_dir=base / "mr",
                quarantine_csv=base / "missing.csv",
                output_csv=base / "live.csv",
                output_json=base / "live.json",
                now=datetime(2026, 8, 13, 9, 5, tzinfo=timezone.utc),
            )

        self.assertEqual(summary.trusted_live_trades, 1)
        self.assertEqual(summary.stale_or_blocked_trades, 0)


def _trade(strategy: str, asset: str) -> dict[str, object]:
    return {
        "trade_id": f"{strategy}-{asset}",
        "candidate_id": strategy,
        "asset": asset,
        "symbol": f"{asset}USDT",
        "side": "LONG",
        "opened_at": "2026-08-13T08:00:00+00:00",
        "entry_price": 100.0,
        "target": 105.0,
        "stop_loss": 98.0,
        "status": "OPEN",
    }


if __name__ == "__main__":
    unittest.main()
