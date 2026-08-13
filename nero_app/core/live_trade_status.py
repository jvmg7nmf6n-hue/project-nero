"""Reconcile Strategy Lab and Mean Reversion open paper-trade state.

This module is deliberately conservative: an ``open_trade`` object in a JSON
state file is not enough evidence that a trade is currently live. The heartbeat
must also be fresh and agree with the state. Quarantined strategies are never
reported as trusted live trades.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nero_app.core.strategy_quarantine import DEFAULT_QUARANTINE_CSV, load_quarantined_strategy_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_LAB_DIR = PROJECT_ROOT / "nero_app" / "data" / "strategy_lab"
DEFAULT_MEAN_REVERSION_DIR = PROJECT_ROOT / "nero_app" / "data" / "mean_reversion"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "live_trade_status.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "live_trade_status.json"

MANUAL_BLOCKED_STRATEGIES = {
    "BREAKOUT_MOMENTUM_V1",
    "MR_DEEP_VALUE_V1",
    "MR_REGIME_FILTER_V1",
    "MR_TARGET_1R_V1",
    "MR_RELAXED_PULLBACK_V1",
}


@dataclass(frozen=True)
class LiveTradeStatusRow:
    system: str
    strategy_id: str
    asset: str
    symbol: str
    trade_id: str
    side: str
    opened_at: str
    entry_price: float | None
    target: float | None
    stop_loss: float | None
    state_status: str
    heartbeat_at: str
    heartbeat_age_minutes: float | None
    heartbeat_open_trade: bool | None
    trusted_live: bool
    issue: str
    action: str


@dataclass(frozen=True)
class LiveTradeStatusSummary:
    state_open_trades: int
    trusted_live_trades: int
    stale_or_blocked_trades: int
    heartbeat_mismatches: int
    generated_at: str


def build_live_trade_status_report(
    strategy_lab_dir: Path = DEFAULT_STRATEGY_LAB_DIR,
    mean_reversion_dir: Path = DEFAULT_MEAN_REVERSION_DIR,
    quarantine_csv: Path = DEFAULT_QUARANTINE_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    now: datetime | None = None,
    stale_after_minutes: int = 360,
) -> tuple[pd.DataFrame, LiveTradeStatusSummary]:
    """Build an auditable report of trusted vs stale paper open states."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    quarantined = load_quarantined_strategy_ids(quarantine_csv) | MANUAL_BLOCKED_STRATEGIES
    rows: list[LiveTradeStatusRow] = []
    rows.extend(_strategy_lab_rows(strategy_lab_dir, quarantined, now, stale_after_minutes))
    rows.extend(_mean_reversion_rows(mean_reversion_dir, now, stale_after_minutes))

    frame = pd.DataFrame([asdict(row) for row in rows])
    if not frame.empty:
        frame = frame.sort_values(["trusted_live", "issue", "strategy_id", "asset"], ascending=[False, True, True, True])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(frame.to_dict("records"), indent=2), encoding="utf-8")
    summary = LiveTradeStatusSummary(
        state_open_trades=len(rows),
        trusted_live_trades=sum(1 for row in rows if row.trusted_live),
        stale_or_blocked_trades=sum(1 for row in rows if not row.trusted_live),
        heartbeat_mismatches=sum(1 for row in rows if row.issue == "HEARTBEAT_STATE_MISMATCH"),
        generated_at=now.isoformat(),
    )
    return frame, summary


def _strategy_lab_rows(base: Path, quarantined: set[str], now: datetime, stale_after_minutes: int) -> list[LiveTradeStatusRow]:
    rows: list[LiveTradeStatusRow] = []
    if not base.exists():
        return rows
    for state_path in sorted(base.glob("*/state/*.json")):
        state = _read_json(state_path)
        trade = state.get("open_trade") if isinstance(state, dict) else None
        if not trade:
            continue
        strategy_id = state_path.parts[-3]
        heartbeat = _latest_heartbeat(base / strategy_id / "heartbeats" / "heartbeats.csv", asset=state_path.stem)
        rows.append(_row_from_trade("strategy_lab", strategy_id, state_path.stem, trade, heartbeat, now, stale_after_minutes, strategy_id in quarantined))
    return rows


def _mean_reversion_rows(base: Path, now: datetime, stale_after_minutes: int) -> list[LiveTradeStatusRow]:
    rows: list[LiveTradeStatusRow] = []
    if not base.exists():
        return rows
    for state_path in sorted(base.glob("state/*.json")):
        state = _read_json(state_path)
        trade = state.get("open_trade") if isinstance(state, dict) else None
        if not trade:
            continue
        heartbeat = _latest_heartbeat(base / "heartbeats" / "heartbeats.csv", asset=state_path.stem)
        rows.append(_row_from_trade("mean_reversion", "MEAN_REVERSION", state_path.stem, trade, heartbeat, now, stale_after_minutes, False))
    return rows


def _row_from_trade(
    system: str,
    strategy_id: str,
    asset_hint: str,
    trade: dict[str, Any],
    heartbeat: dict[str, Any],
    now: datetime,
    stale_after_minutes: int,
    blocked: bool,
) -> LiveTradeStatusRow:
    heartbeat_at = _parse_dt(heartbeat.get("timestamp"))
    age = ((now - heartbeat_at).total_seconds() / 60.0) if heartbeat_at else None
    heartbeat_open = _to_bool_or_none(heartbeat.get("open_trade"))
    stale = age is None or age > stale_after_minutes
    issue = "OK"
    if blocked:
        issue = "STRATEGY_QUARANTINED"
    elif heartbeat_open is False:
        issue = "HEARTBEAT_STATE_MISMATCH"
    elif stale:
        issue = "STALE_HEARTBEAT"
    elif heartbeat_open is None:
        issue = "NO_HEARTBEAT_OPEN_FLAG"
    trusted = issue == "OK"
    return LiveTradeStatusRow(
        system=system,
        strategy_id=strategy_id,
        asset=str(trade.get("asset", asset_hint) or asset_hint),
        symbol=str(trade.get("symbol", "") or ""),
        trade_id=str(trade.get("trade_id", "") or ""),
        side=str(trade.get("side", "") or ""),
        opened_at=str(trade.get("opened_at", "") or ""),
        entry_price=_float_or_none(trade.get("entry_price")),
        target=_float_or_none(trade.get("target") or trade.get("take_profit_1")),
        stop_loss=_float_or_none(trade.get("stop_loss")),
        state_status=str(trade.get("status", "OPEN") or "OPEN"),
        heartbeat_at=heartbeat_at.isoformat() if heartbeat_at else "",
        heartbeat_age_minutes=round(age, 2) if age is not None else None,
        heartbeat_open_trade=heartbeat_open,
        trusted_live=trusted,
        issue=issue,
        action="Count as trusted live paper trade." if trusted else "Do not count as live; keep audit trail and wait for a fresh workflow run or close event.",
    )


def _latest_heartbeat(path: Path, asset: str) -> dict[str, Any]:
    frame = _safe_read_csv(path)
    if frame.empty:
        return {}
    if "asset" in frame.columns:
        asset_rows = frame[frame["asset"].astype(str).str.upper() == asset.upper()]
        if not asset_rows.empty:
            frame = asset_rows
    timestamp_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    frame = frame.copy()
    frame["_parsed_ts"] = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=["_parsed_ts"]).sort_values("_parsed_ts")
    if frame.empty:
        return {}
    row = frame.iloc[-1].to_dict()
    row["timestamp"] = row["_parsed_ts"].isoformat()
    return row


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None
