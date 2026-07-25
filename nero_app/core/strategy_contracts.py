"""Strategy contracts and run/data manifests for Project NERO.

These manifests make Strategy TEST Lab runs auditable. They do not place
orders, change parameters, or promote strategies; they record what was tested,
under which assumptions, and whether the saved data is trustworthy enough for
later review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_LAB_DIR = PROJECT_ROOT / "nero_app" / "data" / "strategy_lab"
DEFAULT_CONTRACTS_CSV = DEFAULT_REPORT_DIR / "strategy_contracts.csv"
DEFAULT_CONTRACTS_JSON = DEFAULT_REPORT_DIR / "strategy_contracts.json"
DEFAULT_RUN_MANIFEST_JSON = DEFAULT_REPORT_DIR / "strategy_run_manifest.json"
DEFAULT_DATA_QUALITY_CSV = DEFAULT_REPORT_DIR / "strategy_data_quality_manifest.csv"
DEFAULT_DATA_QUALITY_JSON = DEFAULT_REPORT_DIR / "strategy_data_quality_manifest.json"


@dataclass(frozen=True)
class StrategyContractRow:
    candidate_id: str
    display_label: str
    bucket: str
    family: str
    title: str
    enabled: bool
    interval: str
    asset_scope: str
    excluded_assets: str
    direction: str
    entry_rule: str
    exit_rule: str
    stop_rule: str
    target_rule: str
    risk_rule: str
    data_requirement: str
    evidence_note: str
    strategy_version: str


@dataclass(frozen=True)
class DataQualityRow:
    candidate_id: str
    display_label: str
    asset: str
    report_exists: bool
    evaluations_exists: bool
    runtime_errors: int
    total_trades: int
    net_pnl: float
    rating: str
    quality_status: str
    trusted_for_promotion: bool
    primary_reason: str


def write_strategy_architecture_manifests(
    candidates: Iterable[Any],
    *,
    assets: dict[str, str] | None = None,
    run_summary: Any | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    lab_dir: Path = DEFAULT_LAB_DIR,
    now: datetime | None = None,
    workflow_name: str = "local-strategy-lab",
) -> dict[str, Path]:
    """Write contracts, run manifest, and data-quality manifests."""
    report_dir = Path(report_dir)
    lab_dir = Path(lab_dir)
    now = now or datetime.now(timezone.utc)
    candidate_list = list(candidates)
    report_dir.mkdir(parents=True, exist_ok=True)

    contracts = pd.DataFrame([asdict(strategy_contract_row(spec)) for spec in candidate_list])
    contracts.to_csv(DEFAULT_CONTRACTS_CSV if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_contracts.csv", index=False)
    contracts_json = DEFAULT_CONTRACTS_JSON if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_contracts.json"
    contracts_json.write_text(json.dumps(contracts.to_dict("records"), indent=2), encoding="utf-8")

    data_quality = build_data_quality_manifest(candidate_list, report_dir=report_dir, lab_dir=lab_dir)
    data_quality_csv = DEFAULT_DATA_QUALITY_CSV if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_data_quality_manifest.csv"
    data_quality_json = DEFAULT_DATA_QUALITY_JSON if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_data_quality_manifest.json"
    data_quality.to_csv(data_quality_csv, index=False)
    data_quality_json.write_text(json.dumps(data_quality.to_dict("records"), indent=2), encoding="utf-8")

    run_manifest = build_run_manifest(
        candidate_list,
        assets=assets or {},
        run_summary=run_summary,
        data_quality=data_quality,
        now=now,
        workflow_name=workflow_name,
    )
    run_manifest_json = DEFAULT_RUN_MANIFEST_JSON if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_run_manifest.json"
    run_manifest_json.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return {
        "contracts_csv": DEFAULT_CONTRACTS_CSV if report_dir == DEFAULT_REPORT_DIR else report_dir / "strategy_contracts.csv",
        "contracts_json": contracts_json,
        "data_quality_csv": data_quality_csv,
        "data_quality_json": data_quality_json,
        "run_manifest_json": run_manifest_json,
    }


def strategy_contract_row(spec: Any) -> StrategyContractRow:
    """Convert a CandidateSpec-like object into a formal contract row."""
    candidate_id = _text(_field(spec, "candidate_id"), "UNKNOWN")
    family = _text(_field(spec, "family"), "UNKNOWN")
    direction = _strategy_direction(spec, family)
    return StrategyContractRow(
        candidate_id=candidate_id,
        display_label=_text(_field(spec, "display_label"), candidate_id),
        bucket=_text(_field(spec, "bucket"), "UNKNOWN"),
        family=family,
        title=_text(_field(spec, "title"), candidate_id),
        enabled=bool(_field(spec, "enabled", True)),
        interval=_text(_field(spec, "interval"), "1h"),
        asset_scope=_csv(_field(spec, "asset_filter", ())) or "ALL",
        excluded_assets=_csv(_field(spec, "asset_exclude", ())) or "AUTO_QUARANTINE",
        direction=direction,
        entry_rule=_entry_rule(spec, family),
        exit_rule=_exit_rule(spec, family),
        stop_rule=f"ATR stop x{_field(spec, 'atr_stop_multiple', 1.5)}",
        target_rule=_target_rule(spec),
        risk_rule="Paper only; risk_per_trade and max_notional_pct controlled by Strategy Lab environment.",
        data_requirement="Closed candles only; no sample/fallback/stale feed may support promotion.",
        evidence_note=_text(_field(spec, "evidence_note"), ""),
        strategy_version=f"strategy-lab-v1.0.0:{candidate_id}",
    )


def build_data_quality_manifest(
    candidates: Iterable[Any],
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> pd.DataFrame:
    """Build one data-quality row per candidate/asset report observation."""
    rows: list[DataQualityRow] = []
    report_dir = Path(report_dir)
    lab_dir = Path(lab_dir)
    for spec in candidates:
        candidate_id = _text(_field(spec, "candidate_id"), "UNKNOWN")
        display_label = _text(_field(spec, "display_label"), candidate_id)
        report_path = report_dir / f"strategy_lab_{candidate_id}.csv"
        report = _safe_read_csv(report_path)
        error_count = _runtime_error_count(lab_dir / candidate_id / "trades" / "runtime_errors.csv")
        evaluations_exists = (lab_dir / candidate_id / "trades" / "evaluations.csv").exists()
        if report.empty:
            rows.append(
                _data_quality_row(
                    candidate_id,
                    display_label,
                    "ALL",
                    report_path.exists(),
                    evaluations_exists,
                    error_count,
                    {},
                )
            )
            continue
        asset_rows = report[report["asset"].astype(str).str.upper() != "COMBINED"] if "asset" in report else report
        for row in asset_rows.to_dict("records"):
            rows.append(
                _data_quality_row(
                    candidate_id,
                    display_label,
                    _text(row.get("asset"), "UNKNOWN"),
                    True,
                    evaluations_exists,
                    error_count,
                    row,
                )
            )
    frame = pd.DataFrame([asdict(row) for row in rows])
    if not frame.empty:
        frame = frame.sort_values(["quality_status", "candidate_id", "asset"])
    return frame


def build_run_manifest(
    candidates: Iterable[Any],
    *,
    assets: dict[str, str],
    run_summary: Any | None,
    data_quality: pd.DataFrame,
    now: datetime,
    workflow_name: str,
) -> dict[str, Any]:
    """Build a run-level manifest suitable for audit/replay notes."""
    candidate_list = list(candidates)
    quality_counts = data_quality["quality_status"].value_counts().to_dict() if not data_quality.empty else {}
    return {
        "run_id": f"strategy-lab-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": now.isoformat(),
        "workflow_name": workflow_name,
        "git_commit": os.getenv("GITHUB_SHA", "local"),
        "strategy_count": len(candidate_list),
        "enabled_strategy_count": sum(1 for spec in candidate_list if bool(_field(spec, "enabled", True))),
        "asset_count": len(assets),
        "assets": assets,
        "assumptions": {
            "paper_only": True,
            "fee_bps": os.getenv("SLAB_FEE_BPS", os.getenv("MR_FEE_BPS", "10")),
            "slippage_bps": os.getenv("SLAB_SLIPPAGE_BPS", os.getenv("MR_SLIPPAGE_BPS", "2")),
            "risk_per_trade": os.getenv("SLAB_RISK_PER_TRADE", "0.01"),
            "daily_loss_guard_r": os.getenv("SLAB_DAILY_LOSS_GUARD_R", "-3"),
            "max_notional_pct": os.getenv("SLAB_MAX_NOTIONAL_PCT", "1"),
        },
        "run_totals": {
            "evaluated": int(_field(run_summary, "evaluated", 0)) if run_summary is not None else 0,
            "entries": int(_field(run_summary, "entries", 0)) if run_summary is not None else 0,
            "exits": int(_field(run_summary, "exits", 0)) if run_summary is not None else 0,
            "alerts": len(_field(run_summary, "alerts", [])) if run_summary is not None else 0,
        },
        "data_quality_counts": quality_counts,
        "promotion_rule": "No promotion unless data quality is OK, sample is adequate, expectancy > 0, profit factor >= 1.10, drawdown acceptable, and independent validation passes.",
    }


def _data_quality_row(
    candidate_id: str,
    display_label: str,
    asset: str,
    report_exists: bool,
    evaluations_exists: bool,
    runtime_errors: int,
    row: dict[str, Any],
) -> DataQualityRow:
    total_trades = int(_num(row.get("total_trades"), 0))
    net_pnl = round(_num(row.get("net_pnl"), 0.0), 2)
    rating = _text(row.get("rating"), "NO_REPORT")
    quality_status, trusted, reason = _quality_status(report_exists, evaluations_exists, runtime_errors, row)
    return DataQualityRow(
        candidate_id=candidate_id,
        display_label=display_label,
        asset=asset,
        report_exists=report_exists,
        evaluations_exists=evaluations_exists,
        runtime_errors=runtime_errors,
        total_trades=total_trades,
        net_pnl=net_pnl,
        rating=rating,
        quality_status=quality_status,
        trusted_for_promotion=trusted,
        primary_reason=reason,
    )


def _quality_status(report_exists: bool, evaluations_exists: bool, runtime_errors: int, row: dict[str, Any]) -> tuple[str, bool, str]:
    if not report_exists:
        return "NO_REPORT", False, "Strategy has no saved report yet."
    if runtime_errors > 0:
        return "CHECK", False, "Runtime errors exist for this candidate."
    text = " ".join(str(row.get(key, "")) for key in ["rating", "evidence_note", "asset_exclude"]).lower()
    if "sample data" in text or "fallback" in text or "stale" in text:
        return "CHECK", False, "Report metadata mentions sample, fallback, or stale data."
    if not evaluations_exists:
        return "CHECK", False, "No evaluation ledger exists for this candidate."
    return "OK", True, "Saved report and evaluation ledger exist with no runtime-error marker."


def _entry_rule(spec: Any, family: str) -> str:
    if family == "Short Momentum":
        return f"SHORT breakdown; lookback={_field(spec, 'breakout_lookback', 20)}, trend support required={_field(spec, 'require_trend_support', False)}"
    if family == "Momentum":
        return f"LONG breakout/retest; lookback={_field(spec, 'breakout_lookback', 20)}, retest={_field(spec, 'require_breakout_retest', False)}"
    if family == "Range Mean Reversion":
        return f"ADX range gate with {_text(_field(spec, 'range_entry_mode'), 'BAND_EXTREME')} entry"
    if family == "Pairs Research":
        return "BTC-ETH relationship research; no direct single-leg promotion."
    return f"LONG mean reversion; RSI below {_field(spec, 'rsi_entry_below', 35.0)}, Bollinger/MA filters."


def _exit_rule(spec: Any, family: str) -> str:
    if family == "Range Mean Reversion":
        return "Exit at SMA20 reversion, ADX regime break, or disaster stop."
    if family == "Pairs Research":
        return "Research-only relationship close; not execution authority."
    return "Exit at target, stop loss, or strategy-specific invalidation."


def _target_rule(spec: Any) -> str:
    mode = _text(_field(spec, "target_mode"), "FROZEN_MA20")
    if mode == "FROZEN_MA20":
        return "Frozen MA20 target at entry."
    return f"{mode} target."


def _strategy_direction(spec: Any, family: str) -> str:
    side = _text(_field(spec, "entry_side"), "")
    if side:
        return side
    if family == "Short Momentum":
        return "SHORT"
    if family == "Pairs Research":
        return "RESEARCH_ONLY"
    return "LONG"


def _runtime_error_count(path: Path) -> int:
    frame = _safe_read_csv(path)
    return int(len(frame)) if not frame.empty else 0


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _csv(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
