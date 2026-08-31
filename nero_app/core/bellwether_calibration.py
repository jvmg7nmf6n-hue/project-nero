from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from nero_app.core.market_data import MarketDataClient, MarketDataResult
from nero_app.core.schema import NeroResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "nero_app" / "data" / "bellwether_calibration"
DEFAULT_LEDGER_PATH = DEFAULT_CALIBRATION_DIR / "forecast_ledger.csv"
DEFAULT_HEARTBEAT_PATH = DEFAULT_CALIBRATION_DIR / "heartbeats.csv"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "reports" / "bellwether_calibration_report.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "bellwether_calibration_report.json"
SCHEMA_VERSION = "bellwether_calibration_v1"
PREREGISTRATION_VERSION = "calibration_v1"
DEADBAND_PCT = 0.0005

STATUS_VALUES = {
    "PENDING",
    "RESOLVED",
    "MISSED",
    "UNRESOLVABLE",
    "SOURCE_MISMATCH",
    "LOW_COVERAGE",
    "LEGACY_UNSCORED",
    "SCHEMA_MISMATCH",
    "NOT_A_PROBABILITY",
    "DATA_QUALITY_FAIL",
}

LEDGER_COLUMNS = [
    "forecast_id",
    "asset",
    "issued_at_utc",
    "published_value",
    "probability_status",
    "direction",
    "confidence",
    "haircut_side",
    "price_at_issue",
    "source_id",
    "source_label",
    "intended_resolution_at_utc",
    "horizon_hours",
    "horizon_class",
    "agent_provenance_snapshot",
    "system_config_hash",
    "coverage_pct",
    "schema_version",
    "status",
    "reason_code",
    "price_at_resolution",
    "resolution_source_id",
    "outcome",
    "outcome_no_deadband",
    "resolved_at_utc",
    "brier_error",
]

HEARTBEAT_COLUMNS = [
    "run_id",
    "started_at_utc",
    "completed_at_utc",
    "status",
    "reason_code",
    "schema_version",
]


@dataclass(frozen=True)
class CalibrationCycleSummary:
    recorded: int
    resolved: int
    report_path: Path
    status_counts: dict[str, int]
    first_real_resolution: str


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def append_calibration_forecast(
    result: NeroResult,
    market_data: MarketDataResult,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    issued_at: datetime | None = None,
    coverage_pct: float = 1.0,
) -> bool:
    """Append one generation-time record, idempotently keyed by forecast_id."""
    issued = _ensure_utc(issued_at or now_utc())
    asset = result.request.asset.value
    prices = market_data.prices
    issue_check = validate_price_frame(prices, market_data.source, market_data.status, now=issued)
    source_id = source_id_for(asset, market_data.source, market_data.status)
    issue_price = ""
    issue_time = ""
    if issue_check == "OK":
        issue_row = prices.sort_values("date").iloc[-1]
        issue_price = float(issue_row["close"])
        issue_time = _to_utc(issue_row["date"]).isoformat()

    forecast_id = build_forecast_id(
        asset=asset,
        issued_at=issued,
        source_id=source_id,
        direction=result.verdict.direction,
        confidence=result.verdict.confidence,
        issue_time=issue_time,
    )
    frame = load_calibration_ledger(ledger_path)
    if not frame.empty and "forecast_id" in frame and forecast_id in set(frame["forecast_id"].astype(str)):
        return False

    intended = intended_resolution_at(asset, issued)
    status = "PENDING" if issue_check == "OK" else "DATA_QUALITY_FAIL"
    reason_code = "" if issue_check == "OK" else issue_check
    probability_status = "NOT_A_PROBABILITY"
    row = {
        "forecast_id": forecast_id,
        "asset": asset,
        "issued_at_utc": issued.isoformat(),
        "published_value": result.verdict.confidence,
        "probability_status": probability_status,
        "direction": result.verdict.direction,
        "confidence": result.verdict.confidence,
        "haircut_side": "NO_HAIRCUT_FOUND",
        "price_at_issue": issue_price,
        "source_id": source_id,
        "source_label": market_data.source,
        "intended_resolution_at_utc": intended.isoformat(),
        "horizon_hours": round((intended - issued).total_seconds() / 3600, 2),
        "horizon_class": horizon_class_for(asset, issued, intended),
        "agent_provenance_snapshot": json.dumps(_provenance_snapshot(result), sort_keys=True),
        "system_config_hash": system_config_hash(result, source_id),
        "coverage_pct": max(0.0, min(1.0, float(coverage_pct))),
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason_code": reason_code,
        "price_at_resolution": "",
        "resolution_source_id": "",
        "outcome": "",
        "outcome_no_deadband": "",
        "resolved_at_utc": "",
        "brier_error": "",
    }
    _append_row(ledger_path, row, LEDGER_COLUMNS)
    return True


def resolve_calibration_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    report_csv: Path = DEFAULT_REPORT_CSV,
    report_json: Path = DEFAULT_REPORT_JSON,
    market_client: MarketDataClient | None = None,
    twelve_data_api_key: str = "",
    now: datetime | None = None,
) -> CalibrationCycleSummary:
    current = _ensure_utc(now or now_utc())
    frame = load_calibration_ledger(ledger_path)
    if frame.empty:
        write_calibration_report(frame, report_csv, report_json)
        return CalibrationCycleSummary(0, 0, report_csv, {}, "")

    market_client = market_client or MarketDataClient(timeout_seconds=15)
    resolved = 0
    for index, row in frame.iterrows():
        if str(row.get("schema_version", "")) != SCHEMA_VERSION:
            frame.loc[index, "status"] = "SCHEMA_MISMATCH"
            frame.loc[index, "reason_code"] = "SCHEMA_VERSION_MISMATCH"
            continue
        if str(row.get("status", "")) != "PENDING":
            continue
        intended = _parse_utc(row.get("intended_resolution_at_utc", ""))
        if intended is None:
            frame.loc[index, "status"] = "UNRESOLVABLE"
            frame.loc[index, "reason_code"] = "MISSING_INTENDED_RESOLUTION"
            continue
        if current < intended:
            continue

        asset = str(row.get("asset", "")).upper()
        data = market_client.load(asset=asset, prefer_live=True, days=15, twelve_data_api_key=twelve_data_api_key)
        quality = validate_price_frame(data.prices, data.source, data.status, now=current)
        if quality != "OK":
            frame.loc[index, "status"] = "DATA_QUALITY_FAIL"
            frame.loc[index, "reason_code"] = quality
            continue
        resolution_source_id = source_id_for(asset, data.source, data.status)
        if resolution_source_id != str(row.get("source_id", "")):
            frame.loc[index, "status"] = "SOURCE_MISMATCH"
            frame.loc[index, "reason_code"] = "ISSUE_RESOLUTION_SOURCE_MISMATCH"
            frame.loc[index, "resolution_source_id"] = resolution_source_id
            continue
        exit_row = _resolution_price_row(data.prices, issued_at=_parse_utc(row.get("issued_at_utc", "")), intended_at=intended)
        if exit_row is None:
            frame.loc[index, "status"] = "UNRESOLVABLE"
            frame.loc[index, "reason_code"] = "NO_RESOLUTION_PRICE_AT_OR_BEFORE_HORIZON"
            continue

        issue_price = _float_or_none(row.get("price_at_issue"))
        resolution_price = _float_or_none(exit_row["close"])
        if issue_price is None or resolution_price is None or issue_price <= 0:
            frame.loc[index, "status"] = "DATA_QUALITY_FAIL"
            frame.loc[index, "reason_code"] = "MISSING_ISSUE_OR_RESOLUTION_PRICE"
            continue

        outcome_no_deadband = "up" if resolution_price > issue_price else ("down" if resolution_price < issue_price else "flat")
        move = (resolution_price - issue_price) / issue_price
        outcome = "flat" if abs(move) <= DEADBAND_PCT else outcome_no_deadband
        final_status = "RESOLVED" if str(row.get("probability_status", "")) == "PROBABILITY_VALID" else "NOT_A_PROBABILITY"
        frame.loc[index, "status"] = final_status
        frame.loc[index, "reason_code"] = "" if final_status == "RESOLVED" else "CONFIDENCE_FIELD_NOT_PROBABILITY"
        frame.loc[index, "price_at_resolution"] = resolution_price
        frame.loc[index, "resolution_source_id"] = resolution_source_id
        frame.loc[index, "outcome"] = outcome
        frame.loc[index, "outcome_no_deadband"] = outcome_no_deadband
        frame.loc[index, "resolved_at_utc"] = current.isoformat()
        if final_status == "RESOLVED":
            frame.loc[index, "brier_error"] = _brier_error(float(row["published_value"]), outcome_no_deadband)
        resolved += 1

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _normalize_columns(frame, LEDGER_COLUMNS)
    frame.to_csv(ledger_path, index=False)
    report_frame = write_calibration_report(frame, report_csv, report_json)
    status_counts = {str(key): int(value) for key, value in frame["status"].value_counts().to_dict().items()}
    first_real = ""
    completed = frame[frame["status"].isin(["RESOLVED", "NOT_A_PROBABILITY"])].copy()
    if not completed.empty:
        first_real = str(completed.sort_values("resolved_at_utc").iloc[0].get("forecast_id", ""))
    return CalibrationCycleSummary(
        recorded=len(frame),
        resolved=resolved,
        report_path=report_csv,
        status_counts=status_counts,
        first_real_resolution=first_real,
    )


def record_calibration_heartbeat(
    heartbeat_path: Path = DEFAULT_HEARTBEAT_PATH,
    status: str = "OK",
    reason_code: str = "",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> str:
    started = _ensure_utc(started_at or now_utc())
    completed = _ensure_utc(completed_at or now_utc())
    run_id = hashlib.sha256(f"{started.isoformat()}|{status}|{reason_code}".encode("utf-8")).hexdigest()[:16]
    row = {
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "status": status,
        "reason_code": reason_code,
        "schema_version": SCHEMA_VERSION,
    }
    frame = load_heartbeats(heartbeat_path)
    if frame.empty or run_id not in set(frame.get("run_id", pd.Series(dtype=str)).astype(str)):
        _append_row(heartbeat_path, row, HEARTBEAT_COLUMNS)
    return run_id


def build_calibration_report(
    ledger: pd.DataFrame,
    heartbeats: pd.DataFrame | None = None,
    legacy_prediction_log: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ledger = _normalize_columns(ledger.copy(), LEDGER_COLUMNS)
    rows: list[dict[str, object]] = []
    if not ledger.empty:
        for asset, group in ledger.groupby("asset", dropna=False):
            probability_valid = group[(group["status"] == "RESOLVED") & (group["probability_status"] == "PROBABILITY_VALID")]
            not_probability = group[group["status"] == "NOT_A_PROBABILITY"]
            pending = group[group["status"] == "PENDING"]
            source_mismatch = group[group["status"] == "SOURCE_MISMATCH"]
            data_fail = group[group["status"] == "DATA_QUALITY_FAIL"]
            brier = pd.to_numeric(probability_valid.get("brier_error", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(
                {
                    "asset": asset or "UNKNOWN",
                    "schema_version": SCHEMA_VERSION,
                    "preregistration_version": PREREGISTRATION_VERSION,
                    "total_records": int(len(group)),
                    "probability_valid_resolved": int(len(probability_valid)),
                    "not_probability_resolved": int(len(not_probability)),
                    "pending": int(len(pending)),
                    "source_mismatch": int(len(source_mismatch)),
                    "data_quality_fail": int(len(data_fail)),
                    "brier_mean": float(brier.mean()) if len(brier) else "",
                    "n_raw": int(len(probability_valid)),
                    "n_eff": int(_non_overlapping_count(probability_valid)),
                    "n_eff_ratio": _n_eff_ratio(probability_valid),
                    "calibration_status": _calibration_status(probability_valid),
                    "warning": _asset_warning(probability_valid, not_probability),
                }
            )
    report = pd.DataFrame(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_version": PREREGISTRATION_VERSION,
        "total_records": int(len(ledger)),
        "status_counts": {} if ledger.empty else {str(k): int(v) for k, v in ledger["status"].value_counts().to_dict().items()},
        "legacy_prediction_rows": _legacy_row_count(legacy_prediction_log),
        "heartbeat": build_fire_rate_summary(heartbeats if heartbeats is not None else pd.DataFrame()),
        "calibration_disclaimer": "Calibration accuracy does not equal tradable profitability.",
        "headline_status": "INSUFFICIENT_PROBABILITY_DATA",
    }
    return report, summary


def write_calibration_report(
    ledger: pd.DataFrame,
    report_csv: Path = DEFAULT_REPORT_CSV,
    report_json: Path = DEFAULT_REPORT_JSON,
    heartbeats: pd.DataFrame | None = None,
    legacy_prediction_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    report, summary = build_calibration_report(ledger, heartbeats=heartbeats, legacy_prediction_log=legacy_prediction_log)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)
    report_json.write_text(json.dumps({"summary": summary, "rows": report.to_dict(orient="records")}, indent=2), encoding="utf-8")
    return report


def load_calibration_ledger(path: Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return _normalize_columns(frame, LEDGER_COLUMNS)


def load_heartbeats(path: Path = DEFAULT_HEARTBEAT_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HEARTBEAT_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame(columns=HEARTBEAT_COLUMNS)
    return _normalize_columns(frame, HEARTBEAT_COLUMNS)


def mark_legacy_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        output["calibration_status"] = []
        return output
    output["calibration_status"] = "LEGACY_UNSCORED"
    output["calibration_reason_code"] = "PREDATES_CALIBRATION_SCHEMA"
    return output


def validate_price_frame(prices: pd.DataFrame, source: str, status: str, now: datetime | None = None) -> str:
    if status != "live":
        return "NON_LIVE_SOURCE"
    source_text = str(source).lower()
    if "generated sample" in source_text or "sample" in source_text or "fallback" in source_text:
        return "FALLBACK_OR_SYNTHETIC_SOURCE"
    if prices is None or prices.empty:
        return "EMPTY_PRICE_FRAME"
    required = {"date", "close"}
    if not required.issubset(prices.columns):
        return "MISSING_REQUIRED_PRICE_COLUMNS"
    dates = pd.to_datetime(prices["date"], errors="coerce", utc=True)
    if dates.isna().any():
        return "MISSING_OR_INVALID_TIMESTAMP"
    if dates.duplicated().any():
        return "DUPLICATE_TIMESTAMP"
    if not dates.is_monotonic_increasing:
        return "NON_MONOTONIC_TIMESTAMP"
    close = pd.to_numeric(prices["close"], errors="coerce")
    if close.isna().any():
        return "MISSING_CLOSE"
    current = _ensure_utc(now or now_utc())
    if (dates > current + timedelta(minutes=5)).any():
        return "FUTURE_TIMESTAMP"
    if dates.iloc[-1].to_pydatetime() < current - timedelta(days=7):
        return "STALE_FEED"
    return "OK"


def source_id_for(asset: str, source: str, status: str) -> str:
    clean_asset = asset.strip().upper()
    clean = " ".join(str(source).strip().split())
    if status != "live":
        return f"nonlive:{clean_asset}"
    lowered = clean.lower()
    if lowered.startswith("binance "):
        parts = clean.split()
        symbol = parts[1] if len(parts) > 1 else clean_asset
        interval = parts[2] if len(parts) > 2 else "daily"
        return f"binance:{symbol}:{interval}"
    if lowered.startswith("coinbase "):
        parts = clean.split()
        symbol = parts[1] if len(parts) > 1 else clean_asset
        interval = parts[2] if len(parts) > 2 else "daily"
        return f"coinbase:{symbol}:{interval}"
    if lowered.startswith("kraken "):
        parts = clean.split()
        symbol = parts[1] if len(parts) > 1 else clean_asset
        interval = parts[2] if len(parts) > 2 else "daily"
        return f"kraken:{symbol}:{interval}"
    if lowered.startswith("twelve data "):
        symbol = clean.replace("Twelve Data ", "").replace(" daily candles", "").strip()
        return f"twelvedata:{symbol}:daily"
    if lowered.startswith("yfinance "):
        parts = clean.split()
        symbol = parts[1] if len(parts) > 1 else clean_asset
        interval = parts[2] if len(parts) > 2 else "daily"
        return f"yfinance:{symbol}:{interval}"
    return f"live:{clean_asset}:{hashlib.sha256(clean.encode('utf-8')).hexdigest()[:10]}"


def intended_resolution_at(asset: str, issued_at: datetime) -> datetime:
    issued = _ensure_utc(issued_at)
    if asset.strip().upper() == "GOLD":
        candidate = issued.replace(hour=17, minute=0, second=0, microsecond=0)
        if candidate <= issued:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate
    return issued + timedelta(hours=24)


def horizon_class_for(asset: str, issued_at: datetime, intended_at: datetime) -> str:
    clean_asset = asset.strip().upper()
    if clean_asset == "GOLD":
        if intended_at.weekday() == 0 and _ensure_utc(issued_at).weekday() >= 4:
            return "gold_weekend_spanning_session"
        return "gold_london_ny_session_approx"
    return "btc_24h_wall_clock" if clean_asset == "BTC" else "generic_24h_wall_clock"


def build_fire_rate_summary(heartbeats: pd.DataFrame, expected_per_day: int = 4) -> dict[str, object]:
    if heartbeats is None or heartbeats.empty or "started_at_utc" not in heartbeats:
        return {
            "heartbeat_records": 0,
            "expected_records_7d": expected_per_day * 7,
            "fire_rate_7d": "",
            "missed_cycles_7d": "",
            "status": "NO_HEARTBEAT_HISTORY",
        }
    frame = heartbeats.copy()
    frame["started_at_utc"] = pd.to_datetime(frame["started_at_utc"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["started_at_utc"])
    if frame.empty:
        return {
            "heartbeat_records": 0,
            "expected_records_7d": expected_per_day * 7,
            "fire_rate_7d": "",
            "missed_cycles_7d": "",
            "status": "NO_VALID_HEARTBEATS",
        }
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    recent = frame[frame["started_at_utc"] >= cutoff]
    expected = expected_per_day * 7
    actual = len(recent)
    missed = max(0, expected - actual)
    return {
        "heartbeat_records": int(len(frame)),
        "expected_records_7d": int(expected),
        "fire_rate_7d": round(actual / expected, 3) if expected else "",
        "missed_cycles_7d": int(missed),
        "status": "OK" if actual else "NO_RECENT_HEARTBEATS",
    }


def build_forecast_id(asset: str, issued_at: datetime, source_id: str, direction: str, confidence: float, issue_time: str = "") -> str:
    payload = "|".join([asset.strip().upper(), issued_at.isoformat(), source_id, direction, f"{float(confidence):.6f}", issue_time])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def system_config_hash(result: NeroResult, source_id: str) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "asset": result.request.asset.value,
        "source_id": source_id,
        "direction_model": "NeroOrchestrator",
        "confidence_semantics": "mapped_research_confidence_not_probability",
        "preregistration": PREREGISTRATION_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _resolution_price_row(prices: pd.DataFrame, issued_at: datetime | None, intended_at: datetime):
    if issued_at is None:
        return None
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    eligible = frame[(frame["date"] > pd.Timestamp(issued_at)) & (frame["date"] <= pd.Timestamp(intended_at))]
    if eligible.empty:
        return None
    return eligible.sort_values("date").iloc[-1]


def _brier_error(probability: float, outcome_no_deadband: str) -> float:
    observed = 1.0 if outcome_no_deadband == "up" else 0.0
    return round((max(0.0, min(1.0, probability)) - observed) ** 2, 6)


def _provenance_snapshot(result: NeroResult) -> dict[str, object]:
    return {
        "producer": "NeroOrchestrator",
        "brain_matches": len(result.brain.matches),
        "dominant_tags": result.brain.dominant_tags,
        "technical_regime": result.assessment.market_regime,
        "volatility_regime": result.assessment.volatility_regime,
        "confidence_semantics": "mapped_research_confidence_not_probability",
    }


def _append_row(path: Path, row: dict[str, object], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def _normalize_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = ""
    return output[list(columns)].astype("object")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).replace(microsecond=0)
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _parse_utc(value: object) -> datetime | None:
    try:
        parsed = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(microsecond=0)


def _to_utc(value: object) -> datetime:
    parsed = pd.to_datetime(value, utc=True)
    return parsed.to_pydatetime().replace(microsecond=0)


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _non_overlapping_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    cleaned = frame.sort_values("issued_at_utc").copy()
    selected: list[pd.Series] = []
    last_resolution: datetime | None = None
    for _, row in cleaned.iterrows():
        issued = _parse_utc(row.get("issued_at_utc"))
        intended = _parse_utc(row.get("intended_resolution_at_utc"))
        if issued is None or intended is None:
            continue
        if last_resolution is None or issued >= last_resolution:
            selected.append(row)
            last_resolution = intended
    return len(selected)


def _n_eff_ratio(frame: pd.DataFrame) -> float | str:
    raw = len(frame)
    if raw == 0:
        return ""
    return round(_non_overlapping_count(frame) / raw, 3)


def _calibration_status(probability_valid: pd.DataFrame) -> str:
    n_eff = _non_overlapping_count(probability_valid)
    if n_eff < 50:
        return "INSUFFICIENT_EFFECTIVE_SAMPLE"
    return "READY_FOR_CALIBRATION_SCORE"


def _asset_warning(probability_valid: pd.DataFrame, not_probability: pd.DataFrame) -> str:
    if len(probability_valid) == 0 and len(not_probability) > 0:
        return "Existing values are resolved context but not scoreable because confidence is not a probability."
    return "Calibration accuracy does not equal tradable profitability."


def _legacy_row_count(frame: pd.DataFrame | None) -> int:
    if frame is None or frame.empty:
        return 0
    return int(len(frame))
