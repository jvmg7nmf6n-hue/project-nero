from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mobile_alerts import send_email_alert, send_ntfy_alert


PKT = ZoneInfo("Asia/Karachi")
DEFAULT_PREDICTION_LOG = Path("nero_app/data/prediction_log.csv")
DEFAULT_DEMO_TRADES = Path("nero_app/data/demo_trades.csv")
DEFAULT_PREDICTION_REPORT = Path("reports/prediction_lab_report.csv")
DEFAULT_MEAN_REVERSION_REPORT = Path("reports/mean_reversion_report.csv")
DEFAULT_WEEKLY_REPORT = Path("reports/nero_weekly_report.txt")


@dataclass(frozen=True)
class WeeklyReportPaths:
    prediction_log: Path = DEFAULT_PREDICTION_LOG
    demo_trades: Path = DEFAULT_DEMO_TRADES
    prediction_report: Path = DEFAULT_PREDICTION_REPORT
    mean_reversion_report: Path = DEFAULT_MEAN_REVERSION_REPORT
    output_report: Path = DEFAULT_WEEKLY_REPORT


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def build_weekly_report(paths: WeeklyReportPaths = WeeklyReportPaths(), now: datetime | None = None) -> str:
    now = now.astimezone(PKT) if now else datetime.now(PKT)
    start = now - timedelta(days=7)

    prediction_log = read_csv_if_exists(paths.prediction_log)
    demo_trades = read_csv_if_exists(paths.demo_trades)
    prediction_report = read_csv_if_exists(paths.prediction_report)
    mean_reversion = read_csv_if_exists(paths.mean_reversion_report)

    lines = [
        "NERO Weekly Performance Report",
        f"Report time: {now.strftime('%Y-%m-%d %I:%M %p')} PKT",
        f"Window: {start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}",
        "",
        "Executive Summary",
        _executive_summary(prediction_report, demo_trades, mean_reversion),
        "",
        "Prediction Lab",
        *_prediction_lab_lines(prediction_log, prediction_report),
        "",
        "NERO Demo Trade Accountability",
        *_demo_trade_lines(demo_trades),
        "",
        "Mean-Reversion Agent",
        *_mean_reversion_lines(mean_reversion),
        "",
        "Important Note",
        "NERO is still in forward-testing mode. Accuracy becomes meaningful after at least 20-30 evaluated predictions/trades per strategy.",
        "This report is decision support only, not financial advice.",
    ]
    return "\n".join(lines)


def write_weekly_report(report: str, path: Path = DEFAULT_WEEKLY_REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def _executive_summary(prediction_report: pd.DataFrame, demo_trades: pd.DataFrame, mean_reversion: pd.DataFrame) -> str:
    pred_total = _sum_column(prediction_report, "total")
    pred_evaluated = _sum_column(prediction_report, "evaluated")
    pred_wins = _sum_column(prediction_report, "wins")
    pred_win_rate = (pred_wins / pred_evaluated) if pred_evaluated else 0.0

    closed_trades = _closed_trades(demo_trades)
    demo_wins = int((closed_trades.get("result", pd.Series(dtype=str)).astype(str).str.lower() == "win").sum()) if not closed_trades.empty else 0
    demo_win_rate = (demo_wins / len(closed_trades)) if len(closed_trades) else 0.0

    mr_combined = _asset_row(mean_reversion, "COMBINED")
    mr_total = _int_value(mr_combined, "total_trades")
    mr_win_rate = _float_value(mr_combined, "win_rate")
    mr_expectancy = _float_value(mr_combined, "expectancy_r")

    return (
        f"Prediction Lab: {pred_total} total predictions, {pred_evaluated} evaluated, accuracy {pred_win_rate:.0%}.\n"
        f"Demo trades: {len(closed_trades)} closed trades, win rate {demo_win_rate:.0%}.\n"
        f"Mean-reversion: {mr_total} trades, win rate {mr_win_rate:.0%}, expectancy {mr_expectancy:.2f}R."
    )


def _prediction_lab_lines(prediction_log: pd.DataFrame, prediction_report: pd.DataFrame) -> list[str]:
    if prediction_report.empty:
        return ["No Prediction Lab report found yet."]
    lines: list[str] = []
    for _, row in prediction_report.sort_values("asset").iterrows():
        evaluated = _int_value(row, "evaluated")
        wins = _int_value(row, "wins")
        win_rate = _float_value(row, "win_rate")
        sample_note = "insufficient sample" if evaluated < 20 else "sample improving"
        lines.append(
            f"- {row.get('asset', 'UNKNOWN')}: total={_int_value(row, 'total')}, evaluated={evaluated}, "
            f"wins={wins}, misses={_int_value(row, 'misses')}, pending={_int_value(row, 'pending')}, "
            f"accuracy={win_rate:.0%} ({sample_note})"
        )
    if not prediction_log.empty and "timestamp" in prediction_log:
        lines.append(f"Latest prediction record: {str(prediction_log.iloc[-1].get('timestamp', 'unknown'))}")
    return lines


def _demo_trade_lines(demo_trades: pd.DataFrame) -> list[str]:
    if demo_trades.empty:
        return ["No demo trade records found yet."]
    closed = _closed_trades(demo_trades)
    if closed.empty:
        return [f"Total records={len(demo_trades)}, but no closed trades yet."]
    lines = [f"Closed trades: {len(closed)}"]
    for asset, group in closed.groupby("asset", dropna=False):
        wins = int((group.get("result", pd.Series(dtype=str)).astype(str).str.lower() == "win").sum())
        win_rate = wins / len(group) if len(group) else 0.0
        expectancy = pd.to_numeric(group.get("r_multiple", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()
        lines.append(f"- {asset}: trades={len(group)}, wins={wins}, win_rate={win_rate:.0%}, expectancy={expectancy:.2f}R")
    return lines


def _mean_reversion_lines(mean_reversion: pd.DataFrame) -> list[str]:
    if mean_reversion.empty:
        return ["No mean-reversion report found yet."]
    lines: list[str] = []
    for _, row in mean_reversion.sort_values("asset").iterrows():
        asset = str(row.get("asset", "UNKNOWN"))
        if asset == "COMBINED":
            continue
        total = _int_value(row, "total_trades")
        insufficient = str(row.get("insufficient_sample", "True")).lower() == "true"
        sample_note = "insufficient sample" if insufficient else "usable sample"
        lines.append(
            f"- {asset}: trades={total}, win_rate={_float_value(row, 'win_rate'):.0%}, "
            f"net_pnl={_float_value(row, 'net_pnl'):.2f}, expectancy={_float_value(row, 'expectancy_r'):.2f}R, "
            f"max_drawdown={_float_value(row, 'max_drawdown'):.2f} ({sample_note})"
        )
    combined = _asset_row(mean_reversion, "COMBINED")
    if combined is not None:
        lines.append(
            f"Combined: trades={_int_value(combined, 'total_trades')}, "
            f"win_rate={_float_value(combined, 'win_rate'):.0%}, expectancy={_float_value(combined, 'expectancy_r'):.2f}R"
        )
    return lines or ["No asset rows found in mean-reversion report."]


def send_weekly_report(report: str) -> None:
    subject = f"NERO Weekly Performance Report | {datetime.now(PKT).strftime('%Y-%m-%d')}"
    email_result = send_email_alert(
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465") or "465"),
        sender_email=os.getenv("SENDER_EMAIL", ""),
        app_password=os.getenv("EMAIL_APP_PASSWORD", ""),
        receiver_email=os.getenv("RECEIVER_EMAIL", ""),
        subject=subject,
        message=report,
    )
    print(email_result.message)

    topic = os.getenv("NTFY_TOPIC", "").strip()
    if topic:
        ntfy_result = send_ntfy_alert(
            server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=topic,
            title="NERO Weekly Report",
            message="NERO weekly performance email sent." if email_result.ok else email_result.message,
            priority="default",
            tags="bar_chart",
        )
        print(ntfy_result.message)


def _closed_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].astype(str).str.lower() == "closed"].copy()


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _asset_row(frame: pd.DataFrame, asset: str):
    if frame.empty or "asset" not in frame:
        return None
    rows = frame[frame["asset"].astype(str).str.upper() == asset.upper()]
    if rows.empty:
        return None
    return rows.iloc[0]


def _int_value(row, column: str) -> int:
    if row is None:
        return 0
    try:
        return int(float(row.get(column, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _float_value(row, column: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(column, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    report = build_weekly_report()
    output = write_weekly_report(report)
    print(f"Weekly report written: {output}")
    send_weekly_report(report)


if __name__ == "__main__":
    main()
