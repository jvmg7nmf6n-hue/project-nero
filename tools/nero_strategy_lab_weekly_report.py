from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.mobile_alerts import send_email_alert, send_ntfy_alert
from nero_app.core.strategy_lab_agent import CANDIDATES, write_strategy_lab_summary

PKT = ZoneInfo("Asia/Karachi")
DEFAULT_SUMMARY = Path("reports/strategy_lab_summary.csv")
DEFAULT_REPORT = Path("reports/strategy_lab_weekly_report.txt")


def main() -> None:
    summary = write_strategy_lab_summary(Path("reports"), list(CANDIDATES.values()))
    report = build_strategy_lab_weekly_report(summary)
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(report, encoding="utf-8")
    print(f"Strategy lab weekly report written: {DEFAULT_REPORT}")
    send_report(report)


def build_strategy_lab_weekly_report(summary: pd.DataFrame | None = None, now: datetime | None = None) -> str:
    now = now.astimezone(PKT) if now else datetime.now(PKT)
    if summary is None:
        summary = pd.read_csv(DEFAULT_SUMMARY) if DEFAULT_SUMMARY.exists() else write_strategy_lab_summary(Path("reports"), list(CANDIDATES.values()))
    lines = [
        "NERO Strategy TEST Lab Weekly Report",
        f"Report time: {now.strftime('%Y-%m-%d %I:%M %p')} PKT",
        "",
        "Executive Summary",
    ]
    if summary.empty:
        lines.extend(["No strategy lab records found yet.", ""])
    else:
        ranked = summary.sort_values(["rating_score", "total_trades"], ascending=False)
        best = ranked.iloc[0]
        total_trades = int(pd.to_numeric(ranked.get("total_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        lines.append(f"5-algo paper-test is active. Total closed paper trades across candidates: {total_trades}.")
        lines.append(f"Current top candidate: {best.get('candidate_id', '-')} | rating={best.get('rating', '-')} | score={float(best.get('rating_score', 0.0)):.0f}/100.")
        if total_trades < 150:
            lines.append("Sample warning: final algo ranking needs roughly 30-50 closed trades per candidate.")
        lines.append("")
        lines.append("Candidate Scoreboard")
        for _, row in ranked.iterrows():
            lines.append(
                f"- {row.get('candidate_id', '-')}: trades={int(float(row.get('total_trades', 0) or 0))}, "
                f"win_rate={float(row.get('win_rate', 0.0) or 0.0):.0%}, "
                f"expectancy={float(row.get('expectancy_r', 0.0) or 0.0):.2f}R, "
                f"profit_factor={float(row.get('profit_factor', 0.0) or 0.0):.2f}, "
                f"drawdown={float(row.get('max_drawdown', 0.0) or 0.0):.0%}, "
                f"rating={row.get('rating', '-')}, score={float(row.get('rating_score', 0.0) or 0.0):.0f}/100"
            )
        lines.append("")
    lines.extend([
        "Operating Rule",
        "All candidates are paper-trading only. NERO must not place real orders and must not auto-promote parameters without manual version approval.",
    ])
    return "\n".join(lines)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def send_report(report: str) -> None:
    receiver_email = _env_first("RECEIVER_EMAIL", default="tareekh39@gmail.com")
    sender_email = _env_first("SENDER_EMAIL", "GMAIL_EMAIL", "GMAIL_USER", "EMAIL_SENDER", default=receiver_email)
    app_password = _env_first("EMAIL_APP_PASSWORD", "GMAIL_APP_PASSWORD", "APP_PASSWORD", "EMAIL_PASSWORD")
    smtp_host = _env_first("SMTP_HOST", default="smtp.gmail.com")
    smtp_port = int(_env_first("SMTP_PORT", default="465") or "465")
    result = None
    if _truthy(os.getenv("STRATEGY_LAB_WEEKLY_EMAIL_ENABLED"), default=False):
        result = send_email_alert(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            sender_email=sender_email,
            app_password=app_password,
            receiver_email=receiver_email,
            subject=f"NERO Strategy TEST Lab Weekly Report | {datetime.now(PKT).strftime('%Y-%m-%d')}",
            message=report,
        )
        print(result.message)
    else:
        print("Strategy Lab weekly email skipped; set STRATEGY_LAB_WEEKLY_EMAIL_ENABLED=true to enable.")
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if topic and _truthy(os.getenv("STRATEGY_LAB_WEEKLY_NTFY_ENABLED"), default=True):
        ntfy = send_ntfy_alert(
            server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=topic,
            title="NERO Strategy TEST Lab Report",
            message="Friday strategy lab report generated." if result is None else ("Friday strategy lab report emailed." if result.ok else result.message),
            priority="default",
            tags="bar_chart",
        )
        print(ntfy.message)


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default.strip()


if __name__ == "__main__":
    main()
