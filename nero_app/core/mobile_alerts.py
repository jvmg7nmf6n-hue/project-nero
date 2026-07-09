from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
import smtplib

import requests


from nero_app.core.trade_desk import IntradayTradePlan


@dataclass(frozen=True)
class AlertResult:
    ok: bool
    message: str


def format_trade_alert(asset: str, plan: IntradayTradePlan) -> str:
    return "\n".join(
        [
            f"NERO ALERT | {asset}",
            f"Action: {plan.action.replace('_', ' ')}",
            f"Bias: {plan.bias} | Confidence: {plan.confidence:.0%}",
            f"Entry: {plan.entry_price:,.2f}",
            f"SL: {plan.stop_loss:,.2f}",
            f"TP1: {plan.take_profit_1:,.2f}",
            f"TP2: {plan.take_profit_2:,.2f}",
            f"RR: {plan.risk_reward_1:.2f} / {plan.risk_reward_2:.2f}",
            f"Trigger: {plan.entry_trigger}",
            f"Invalidation: {plan.invalidation}",
            "Decision support only. Final trade decision is yours.",
        ]
    )


def send_email_alert(
    smtp_host: str,
    smtp_port: int,
    sender_email: str,
    app_password: str,
    receiver_email: str,
    subject: str,
    message: str,
    timeout_seconds: int = 12,
) -> AlertResult:
    host = smtp_host.strip()
    sender = sender_email.strip()
    receiver = receiver_email.strip()
    password = app_password.replace(" ", "").strip()
    if not host or not sender or not receiver or not password:
        return AlertResult(ok=False, message="SMTP host, sender email, app password, and receiver email are required.")
    if "@" not in sender or "@" not in receiver:
        return AlertResult(ok=False, message="Sender email and receiver email must be valid email addresses.")

    email = EmailMessage()
    email["From"] = sender
    email["To"] = receiver
    email["Subject"] = subject
    email.set_content(message)

    try:
        with smtplib.SMTP_SSL(host, int(smtp_port), timeout=timeout_seconds) as server:
            server.login(sender, password)
            server.send_message(email)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        return AlertResult(ok=False, message=f"Email alert failed: {exc.__class__.__name__}")

    return AlertResult(ok=True, message="Email alert sent.")


def send_ntfy_alert(
    server_url: str,
    topic: str,
    title: str,
    message: str,
    priority: str = "high",
    tags: str = "warning",
    timeout_seconds: int = 12,
) -> AlertResult:
    server = (server_url or "https://ntfy.sh").strip().rstrip("/")
    clean_topic = topic.strip().strip("/")
    if not clean_topic:
        return AlertResult(ok=False, message="Ntfy topic is required.")

    try:
        response = requests.post(
            f"{server}/{clean_topic}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        return AlertResult(ok=False, message=f"Ntfy alert failed: {exc.__class__.__name__}")

    return AlertResult(ok=True, message="Ntfy alert sent.")
