from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from nero_app.core.market_data import MarketDataClient
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert
from nero_app.core.trade_desk import IntradayTradePlan, build_intraday_trade_plan


STATE_PATH = Path("nero_monitor_state.json")
COOLDOWN_RUNS = 8


def main() -> None:
    state = _load_state()
    assets = [asset.strip().upper() for asset in os.getenv("NERO_MONITOR_ASSETS", "BTC,GOLD").split(",") if asset.strip()]
    market_client = MarketDataClient(timeout_seconds=15)
    sent = 0

    for asset in assets:
        market_data = market_client.load_intraday(
            asset=asset,
            prefer_live=True,
            interval=os.getenv("NERO_MONITOR_INTERVAL", "1h"),
            candles=240,
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
        )
        if market_data.status != "live":
            print(f"{asset}: skipped because source is {market_data.status}")
            continue

        macro_direction = _price_bias(market_data.prices)
        plan = build_intraday_trade_plan(
            market_data.prices,
            asset=asset,
            macro_direction=macro_direction,
            news_sentiment="Neutral",
            news_score=0,
            risk_score=0.35,
        )
        print(f"{asset}: {plan.action} confidence={plan.confidence:.0%} source={market_data.source}")
        if plan.action == "NO_TRADE":
            continue

        key = f"{asset}:{plan.action}:{round(plan.entry_price, 2)}"
        if not _can_send(state, key):
            print(f"{asset}: cooldown active for {key}")
            continue

        alert = _send_plan_email(asset, plan)
        if alert.ok:
            sent += 1
            state[key] = {"cooldown": COOLDOWN_RUNS}
            print(f"{asset}: email sent")
        else:
            print(f"{asset}: {alert.message}")

    _tick_cooldowns(state)
    _save_state(state)
    print(f"Nero monitor complete. Emails sent: {sent}")


def _price_bias(prices: pd.DataFrame) -> str:
    close = prices.sort_values("date")["close"].astype(float)
    if len(close) < 40:
        return "neutral"
    fast = close.tail(12).mean()
    slow = close.tail(36).mean()
    if fast > slow * 1.002:
        return "bullish"
    if fast < slow * 0.998:
        return "bearish"
    return "neutral"


def _send_plan_email(asset: str, plan: IntradayTradePlan):
    return send_email_alert(
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        sender_email=os.getenv("SENDER_EMAIL", ""),
        app_password=os.getenv("EMAIL_APP_PASSWORD", ""),
        receiver_email=os.getenv("RECEIVER_EMAIL", ""),
        subject=f"Nero trade alert | {asset} | {plan.action.replace('_', ' ')}",
        message=format_trade_alert(asset, plan),
    )


def _load_state() -> dict[str, dict[str, int]]:
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state: dict[str, dict[str, int]]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _can_send(state: dict[str, dict[str, int]], key: str) -> bool:
    record = state.get(key, {})
    return int(record.get("cooldown", 0)) <= 0


def _tick_cooldowns(state: dict[str, dict[str, int]]) -> None:
    expired = []
    for key, record in state.items():
        record["cooldown"] = max(0, int(record.get("cooldown", 0)) - 1)
        if record["cooldown"] <= 0:
            expired.append(key)
    for key in expired:
        state.pop(key, None)


if __name__ == "__main__":
    main()
