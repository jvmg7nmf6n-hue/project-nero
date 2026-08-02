from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nero_app.core.market_data import MarketDataClient
from nero_app.core.demo_trader import run_demo_trader
from nero_app.core.market_scanner import DEFAULT_SCANNER_ASSETS, ScannerAlert, scan_market_activity
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert, send_ntfy_alert
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
        demo_summary = run_demo_trader(
            asset=asset,
            plan=plan,
            prices=market_data.prices,
            source=f"{market_data.source} ({market_data.status})",
        )
        print(
            f"{asset}: demo recorded={demo_summary.opened} activated={demo_summary.activated} closed={demo_summary.closed} "
            f"pending={demo_summary.pending_trades} open={demo_summary.open_trades} "
            f"win_rate={demo_summary.win_rate:.0%} expectancy={demo_summary.expectancy_r:.2f}R"
        )
        if plan.action == "NO_TRADE":
            continue

        key = f"{asset}:{plan.action}:{round(plan.entry_price, 2)}"
        if not _can_send(state, key):
            print(f"{asset}: cooldown active for {key}")
            continue

        alert = _send_plan_alert(asset, plan)
        if alert.ok:
            sent += 1
            state[key] = {"cooldown": COOLDOWN_RUNS}
            print(f"{asset}: {alert.message}")
        else:
            print(f"{asset}: {alert.message}")

    sent += _run_market_scanner(state, market_client)
    _tick_cooldowns(state)
    _save_state(state)
    print(f"Nero monitor complete. Alerts sent: {sent}")



def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_market_scanner(state: dict[str, dict[str, int]], market_client: MarketDataClient) -> int:
    if not _truthy(os.getenv("NERO_SCANNER_ALERTS_ENABLED"), default=False):
        print("scanner alerts disabled; set NERO_SCANNER_ALERTS_ENABLED=true to enable.")
        return 0
    raw_assets = os.getenv("NERO_SCANNER_ASSETS", ",".join(DEFAULT_SCANNER_ASSETS))
    assets = [asset.strip().upper() for asset in raw_assets.split(",") if asset.strip()]
    if not assets:
        return 0

    interval = os.getenv("NERO_SCANNER_INTERVAL", "30m")
    sent = 0
    for asset in assets:
        market_data = market_client.load_intraday(asset=asset, prefer_live=True, interval=interval, candles=60)
        if market_data.status != "live":
            print(f"scanner {asset}: skipped because source is {market_data.status}")
            continue
        alerts = scan_market_activity(market_data.prices, asset=asset, bar_label=interval)
        if not alerts:
            print(f"scanner {asset}: no new activity source={market_data.source}")
            continue
        for alert in alerts:
            key = f"scanner:{alert.asset}:{alert.event_type}"
            if not _can_send(state, key):
                print(f"scanner {asset}: cooldown active for {alert.event_type}")
                continue
            result = _send_scanner_alert(alert)
            if result.ok:
                state[key] = {"cooldown": COOLDOWN_RUNS}
                sent += 1
            print(f"scanner {asset}: {alert.event_type} {result.message}")
    return sent


def _send_scanner_alert(alert: ScannerAlert):
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if ntfy_topic:
        ntfy = send_ntfy_alert(
            server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=ntfy_topic,
            title=f"Nero scanner | {alert.title}",
            message=alert.message,
            priority=alert.priority,
            tags=alert.tags,
        )
        if ntfy.ok:
            return ntfy
        print(f"scanner {alert.asset}: {ntfy.message}; trying email backup")

    return send_email_alert(
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        sender_email=os.getenv("SENDER_EMAIL", ""),
        app_password=os.getenv("EMAIL_APP_PASSWORD", ""),
        receiver_email=os.getenv("RECEIVER_EMAIL", ""),
        subject=f"Nero scanner | {alert.title}",
        message=alert.message,
    )

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


def _send_plan_alert(asset: str, plan: IntradayTradePlan):
    subject = f"Nero trade alert | {asset} | {plan.action.replace('_', ' ')}"
    message = format_trade_alert(asset, plan)
    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if ntfy_topic:
        ntfy = send_ntfy_alert(
            server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=ntfy_topic,
            title=subject,
            message=message,
            priority="high",
            tags="chart_with_upwards_trend",
        )
        if ntfy.ok:
            return ntfy
        print(f"{asset}: {ntfy.message}; trying email backup")

    return send_email_alert(
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        sender_email=os.getenv("SENDER_EMAIL", ""),
        app_password=os.getenv("EMAIL_APP_PASSWORD", ""),
        receiver_email=os.getenv("RECEIVER_EMAIL", ""),
        subject=subject,
        message=message,
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
