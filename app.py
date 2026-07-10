from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run `pip install -r requirements.txt` and then `streamlit run app.py`."
    ) from exc

from nero_app.core.ai_sentiment import analyze_news_sentiment
from nero_app.core.backtester import run_event_backtest
from nero_app.core.data_loader import load_macro_events
from nero_app.core.demo_trader import accountability_scorecard, load_demo_trades, run_demo_trader
from nero_app.core.market_data import MarketDataClient
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert
from nero_app.core.news_feed import NewsFeedClient
from nero_app.core.orchestrator import NeroOrchestrator
from nero_app.core.prediction_log import append_prediction, evaluate_prediction_log, load_prediction_log
from nero_app.core.schema import AnalysisRequest, AssetSymbol
from nero_app.core.settings import load_settings, save_settings
from nero_app.core.trade_desk import build_intraday_trade_plan


def _install_terminal_skin() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #080c12; }
        [data-testid="stSidebar"] { background: #0d131c; border-right: 1px solid #1c2531; }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17,23,34,.96), rgba(13,19,28,.96));
            border: 1px solid #1c2531;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.015);
        }
        div[data-testid="stDataFrame"] { border: 1px solid #1c2531; border-radius: 8px; }
        .trade-pill { display: inline-block; border: 1px solid #2a3443; border-radius: 8px; padding: 6px 10px; margin: 3px 6px 3px 0; background: #111722; color: #d6dde8; font-size: 12px; }
        .trade-pill strong { color: #f0b90b; }
        .nero-ticker {
            display: flex;
            align-items: center;
            gap: 10px;
            border: 1px solid rgba(240,185,11,.35);
            background: linear-gradient(90deg, rgba(240,185,11,.16), rgba(17,23,34,.8));
            color: #d6dde8;
            border-radius: 8px;
            overflow: hidden;
            margin: 10px 0 12px;
            min-height: 38px;
        }
        .nero-ticker-label {
            flex: 0 0 auto;
            background: #f0b90b;
            color: #080c12;
            font-weight: 800;
            letter-spacing: .08em;
            padding: 10px 12px;
            font-size: 12px;
        }
        .nero-ticker-track {
            white-space: nowrap;
            display: inline-block;
            padding-left: 100%;
            animation: neroTicker 38s linear infinite;
            font-size: 13px;
        }
        .nero-ticker:hover .nero-ticker-track { animation-play-state: paused; }
        .nero-ticker-item { margin-right: 42px; }
        .nero-ticker-tag { color: #f0b90b; font-weight: 700; }
        @keyframes neroTicker { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_news_ticker(news_result) -> None:
    if not news_result or not news_result.headlines:
        return
    items = []
    for item in news_result.headlines[:10]:
        tag_text = ", ".join(item.tags[:3]) if item.tags else item.source
        title = item.title.replace("<", "&lt;").replace(">", "&gt;")
        items.append(f'<span class="nero-ticker-item"><span class="nero-ticker-tag">{tag_text}</span> / {title}</span>')
    st.markdown(
        f'<div class="nero-ticker"><div class="nero-ticker-label">LIVE NEWS</div><div class="nero-ticker-track">{"".join(items)}</div></div>',
        unsafe_allow_html=True,
    )


DISCLAIMER = (
    "Project Nero is an educational research and historical probability modeling tool. "
    "It does not provide financial, investment, legal, tax, or execution advice."
)


def _read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _parse_rejection_counts(value: object) -> dict[str, int]:
    if pd.isna(value) or not str(value).strip():
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_mean_reversion_tab() -> None:
    st.subheader("Mean-Reversion Agent")
    st.caption("Standalone paper-trading forward test. Closed 1h candles only. Long-only. No real orders.")
    report = _read_csv_if_exists("reports/mean_reversion_report.csv")
    closed = _read_csv_if_exists("nero_app/data/mean_reversion/trades/closed_trades.csv")
    evaluations = _read_csv_if_exists("nero_app/data/mean_reversion/trades/evaluations.csv")
    events = _read_csv_if_exists("nero_app/data/mean_reversion/trades/trade_events.csv")
    heartbeats = _read_csv_if_exists("nero_app/data/mean_reversion/heartbeats/heartbeats.csv")
    errors = _read_csv_if_exists("nero_app/data/mean_reversion/trades/runtime_errors.csv")

    if report.empty:
        st.info("Mean-reversion report is not available locally yet. Run the GitHub workflow, then pull latest repo data to view it here.")
    else:
        combined = report[report["asset"].astype(str).str.upper() == "COMBINED"]
        row = combined.iloc[0] if not combined.empty else report.iloc[0]
        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
        col_a.metric("Trades", str(int(row.get("total_trades", 0))))
        col_b.metric("Win Rate", f"{float(row.get('win_rate', 0.0)):.0%}")
        col_c.metric("Net P/L", f"${float(row.get('net_pnl', 0.0)):,.2f}")
        col_d.metric("Expectancy", f"{float(row.get('expectancy_r', 0.0)):.2f}R")
        col_e.metric("Profit Factor", f"{float(row.get('profit_factor', 0.0)):.2f}")
        col_f.metric("Max DD", f"{float(row.get('max_drawdown', 0.0)):.2%}")
        if bool(row.get("insufficient_sample", True)):
            st.warning("Insufficient sample: wait for at least 20-30 closed trades before trusting the statistics.")
        st.dataframe(report, use_container_width=True, hide_index=True)

    st.subheader("Latest Paper Trades")
    if closed.empty:
        st.info("No closed mean-reversion trades are present in the local CSV yet.")
    else:
        visible_cols = [
            col for col in [
                "trade_id", "asset", "opened_at", "closed_at", "exit_reason", "entry_price", "target", "stop_loss",
                "exit_price", "net_pnl", "r_multiple", "equity_after", "holding_hours", "strategy_version"
            ] if col in closed.columns
        ]
        sort_col = "closed_at" if "closed_at" in closed.columns else closed.columns[0]
        st.dataframe(closed.sort_values(sort_col, ascending=False)[visible_cols].head(50), use_container_width=True, hide_index=True)

    st.subheader("Open/Recent Events")
    if events.empty:
        st.caption("No entry/exit event ledger is present locally yet.")
    else:
        sort_col = "timestamp" if "timestamp" in events.columns else events.columns[0]
        st.dataframe(events.sort_values(sort_col, ascending=False).head(50), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Rejected Setup Reasons")
        counts: dict[str, int] = {}
        if not report.empty and "rejected_setup_counts" in report.columns:
            for _, report_row in report.iterrows():
                if str(report_row.get("asset", "")).upper() == "COMBINED":
                    counts = _parse_rejection_counts(report_row.get("rejected_setup_counts"))
                    break
        if counts:
            reason_frame = pd.DataFrame([{"Reason": key, "Count": value} for key, value in counts.items()]).sort_values("Count", ascending=False)
            st.dataframe(reason_frame, use_container_width=True, hide_index=True)
        elif evaluations.empty:
            st.caption("No evaluation/rejection ledger is present locally yet.")
        else:
            st.caption("No rejected setup counts found in the report yet.")

    with col_right:
        st.subheader("Heartbeat / Errors")
        if heartbeats.empty:
            st.caption("No heartbeat records are present locally yet.")
        else:
            sort_col = "timestamp" if "timestamp" in heartbeats.columns else heartbeats.columns[0]
            st.dataframe(heartbeats.sort_values(sort_col, ascending=False).head(10), use_container_width=True, hide_index=True)
        if not errors.empty:
            st.error("Runtime/data-source errors found in local ledger.")
            sort_col = "timestamp" if "timestamp" in errors.columns else errors.columns[0]
            st.dataframe(errors.sort_values(sort_col, ascending=False).head(10), use_container_width=True, hide_index=True)

    with st.expander("Strategy Rules"):
        st.markdown(
            """
- Assets: BTCUSDT, SOLUSDT, PAXGUSDT gold proxy
- Timeframe: 1-hour fully closed candles
- Entry: RSI(14) < 35, close below lower Bollinger Band, close above MA200
- Target: frozen MA20 recorded at entry
- Stop: 1.5 x ATR(14) below entry
- Max hold: 24 hours
- Risk: 1% of virtual equity, paper trading only
            """.strip()
        )


def main() -> None:
    st.set_page_config(page_title="Project Nero", page_icon="N", layout="wide")
    _install_terminal_skin()
    st.title("Project Nero")
    st.caption("Multi-agent macro and market-structure research terminal")
    st.warning(DISCLAIMER)
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0
    if "last_run_at" not in st.session_state:
        st.session_state.last_run_at = "Not run in this browser session yet"

    macro_events = load_macro_events()
    market_client = MarketDataClient()
    news_client = NewsFeedClient()
    orchestrator = NeroOrchestrator(macro_events)
    local_settings = load_settings()

    with st.sidebar:
        st.header("Scenario")
        asset = st.selectbox("Asset", [item.value for item in AssetSymbol], index=0)
        prefer_live = True
        use_latest_news = True
        mobile_alerts_enabled = True
        twelve_data_api_key = str(local_settings.get("twelve_data_api_key", ""))
        gemini_api_key = str(local_settings.get("gemini_api_key", ""))
        smtp_host = str(local_settings.get("smtp_host", "smtp.gmail.com"))
        smtp_port = int(local_settings.get("smtp_port", 465))
        sender_email = str(local_settings.get("sender_email", ""))
        email_app_password = str(local_settings.get("email_app_password", ""))
        receiver_email = str(local_settings.get("receiver_email", ""))
        test_alert_button = False
        st.caption("Live market data: auto ON")
        st.caption("Latest news: auto ON")
        if sender_email and receiver_email and email_app_password:
            st.caption(f"Email alerts: auto ON -> {receiver_email}")
        else:
            st.warning("Email alerts need saved sender/app password settings.")
        if asset in {"GOLD", "OIL", "FDX"} and not twelve_data_api_key:
            st.warning("Requires Twelve Data key for live candles; sample data will be used until then.")
        news_result = news_client.load(asset) if use_latest_news else None
        if news_result:
            selected_title = st.selectbox("Latest headline", [item.title for item in news_result.headlines])
            headline = selected_title
            st.caption(f"News source status: {news_result.status}")
        else:
            headline = st.text_area(
                "Macro headline or thesis",
                value="Fed signals rate cuts as liquidity expectations improve for risk assets.",
                height=120,
            )
        lookback = st.slider("Technical lookback", min_value=10, max_value=90, value=30, step=5)
        run_button = st.button("Run Nero Verdict", type="primary")

    settings_to_save = {
        "prefer_live": prefer_live,
        "use_latest_news": use_latest_news,
        "twelve_data_api_key": twelve_data_api_key.strip(),
        "gemini_api_key": gemini_api_key.strip(),
        "mobile_alerts_enabled": mobile_alerts_enabled,
        "smtp_host": smtp_host.strip(),
        "smtp_port": int(smtp_port),
        "sender_email": sender_email.strip(),
        "email_app_password": email_app_password.strip(),
        "receiver_email": receiver_email.strip(),
    }
    if settings_to_save != local_settings:
        save_settings(settings_to_save)

    market_data = market_client.load(
        asset=asset,
        prefer_live=prefer_live,
        days=365,
        twelve_data_api_key=twelve_data_api_key,
    )
    intraday_data = market_client.load_intraday(
        asset=asset,
        prefer_live=prefer_live,
        interval="1h",
        candles=240,
        twelve_data_api_key=twelve_data_api_key,
    )
    price_history = market_data.prices
    sentiment_result = analyze_news_sentiment(
        news_result.headlines if news_result else [],
        asset=asset,
        gemini_api_key=gemini_api_key,
    )
    enriched_headline = (
        f"{headline} | AI news sentiment: {sentiment_result.overall_sentiment} "
        f"({sentiment_result.sentiment_score}/10). {sentiment_result.summary}"
    )
    request = AnalysisRequest(asset=AssetSymbol(asset), headline=enriched_headline, lookback_days=lookback)
    result = orchestrator.run(request, price_history)
    trade_plan = build_intraday_trade_plan(
        intraday_data.prices,
        asset=asset,
        macro_direction=result.verdict.direction,
        news_sentiment=sentiment_result.overall_sentiment,
        news_score=sentiment_result.sentiment_score,
        risk_score=result.verdict.risk_score,
    )
    backtest = run_event_backtest(result.brain.matches, price_history)
    if test_alert_button:
        alert = send_email_alert(
            smtp_host,
            int(smtp_port),
            sender_email,
            email_app_password,
            receiver_email,
            f"Nero test alert | {asset}",
            f"NERO TEST ALERT | {asset} email alerts are connected.",
        )
        if alert.ok:
            st.toast(alert.message)
        else:
            st.error(alert.message)
    if run_button:
        st.session_state.run_count += 1
        st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_prediction(result, data_source=f"{market_data.source} ({market_data.status})", prices=price_history)
        demo_summary = run_demo_trader(
            asset=asset,
            plan=trade_plan,
            prices=intraday_data.prices,
            source=f"{intraday_data.source} ({intraday_data.status})",
        )
        st.toast("Prediction saved to log")
        st.success(f"Analysis #{st.session_state.run_count} saved at {st.session_state.last_run_at}")
        if demo_summary.opened:
            st.info("Demo trader recorded the current signal for accountability tracking.")
        if demo_summary.activated:
            st.info(f"Demo trader activated {demo_summary.activated} pending setup(s).")
        if demo_summary.closed:
            st.info(f"Demo trader closed {demo_summary.closed} paper trade(s).")
        if mobile_alerts_enabled and trade_plan.action != "NO_TRADE":
            alert = send_email_alert(
                smtp_host,
                int(smtp_port),
                sender_email,
                email_app_password,
                receiver_email,
                f"Nero trade alert | {asset} | {trade_plan.action.replace('_', ' ')}",
                format_trade_alert(asset, trade_plan),
            )
            if alert.ok:
                st.toast(alert.message)
            else:
                st.error(alert.message)

    _render_news_ticker(news_result)
    status_message = f"Current scenario: {asset} | Data: {market_data.status} | Last run: {st.session_state.last_run_at}"
    if market_data.status == "live":
        st.success(status_message)
    elif market_data.status.startswith("fallback"):
        st.warning(status_message)
    else:
        st.info(status_message)

    verdict_tab, trade_tab, accountability_tab, mean_reversion_tab, structure_tab, news_tab, knowledge_tab, backtest_tab, log_tab = st.tabs(
        ["Verdict", "Trade Desk", "Accountability", "Mean Reversion", "Market Structure", "News", "Knowledge Store", "Backtest", "Prediction Log"]
    )

    with verdict_tab:
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Direction", result.verdict.direction.upper())
        col_b.metric("Confidence", f"{result.verdict.confidence:.0%}")
        col_c.metric("Risk Score", f"{result.verdict.risk_score:.0%}")
        col_d.metric("Confluence", f"{result.assessment.confluence_score:.0f}/100", result.assessment.confluence_label)
        col_e.metric("AI Sentiment", sentiment_result.overall_sentiment, f"{sentiment_result.sentiment_score}/10")
        st.subheader("Rationale")
        st.write(result.verdict.summary)
        st.subheader("Driver Breakdown")
        driver_frame = pd.DataFrame(
            {
                "Signal": ["Macro Theme", "AI News Sentiment", "Technical Confluence", "Market Regime", "Momentum", "FVG / BOS", "Liquidity Sweep"],
                "Reading": [
                    f"{result.brain.thematic_score:.2f}",
                    f"{sentiment_result.overall_sentiment} ({sentiment_result.sentiment_score}/10)",
                    f"{result.assessment.confluence_score:.0f}/100",
                    f"{result.assessment.market_regime} / {result.assessment.volatility_regime}",
                    f"{result.assessment.momentum_score:.2f}",
                    f"{result.assessment.fair_value_gap} / {result.assessment.bos_signal}",
                    result.assessment.liquidity_sweep,
                ],
                "Meaning": [
                    "Historical macro matches and asset bias",
                    f"{sentiment_result.source}: {sentiment_result.summary}",
                    result.assessment.confluence_label,
                    f"ATR about {result.assessment.atr_pct:.2f}% of price",
                    "RSI plus short-term trend pressure",
                    "Imbalance plus break-of-structure context",
                    "Failed breakout/breakdown context",
                ],
            }
        )
        st.dataframe(driver_frame, use_container_width=True, hide_index=True)
        with st.expander("Technical JSON payload"):
            st.json(result.verdict.model_dump())


    with trade_tab:
        st.subheader("Intraday Trade Desk")
        if intraday_data.status == "live":
            st.success(f"Intraday source: {intraday_data.source} ({intraday_data.status})")
        elif intraday_data.status.startswith("fallback"):
            st.warning(f"Intraday source: {intraday_data.source} ({intraday_data.status})")
        else:
            st.caption(f"Intraday source: {intraday_data.source} ({intraday_data.status})")
        st.caption("Decision support only. Nero waits for trigger confirmation and does not guarantee profit.")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Action", trade_plan.action.replace("_", " "))
        col_b.metric("Bias", trade_plan.bias)
        col_c.metric("Confidence", f"{trade_plan.confidence:.0%}")
        col_d.metric("Status", trade_plan.status)
        st.markdown(
            f'<span class="trade-pill"><strong>Entry</strong> {trade_plan.entry_price:,.2f}</span>'
            f'<span class="trade-pill"><strong>SL</strong> {trade_plan.stop_loss:,.2f}</span>'
            f'<span class="trade-pill"><strong>TP1</strong> {trade_plan.take_profit_1:,.2f}</span>'
            f'<span class="trade-pill"><strong>TP2</strong> {trade_plan.take_profit_2:,.2f}</span>'
            f'<span class="trade-pill"><strong>RR</strong> {trade_plan.risk_reward_1:.2f} / {trade_plan.risk_reward_2:.2f}</span>',
            unsafe_allow_html=True,
        )
        st.write(trade_plan.entry_trigger)
        st.info(trade_plan.invalidation)
        st.subheader("Trade Options")
        if trade_plan.action == "WAIT_LONG_TRIGGER":
            st.success("Active option: LONG setup. Entry only after trigger confirmation.")
        elif trade_plan.action == "WAIT_SHORT_TRIGGER":
            st.warning("Active option: SHORT setup. Entry only after trigger confirmation.")
        else:
            st.info("Active option: NO TRADE. Nero is protecting capital until the setup improves.")
        long_status = "Active" if trade_plan.action == "WAIT_LONG_TRIGGER" else "Blocked"
        short_status = "Active" if trade_plan.action == "WAIT_SHORT_TRIGGER" else "Blocked"
        no_trade_status = "Active" if trade_plan.action == "NO_TRADE" else "Safety fallback"
        trade_options = pd.DataFrame(
            [
                {
                    "Option": "LONG",
                    "Status": long_status,
                    "How to use": trade_plan.entry_trigger if long_status == "Active" else "Do not buy until Nero shows a long trigger.",
                    "Entry": f"{trade_plan.entry_price:,.2f}" if long_status == "Active" else "-",
                    "SL": f"{trade_plan.stop_loss:,.2f}" if long_status == "Active" else "-",
                    "TP1": f"{trade_plan.take_profit_1:,.2f}" if long_status == "Active" else "-",
                    "TP2": f"{trade_plan.take_profit_2:,.2f}" if long_status == "Active" else "-",
                    "RR": f"{trade_plan.risk_reward_1:.2f} / {trade_plan.risk_reward_2:.2f}" if long_status == "Active" else "-",
                },
                {
                    "Option": "SHORT",
                    "Status": short_status,
                    "How to use": trade_plan.entry_trigger if short_status == "Active" else "Do not short until Nero shows a short trigger.",
                    "Entry": f"{trade_plan.entry_price:,.2f}" if short_status == "Active" else "-",
                    "SL": f"{trade_plan.stop_loss:,.2f}" if short_status == "Active" else "-",
                    "TP1": f"{trade_plan.take_profit_1:,.2f}" if short_status == "Active" else "-",
                    "TP2": f"{trade_plan.take_profit_2:,.2f}" if short_status == "Active" else "-",
                    "RR": f"{trade_plan.risk_reward_1:.2f} / {trade_plan.risk_reward_2:.2f}" if short_status == "Active" else "-",
                },
                {
                    "Option": "NO TRADE",
                    "Status": no_trade_status,
                    "How to use": "Stay out if trigger does not confirm or risk feels high.",
                    "Entry": "-",
                    "SL": "-",
                    "TP1": "-",
                    "TP2": "-",
                    "RR": "-",
                },
            ]
        )
        st.dataframe(trade_options, use_container_width=True, hide_index=True)
        latest_intraday = intraday_data.prices.sort_values("date").tail(120)
        st.line_chart(latest_intraday.set_index("date")[["close"]])
        plan_frame = pd.DataFrame(
            {
                "Field": ["Action", "Bias", "Entry Trigger", "Entry", "Stop Loss", "Take Profit 1", "Take Profit 2", "RR TP1", "RR TP2", "Invalidation"],
                "Value": [
                    trade_plan.action.replace("_", " "),
                    trade_plan.bias,
                    trade_plan.entry_trigger,
                    f"{trade_plan.entry_price:,.2f}",
                    f"{trade_plan.stop_loss:,.2f}",
                    f"{trade_plan.take_profit_1:,.2f}",
                    f"{trade_plan.take_profit_2:,.2f}",
                    f"{trade_plan.risk_reward_1:.2f}",
                    f"{trade_plan.risk_reward_2:.2f}",
                    trade_plan.invalidation,
                ],
            }
        )
        st.dataframe(plan_frame, use_container_width=True, hide_index=True)
        col_alert_a, col_alert_b = st.columns(2)
        with col_alert_a:
            if st.button("Send current plan by email"):
                alert = send_email_alert(
                    smtp_host,
                    int(smtp_port),
                    sender_email,
                    email_app_password,
                    receiver_email,
                    f"Nero current plan | {asset}",
                    format_trade_alert(asset, trade_plan),
                )
                if alert.ok:
                    st.success(alert.message)
                else:
                    st.error(alert.message)
        with col_alert_b:
            if mobile_alerts_enabled:
                st.success("Email auto-alerts enabled")
            else:
                st.caption("Email auto-alerts disabled")
        st.subheader("Reasoning Stack")
        st.dataframe(pd.DataFrame({"Reason": trade_plan.reasons}), use_container_width=True, hide_index=True)

    with accountability_tab:
        st.subheader("Nero Self-Accountability")
        st.caption("Read-only paper trading ledger. Auto demo trading runs on GitHub Actions every 15 minutes.")
        st.success("Auto demo trading: ON via GitHub Actions. No manual button required.")
        demo_frame = load_demo_trades()
        scorecard = accountability_scorecard(demo_frame)
        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
        col_a.metric("Total Paper Trades", str(scorecard.get("total", 0)))
        col_b.metric("Pending", str(scorecard.get("pending", 0)))
        col_c.metric("Open", str(scorecard.get("open", 0)))
        col_d.metric("Closed", str(scorecard.get("closed", 0)))
        col_e.metric("Win Rate", f"{float(scorecard.get('win_rate', 0.0)):.0%}")
        col_f.metric("Expectancy", f"{float(scorecard.get('expectancy_r', 0.0)):.2f}R")
        if demo_frame.empty:
            st.info("No demo trades yet. Nero records each LONG/SHORT signal as pending, then activates it when the trigger is touched.")
        else:
            st.dataframe(demo_frame.sort_values("opened_at", ascending=False), use_container_width=True)

    with mean_reversion_tab:
        _render_mean_reversion_tab()

    with structure_tab:
        if market_data.status == "live":
            st.success(f"Data source: {market_data.source} ({market_data.status})")
        elif market_data.status.startswith("fallback"):
            st.warning(f"Data source: {market_data.source} ({market_data.status})")
        else:
            st.caption(f"Data source: {market_data.source} ({market_data.status})")
        latest = price_history.sort_values("date").tail(2)
        last_close = float(latest.iloc[-1]["close"])
        previous_close = float(latest.iloc[-2]["close"]) if len(latest) > 1 else last_close
        daily_change = (last_close - previous_close) / previous_close if previous_close else 0.0
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Last Close", f"{last_close:,.2f}", f"{daily_change:.2%}")
        col_b.metric("RSI", f"{result.assessment.rsi:.2f}")
        col_c.metric("Regime", result.assessment.market_regime, result.assessment.volatility_regime)
        col_d.metric("ATR", f"{result.assessment.atr_pct:.2f}%")
        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("MACD", result.assessment.macd_signal.upper())
        col_f.metric("MA Alignment", result.assessment.ma_alignment.upper())
        col_g.metric("BOS", result.assessment.bos_signal.upper())
        col_h.metric("Tech Bias", f"{result.assessment.technical_bias_score:.2f}")
        st.subheader("Recent Price History")
        chart_data = price_history.tail(120).set_index("date")[["close"]]
        st.line_chart(chart_data)
        st.subheader("Latest Candles")
        st.dataframe(price_history.tail(8).sort_values("date", ascending=False), use_container_width=True)
        st.subheader("Assessment Signals")
        st.dataframe(pd.DataFrame([result.assessment.model_dump()]), use_container_width=True)

    with news_tab:
        st.subheader("Latest Macro Headlines")
        if news_result:
            st.caption(f"News status: {news_result.status}")
            st.metric("AI News Sentiment", sentiment_result.overall_sentiment, f"{sentiment_result.sentiment_score}/10")
            st.write(sentiment_result.summary)
            news_frame = pd.DataFrame([item.__dict__ for item in news_result.headlines])
            if "tags" in news_frame.columns:
                news_frame["tags"] = news_frame["tags"].apply(lambda tags: ", ".join(tags))
            st.dataframe(news_frame, use_container_width=True)
        else:
            st.info("Turn on Use latest news in the sidebar to load RSS headlines.")

    with knowledge_tab:
        st.subheader("Closest Historical Macro Matches")
        st.dataframe(pd.DataFrame([match.model_dump() for match in result.brain.matches]), use_container_width=True)

    with backtest_tab:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Average Forward Return", f"{backtest.average_forward_return:.2%}")
        col_b.metric("Win Rate", f"{backtest.win_rate:.0%}")
        col_c.metric("Samples", str(backtest.sample_count))
        st.dataframe(pd.DataFrame(backtest.trades), use_container_width=True)

    with log_tab:
        st.subheader("Saved Predictions")
        if st.button("Evaluate Pending Predictions"):
            evaluate_prediction_log(price_history)
            st.success("Prediction outcomes refreshed against the current price history.")
        log_frame = load_prediction_log()
        if log_frame.empty:
            st.info("No saved predictions yet. Press Run Nero Verdict to save the current analysis.")
        else:
            evaluated = log_frame[log_frame["evaluation_status"] == "evaluated"]
            wins = int((evaluated["outcome"] == "win").sum()) if not evaluated.empty else 0
            win_rate = wins / len(evaluated) if len(evaluated) else 0.0
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Saved Rows", str(len(log_frame)))
            col_b.metric("Evaluated", str(len(evaluated)))
            col_c.metric("Win Rate", f"{win_rate:.0%}")
            col_d.metric("Latest Direction", str(log_frame.iloc[-1]["direction"]).upper())
            st.dataframe(log_frame.sort_values("timestamp", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
