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
from nero_app.core.consensus_engine import build_consensus_decision
from nero_app.core.data_loader import load_macro_events
from nero_app.core.demo_trader import accountability_scorecard, load_demo_trades, run_demo_trader
from nero_app.core.etf_flow_intelligence import fetch_etf_flow_score
from nero_app.core.gold_real_yield import fetch_gold_real_yield_score
from nero_app.core.historical_market_memory import (
    format_regime_report,
    infer_environment_tags,
    load_historical_events,
    score_regime_similarity,
)
from nero_app.core.market_data import MarketDataClient
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert
from nero_app.core.news_feed import NewsFeedClient
from nero_app.core.orchestrator import NeroOrchestrator
from nero_app.core.prediction_log import append_prediction, build_prediction_truth_report, evaluate_prediction_log, load_prediction_log
from nero_app.core.quant_intelligence import build_cointegration_report, build_cross_asset_driver_report, build_garch_volatility_report, build_granger_causality_report, build_kalman_beta_report, build_lead_lag_driver_report, build_quant_consensus_report, build_quant_snapshot, fetch_cross_asset_price_data, quant_driver_rows
from nero_app.core.schema import AnalysisRequest, AssetSymbol
from nero_app.core.settings import load_settings, save_settings
from nero_app.core.strategy_performance_auditor import DEFAULT_CLOSED_TRADES_PATH, DEFAULT_EVALUATIONS_PATH, DEFAULT_MEAN_REVERSION_REPORT_PATH, DEFAULT_PREDICTION_LOG_PATH, build_strategy_performance_audit
from nero_app.core.social_intelligence import (
    build_social_reliability_report,
    filter_watchlist_for_asset,
    load_social_call_ledger,
    load_social_watchlist,
    summarize_social_intel,
)
from nero_app.core.trade_desk import build_intraday_trade_plan
from nero_app.core.trade_opportunity_scanner import PaperTradeState, ScannerInputs, TechnicalSnapshot, scan_trade_opportunity
from nero_app.core.trade_readiness import ReadinessInputs, build_trade_readiness_report
from nero_app.core.verdict_modifiers import apply_white_house_modifier


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


def _render_social_intel_tab(asset: str) -> None:
    st.subheader("Social Intelligence")
    st.caption("Curated X/market voice watchlist plus call accountability ledger. Context only, not a direct trade command.")
    watchlist = load_social_watchlist()
    ledger = load_social_call_ledger()
    summary = summarize_social_intel(asset, watchlist)
    filtered = filter_watchlist_for_asset(asset, watchlist)
    reliability = build_social_reliability_report(ledger)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tracked Voices", str(summary.tracked_voices))
    col_b.metric("Avg Starting Reliability", f"{summary.average_reliability:.1f}/100")
    audited_calls = int((ledger.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "evaluated").sum()) if not ledger.empty else 0
    col_c.metric("Audited Calls", str(audited_calls))
    st.write(summary.note)

    if summary.dominant_styles:
        st.caption("Dominant styles: " + ", ".join(summary.dominant_styles))
    if summary.caution_flags:
        st.warning("Caution flags: " + ", ".join(summary.caution_flags[:5]))

    st.subheader("Voice Watchlist")
    if filtered.empty:
        st.info("No social voices mapped to this asset yet.")
    else:
        visible_cols = [
            col for col in [
                "name", "handle", "platform", "asset_focus", "style", "role",
                "starting_reliability", "risk_flags", "notes"
            ] if col in filtered.columns
        ]
        st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)

    st.subheader("Guru Reliability Report")
    if reliability.empty:
        st.info("No evaluated social calls yet. Add calls to social_call_ledger.csv, then evaluate with price data.")
    else:
        st.dataframe(reliability, use_container_width=True, hide_index=True)

    st.subheader("Social Call Ledger")
    if ledger.empty:
        st.caption("No calls recorded yet.")
    else:
        asset_ledger = ledger[ledger["asset"].astype(str).str.upper() == asset] if "asset" in ledger.columns else ledger
        st.dataframe(asset_ledger.tail(50), use_container_width=True, hide_index=True)

    with st.expander("How NERO finds the real guru"):
        st.markdown(
            """
- Record a public call only if it has asset, direction, timeframe, and preferably entry/stop/target.
- Evaluate the call against later price action.
- Score each voice by win rate, average R, and sample size.
- Penalize vague hype that has no clear trade plan.
- Use social consensus only as one input, never as a direct trade command.
            """.strip()
        )


def _render_market_memory_tab(asset: str, context_text: str) -> None:
    st.subheader("Historical Market Memory")
    st.caption("Current environment compared with stored BTC 120k and Gold rally regimes.")
    if asset not in {"BTC", "GOLD"}:
        st.info("Historical Market Memory is currently calibrated for BTC and GOLD.")
        return

    events = load_historical_events()
    tags = infer_environment_tags(asset=asset, news_text=context_text)
    memory_result = score_regime_similarity(asset, tags, events)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regime Similarity", f"{memory_result.score:.0f}/100")
    col_b.metric("Reference", memory_result.reference_regime.replace("_", " "))
    col_c.metric("Memory Events", str(memory_result.matched_events))
    st.write(memory_result.verdict)

    factor_rows = []
    factor_rows.extend({"Type": "Supportive", "Factor": factor} for factor in memory_result.supportive_factors)
    factor_rows.extend({"Type": "Missing", "Factor": factor} for factor in memory_result.missing_factors)
    factor_rows.extend({"Type": "Risk", "Factor": factor} for factor in memory_result.risk_factors)
    if factor_rows:
        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No historical memory factors detected in the current context.")

    with st.expander("Market memory report"):
        st.code(format_regime_report(memory_result), language="text")

    st.subheader("Historical Memory Events")
    if events.empty:
        st.info("Historical event memory CSV is empty or unavailable.")
    else:
        filtered = events[events.get("asset_focus", pd.Series(dtype=str)).astype(str).str.upper().eq(asset)]
        if filtered.empty:
            filtered = events[events.get("reference_regime", pd.Series(dtype=str)).astype(str).str.contains(asset, case=False, na=False)]
        visible_cols = [
            col for col in ["date", "reference_regime", "event_type", "headline", "macro_tags", "impact_score", "confidence"]
            if col in filtered.columns
        ]
        st.dataframe(filtered[visible_cols].head(20), use_container_width=True, hide_index=True)

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


def _render_strategy_audit_tab(compact: bool = False) -> None:
    st.subheader("Strategy Performance Auditor")
    st.caption("Proof layer: paper-trading outcomes, prediction truth, rejection reasons, and sample quality.")
    audit = build_strategy_performance_audit()
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Audit Score", f"{audit.score:.0f}/100")
    col_b.metric("Grade", audit.grade)
    col_c.metric("Closed Trades", str(audit.total_closed_trades))
    col_d.metric("Evaluated Signals", str(audit.evaluated_signals))
    col_e.metric("Best Asset", audit.best_asset.upper())
    st.metric("Top Setup Blocker", audit.top_blocker)
    if audit.insufficient_sample:
        st.warning("Insufficient sample: collect at least 20-30 closed trades/signals before treating results as reliable evidence.")
    for note in audit.notes:
        st.info(note)
    if compact:
        with st.expander("Open full strategy audit rows"):
            _render_strategy_audit_rows(audit)
    else:
        _render_strategy_audit_rows(audit)
    with st.expander("Strategy audit data sources"):
        source_rows = [
            {"File": "mean_reversion_report.csv", "Path": str(DEFAULT_MEAN_REVERSION_REPORT_PATH), "Exists": DEFAULT_MEAN_REVERSION_REPORT_PATH.exists()},
            {"File": "closed_trades.csv", "Path": str(DEFAULT_CLOSED_TRADES_PATH), "Exists": DEFAULT_CLOSED_TRADES_PATH.exists()},
            {"File": "evaluations.csv", "Path": str(DEFAULT_EVALUATIONS_PATH), "Exists": DEFAULT_EVALUATIONS_PATH.exists()},
            {"File": "prediction_log.csv", "Path": str(DEFAULT_PREDICTION_LOG_PATH), "Exists": DEFAULT_PREDICTION_LOG_PATH.exists()},
        ]
        st.table(pd.DataFrame(source_rows))


def _render_strategy_audit_rows(audit) -> None:
    if not audit.rows:
        st.error("Strategy audit data files were found, but no readable audit rows were produced.")
        return
    audit_frame = pd.DataFrame(audit.rows)
    st.table(audit_frame)
    with st.expander("Raw audit payload"):
        st.json(audit.as_dict())

def _scanner_sentiment_score(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if -10 <= value <= 10:
        return (value + 10.0) * 5.0
    return value


def _scanner_technical_snapshot(snapshot, garch_report, price_history: pd.DataFrame) -> TechnicalSnapshot:
    if snapshot.trend_20d > 0 and snapshot.trend_60d > 0:
        trend = "UP"
    elif snapshot.trend_20d < 0 and snapshot.trend_60d < 0:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"
    regime_map = {"VOL_COMPRESSED": "LOW", "VOL_NORMAL": "NORMAL", "VOL_ELEVATED": "HIGH", "VOL_STRESS": "EXTREME"}
    close = pd.to_numeric(price_history.get("close", pd.Series(dtype=float)), errors="coerce").dropna() if not price_history.empty else pd.Series(dtype=float)
    above_ma20 = None
    above_ma200 = None
    if len(close) >= 20:
        above_ma20 = bool(close.iloc[-1] > close.rolling(20).mean().iloc[-1])
    if len(close) >= 200:
        above_ma200 = bool(close.iloc[-1] > close.rolling(200).mean().iloc[-1])
    return TechnicalSnapshot(
        trend=trend,
        rsi=None,
        zscore=snapshot.zscore_20,
        volatility_regime=regime_map.get(garch_report.regime, "NORMAL"),
        price_above_ma20=above_ma20,
        price_above_ma200=above_ma200,
    )


def _scanner_paper_trade_state(asset: str) -> PaperTradeState:
    frame = load_demo_trades()
    if frame.empty:
        return PaperTradeState(asset=asset)
    active = frame[(frame["asset"].astype(str).str.upper() == asset.upper()) & (frame["status"].astype(str).isin(["pending", "open"]))]
    return PaperTradeState(
        has_open_position=bool((active["status"].astype(str) == "open").any()) if not active.empty else False,
        has_pending_order=bool((active["status"].astype(str) == "pending").any()) if not active.empty else False,
        asset=asset,
    )


def _render_quant_intelligence_tab(asset: str, price_history: pd.DataFrame, source: str, sentiment_score: float | None = None) -> None:
    st.subheader("Quant Intelligence")
    st.caption("Statistical layer from the Gold/BTC quant toolkit: log returns, z-score, realized volatility, risk-adjusted return, and drawdown.")
    snapshot = build_quant_snapshot(price_history, asset=asset, source=source)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Regime", snapshot.regime)
    col_b.metric("Pressure", snapshot.pressure)
    col_c.metric("20D Z", f"{snapshot.zscore_20:.2f}")
    col_d.metric("30D Vol", f"{snapshot.realized_vol_30d:.1%}")
    col_e.metric("90D Sharpe", f"{snapshot.sharpe_90d:.2f}")

    if snapshot.notes:
        for note in snapshot.notes:
            st.info(note)

    st.dataframe(pd.DataFrame(quant_driver_rows(snapshot)), use_container_width=True, hide_index=True)
    st.caption(f"Source: {snapshot.source} | Observations: {snapshot.observation_count} | Latest close: {snapshot.latest_close:,.2f}")

    garch_report = build_garch_volatility_report(price_history, asset)
    st.subheader("GARCH Volatility Engine")
    if garch_report.rows:
        vcol_a, vcol_b, vcol_c, vcol_d = st.columns(4)
        vcol_a.metric("Vol Regime", garch_report.regime)
        vcol_b.metric("Conditional Vol", f"{garch_report.conditional_vol:.1%}")
        vcol_c.metric("Vol Ratio", f"{garch_report.vol_ratio:.2f}x")
        vcol_d.metric("Shock Score", f"{garch_report.shock_score:.0f}/100")
        for note in garch_report.notes:
            st.info(note)
        st.dataframe(pd.DataFrame(garch_report.rows), use_container_width=True, hide_index=True)
    else:
        for note in garch_report.notes:
            st.caption(note)
    local_consensus = build_quant_consensus_report(snapshot, garch_report)
    st.subheader("Quant Consensus Score")
    qcol_a, qcol_b, qcol_c = st.columns(3)
    qcol_a.metric("Quant Score", f"{local_consensus.score:.0f}/100")
    qcol_b.metric("Quant Label", local_consensus.label)
    qcol_c.metric("Bias", local_consensus.bias)
    for note in local_consensus.notes:
        st.info(note)
    st.dataframe(pd.DataFrame(local_consensus.rows), use_container_width=True, hide_index=True)
    st.subheader("Trade Opportunity Scanner")
    st.caption("Explains why NERO should trade, wait, or block risk using quant, sentiment, ETF-flow/real-yield, and paper-trade state.")
    external_score = None
    external_label = "not loaded"
    external_rows: list[dict[str, object]] = []
    external_notes: list[str] = []
    if asset == "BTC":
        if st.button("Refresh ETF flow", key="refresh_etf_flow_proxy"):
            etf_report = fetch_etf_flow_score()
            external_score = etf_report.etf_flow_score if etf_report.etf_flow_label != "DATA_INSUFFICIENT" else None
            external_label = etf_report.etf_flow_label
            external_rows = etf_report.evidence_frame().to_dict("records")
            external_notes = etf_report.notes
            ecol_a, ecol_b, ecol_c = st.columns(3)
            ecol_a.metric("ETF Flow Score", f"{etf_report.etf_flow_score:.0f}/100")
            ecol_b.metric("ETF Label", etf_report.etf_flow_label)
            ecol_c.metric("Dominant ETF", etf_report.dominant_etf or "none")
            st.caption("ETF flow source: " + ("actual net-flow CSV/API" if not etf_report.is_proxy else "price/volume proxy fallback"))
    elif asset == "GOLD":
        if st.button("Refresh Gold real-yield proxy", key="refresh_gold_real_yield_proxy"):
            real_yield_report = fetch_gold_real_yield_score()
            external_score = real_yield_report.real_yield_score if real_yield_report.real_yield_label != "DATA_INSUFFICIENT" else None
            external_label = real_yield_report.real_yield_label
            external_rows = [real_yield_report.as_dict()]
            external_notes = real_yield_report.notes
            rcol_a, rcol_b, rcol_c, rcol_d = st.columns(4)
            rcol_a.metric("Real Yield Score", f"{real_yield_report.real_yield_score:.0f}/100")
            rcol_b.metric("Macro Label", real_yield_report.real_yield_label)
            rcol_c.metric("Est. Real Yield", "n/a" if real_yield_report.estimated_real_yield is None else f"{real_yield_report.estimated_real_yield:.2f}%")
            rcol_d.metric("DXY Pressure", real_yield_report.dxy_pressure or "unknown")
    else:
        st.caption("ETF flow is BTC-specific and real-yield scoring is Gold-specific; scanner will use local quant evidence only for this asset.")

    if external_notes:
        for note in external_notes:
            st.info(str(note))
    if external_rows:
        with st.expander(f"External evidence: {external_label}"):
            st.dataframe(pd.DataFrame(external_rows), use_container_width=True, hide_index=True)

    scanner_inputs = ScannerInputs(
        asset=asset,
        quant_consensus_score=local_consensus.score,
        sentiment_score=_scanner_sentiment_score(sentiment_score),
        etf_flow_score=external_score if asset == "BTC" else None,
        real_yield_score=external_score if asset == "GOLD" else None,
        sharpe_90d=snapshot.sharpe_90d,
        technical=_scanner_technical_snapshot(snapshot, garch_report, price_history),
        paper_trade_state=_scanner_paper_trade_state(asset),
    )
    scanner = scan_trade_opportunity(scanner_inputs)
    scol_a, scol_b, scol_c = st.columns(3)
    scol_a.metric("Opportunity Score", f"{scanner.opportunity_score:.0f}/100")
    scol_b.metric("Decision", scanner.decision)
    scol_c.metric("Direction Bias", scanner.direction_bias)
    pass_col, fail_col, near_col = st.columns(3)
    with pass_col:
        st.markdown("**Passed**")
        for item in scanner.passed_conditions or ["none"]:
            st.caption(item)
    with fail_col:
        st.markdown("**Failed**")
        for item in scanner.failed_conditions or ["none"]:
            st.caption(item)
    with near_col:
        st.markdown("**Near Miss**")
        for item in scanner.near_miss_conditions or ["none"]:
            st.caption(item)
    if scanner.blocker_reason:
        st.warning(scanner.blocker_reason)
    st.code(scanner.explanation, language=None)

    readiness = build_trade_readiness_report(
        ReadinessInputs(
            asset=asset,
            opportunity_decision=scanner.decision,
            opportunity_score=scanner.opportunity_score,
            quant_score=local_consensus.score,
            volatility_regime=garch_report.regime,
            sentiment_score=_scanner_sentiment_score(sentiment_score),
            has_active_paper_trade=scanner_inputs.paper_trade_state.has_open_position or scanner_inputs.paper_trade_state.has_pending_order,
            missing_inputs=[] if sentiment_score is not None else ["news sentiment"],
        )
    )
    st.subheader("Trade Readiness Engine")
    rcol_a, rcol_b, rcol_c = st.columns(3)
    rcol_a.metric("Readiness Score", f"{readiness.readiness_score:.0f}/100")
    rcol_b.metric("Readiness", readiness.label)
    rcol_c.metric("Blockers", str(len(readiness.blockers)))
    st.info(readiness.action)
    if readiness.blockers:
        st.warning("; ".join(readiness.blockers))
    for reason in readiness.reasons:
        st.info(reason)
    if readiness.missing_inputs:
        st.caption("Missing inputs: " + ", ".join(readiness.missing_inputs))

    st.subheader("Cross-Asset Driver Matrix")
    st.caption("Optional live driver map: DXY, SPX/Nasdaq, ETF proxies, MSTR/COIN/miners for BTC; DXY/yields/miners/ETF proxies for Gold.")
    if asset in {"BTC", "GOLD"}:
        if st.button("Refresh cross-asset drivers"):
            driver_prices, driver_source = fetch_cross_asset_price_data(asset)
            driver_report = build_cross_asset_driver_report(asset, driver_prices)
            if driver_report.rows:
                dcol_a, dcol_b = st.columns(2)
                dcol_a.metric("Strongest Driver", driver_report.strongest_driver.upper())
                dcol_b.metric("60D Correlation", f"{driver_report.strongest_correlation:.2f}")
                for note in driver_report.notes:
                    st.info(note)
                st.dataframe(pd.DataFrame(driver_report.rows), use_container_width=True, hide_index=True)
                lead_lag_report = build_lead_lag_driver_report(asset, driver_prices)
                st.subheader("Lead/Lag Driver Engine")
                if lead_lag_report.rows:
                    lcol_a, lcol_b, lcol_c = st.columns(3)
                    lcol_a.metric("Strongest Leader", lead_lag_report.strongest_leader.upper())
                    lcol_b.metric("Lead Days", str(lead_lag_report.strongest_lag_days))
                    lcol_c.metric("Lead Corr", f"{lead_lag_report.strongest_lead_correlation:.2f}")
                    for note in lead_lag_report.notes:
                        st.info(note)
                    st.dataframe(pd.DataFrame(lead_lag_report.rows), use_container_width=True, hide_index=True)
                else:
                    for note in lead_lag_report.notes:
                        st.caption(note)
                cointegration_report = build_cointegration_report(asset, driver_prices)
                st.subheader("Cointegration Engine")
                if cointegration_report.rows:
                    ccol_a, ccol_b = st.columns(2)
                    ccol_a.metric("Best Long-Run Pair", cointegration_report.strongest_pair.upper())
                    ccol_b.metric("Best p-value", "n/a" if cointegration_report.strongest_pvalue is None else f"{cointegration_report.strongest_pvalue:.4f}")
                    for note in cointegration_report.notes:
                        st.info(note)
                    st.dataframe(pd.DataFrame(cointegration_report.rows), use_container_width=True, hide_index=True)
                else:
                    for note in cointegration_report.notes:
                        st.caption(note)
                kalman_report = build_kalman_beta_report(asset, driver_prices)
                st.subheader("Kalman Dynamic Beta Engine")
                if kalman_report.rows:
                    kcol_a, kcol_b, kcol_c = st.columns(3)
                    kcol_a.metric("Shifting Driver", kalman_report.strongest_dynamic_driver.upper())
                    kcol_b.metric("Latest Beta", f"{kalman_report.latest_beta:.2f}")
                    kcol_c.metric("30D Change", f"{kalman_report.beta_change:+.2f}")
                    for note in kalman_report.notes:
                        st.info(note)
                    st.dataframe(pd.DataFrame(kalman_report.rows), use_container_width=True, hide_index=True)
                else:
                    for note in kalman_report.notes:
                        st.caption(note)
                granger_report = build_granger_causality_report(asset, driver_prices)
                st.subheader("Granger Causality Engine")
                if granger_report.rows:
                    gcol_a, gcol_b, gcol_c = st.columns(3)
                    gcol_a.metric("Best Predictor", granger_report.strongest_predictor.upper())
                    gcol_b.metric("Best Lag", str(granger_report.strongest_lag))
                    gcol_c.metric("Best p-value", "n/a" if granger_report.strongest_pvalue is None else f"{granger_report.strongest_pvalue:.4f}")
                    for note in granger_report.notes:
                        st.info(note)
                    st.dataframe(pd.DataFrame(granger_report.rows), use_container_width=True, hide_index=True)
                else:
                    for note in granger_report.notes:
                        st.caption(note)
                full_consensus = build_quant_consensus_report(snapshot, garch_report, driver_report, kalman_report, granger_report)
                st.subheader("Full Quant Consensus")
                fqcol_a, fqcol_b, fqcol_c = st.columns(3)
                fqcol_a.metric("Full Quant Score", f"{full_consensus.score:.0f}/100")
                fqcol_b.metric("Full Label", full_consensus.label)
                fqcol_c.metric("Full Bias", full_consensus.bias)
                for note in full_consensus.notes:
                    st.info(note)
                st.dataframe(pd.DataFrame(full_consensus.rows), use_container_width=True, hide_index=True)
                st.caption(driver_source)
            else:
                st.warning("Cross-asset driver matrix is not available yet.")
                for note in driver_report.notes:
                    st.caption(note)
                st.caption(driver_source)
    else:
        st.caption("Cross-asset driver matrix is currently calibrated for BTC and GOLD.")
    with st.expander("How to read this quant layer"):
        st.markdown(
            """
- Z-score tells whether price is stretched versus its 20-day mean.
- Realized volatility tells how dangerous the current environment is.
- Sharpe/Sortino tell whether recent returns are worth the volatility.
- Drawdown shows how deep recent downside pressure has been.
- This is not a buy/sell signal by itself; it becomes evidence for NERO's consensus brain.
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
    adjusted_verdict, white_house_impact = apply_white_house_modifier(asset, enriched_headline, result.verdict)
    result = result.model_copy(update={"verdict": adjusted_verdict})
    historical_events = load_historical_events()
    market_memory_tags = infer_environment_tags(asset=asset, news_text=enriched_headline)
    market_memory_result = score_regime_similarity(asset, market_memory_tags, historical_events) if asset in {"BTC", "GOLD"} else None
    trade_plan = build_intraday_trade_plan(
        intraday_data.prices,
        asset=asset,
        macro_direction=result.verdict.direction,
        news_sentiment=sentiment_result.overall_sentiment,
        news_score=sentiment_result.sentiment_score,
        risk_score=result.verdict.risk_score,
    )
    consensus_decision = build_consensus_decision(
        verdict=result.verdict,
        assessment=result.assessment,
        trade_plan=trade_plan,
        news_sentiment=sentiment_result.overall_sentiment,
        news_score=sentiment_result.sentiment_score,
        market_memory=market_memory_result,
        white_house_impact=white_house_impact,
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

    verdict_tab, trade_tab, accountability_tab, mean_reversion_tab, strategy_audit_tab, market_memory_tab, quant_intel_tab, social_intel_tab, structure_tab, news_tab, knowledge_tab, backtest_tab, log_tab = st.tabs(
        ["Verdict", "Trade Desk", "Accountability", "Mean Reversion", "Strategy Audit", "Market Memory", "Quant Intel", "Social Intel", "Market Structure", "News", "Knowledge Store", "Backtest", "Prediction Log"]
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
        signal_rows = ["Macro Theme", "AI News Sentiment", "Technical Confluence", "Market Regime", "Momentum", "FVG / BOS", "Liquidity Sweep"]
        reading_rows = [
            f"{result.brain.thematic_score:.2f}",
            f"{sentiment_result.overall_sentiment} ({sentiment_result.sentiment_score}/10)",
            f"{result.assessment.confluence_score:.0f}/100",
            f"{result.assessment.market_regime} / {result.assessment.volatility_regime}",
            f"{result.assessment.momentum_score:.2f}",
            f"{result.assessment.fair_value_gap} / {result.assessment.bos_signal}",
            result.assessment.liquidity_sweep,
        ]
        meaning_rows = [
            "Historical macro matches and asset bias",
            f"{sentiment_result.source}: {sentiment_result.summary}",
            result.assessment.confluence_label,
            f"ATR about {result.assessment.atr_pct:.2f}% of price",
            "RSI plus short-term trend pressure",
            "Imbalance plus break-of-structure context",
            "Failed breakout/breakdown context",
        ]
        if white_house_impact is not None:
            asset_impact = white_house_impact.btc_average_impact if asset == "BTC" else white_house_impact.gold_average_impact
            asset_direction = white_house_impact.btc_direction if asset == "BTC" else white_house_impact.gold_direction
            signal_rows.append("White House Impact")
            reading_rows.append(f"{asset_direction} ({asset_impact:.0f}/100)")
            meaning_rows.append(f"{white_house_impact.matched_events} similar event(s), confidence {white_house_impact.confidence:.0%}")
        driver_frame = pd.DataFrame({"Signal": signal_rows, "Reading": reading_rows, "Meaning": meaning_rows})
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
        st.subheader("Consensus Decision")
        con_a, con_b, con_c = st.columns(3)
        con_a.metric("Decision Class", consensus_decision.decision_class.replace("_", " "))
        con_b.metric("Trade Quality", f"{consensus_decision.trade_quality:.0f}/100")
        con_c.metric("Direction", consensus_decision.direction)
        if consensus_decision.decision_class == "NO_TRADE":
            st.info(consensus_decision.human_note)
        elif consensus_decision.decision_class == "SCALP_ONLY":
            st.warning(consensus_decision.human_note)
        else:
            st.success(consensus_decision.human_note)
        if consensus_decision.blockers:
            st.dataframe(pd.DataFrame({"Blocker": consensus_decision.blockers}), use_container_width=True, hide_index=True)
        with st.expander("Consensus reasoning"):
            st.dataframe(pd.DataFrame({"Reason": consensus_decision.reasons}), use_container_width=True, hide_index=True)

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
        st.divider()
        _render_strategy_audit_tab(compact=True)

    with mean_reversion_tab:
        _render_mean_reversion_tab()

    with market_memory_tab:
        _render_market_memory_tab(asset, enriched_headline)


    with quant_intel_tab:
        _render_quant_intelligence_tab(asset, price_history, f"{market_data.source} ({market_data.status})", sentiment_result.sentiment_score)


    with social_intel_tab:
        _render_social_intel_tab(asset)


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
        st.subheader("Signal Truth Dashboard v2")
        st.caption("NERO accountability layer: records every saved verdict, evaluates outcomes, and shows whether signals are actually working.")
        if st.button("Evaluate Pending Predictions"):
            evaluate_prediction_log(price_history, asset=asset)
            st.success("Prediction outcomes refreshed for the selected asset only.")
        log_frame = load_prediction_log()
        if log_frame.empty:
            st.info("No saved predictions yet. Press Run Nero Verdict to save the current analysis.")
        else:
            truth = build_prediction_truth_report(log_frame)
            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            col_a.metric("Saved Signals", str(truth["total"]))
            col_b.metric("Evaluated", str(truth["evaluated"]))
            col_c.metric("Pending", str(truth["pending"]))
            col_d.metric("Win Rate", f"{float(truth['win_rate']):.0%}")
            col_e.metric("Avg Return", f"{float(truth['average_return']):.2%}")
            hcol_a, hcol_b, hcol_c = st.columns(3)
            hcol_a.metric("Wins", str(truth["wins"]))
            hcol_b.metric("Misses", str(truth["misses"]))
            hcol_c.metric("High-Conf Win Rate", f"{float(truth['high_confidence_win_rate']):.0%}")
            for note in truth["notes"]:
                st.info(str(note))
            if truth["rows"]:
                st.subheader("Truth By Asset")
                st.dataframe(pd.DataFrame(truth["rows"]), use_container_width=True, hide_index=True)
            st.subheader("Prediction Ledger")
            visible_cols = [
                "timestamp", "asset", "direction", "confidence", "risk_score", "entry_date", "entry_close",
                "target_date", "evaluation_status", "exit_date", "exit_close", "actual_return", "outcome", "headline",
            ]
            visible_cols = [column for column in visible_cols if column in log_frame.columns]
            st.dataframe(log_frame.sort_values("timestamp", ascending=False)[visible_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
