from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import streamlit.components.v1 as components

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run `pip install -r requirements.txt` and then `streamlit run app.py`."
    ) from exc

from nero_app.core.ai_sentiment import analyze_news_sentiment
from nero_app.core.backtester import run_event_backtest
from nero_app.core.btc_structural_models import build_btc_structural_report
from nero_app.core.consensus_engine import build_consensus_decision
from nero_app.core.cycle_intelligence import build_cycle_intelligence_report, cycle_dashboard_rows
from nero_app.core.data_loader import load_macro_events
from nero_app.core.demo_trader import accountability_scorecard, load_demo_trades, run_demo_trader
from nero_app.core.etf_flow_intelligence import fetch_etf_flow_score
from nero_app.core.gold_real_yield import fetch_gold_real_yield_score
from nero_app.core.hypothesis_quality_gate import build_hypothesis_quality_gate
from nero_app.core.historical_market_memory import (
    format_regime_report,
    infer_environment_tags,
    load_historical_events,
    score_regime_similarity,
)
from nero_app.core.market_data import MarketDataClient
from nero_app.core.mobile_alerts import format_trade_alert, send_email_alert
from nero_app.core.news_feed import NewsFeedClient
from nero_app.core.nero_chat import NeroChatContext, SUGGESTED_QUESTIONS, answer_nero_chat
from nero_app.core.orchestrator import NeroOrchestrator
from nero_app.core.prediction_log import append_prediction, build_prediction_truth_report, evaluate_prediction_log, load_prediction_log
from nero_app.core.quant_intelligence import build_cointegration_report, build_cross_asset_driver_report, build_garch_volatility_report, build_granger_causality_report, build_kalman_beta_report, build_lead_lag_driver_report, build_quant_consensus_report, build_quant_snapshot, fetch_cross_asset_price_data, quant_driver_rows
from nero_app.core.schema import AnalysisRequest, AssetSymbol
from nero_app.core.settings import load_settings, save_settings
from nero_app.core.strategy_performance_auditor import DEFAULT_CLOSED_TRADES_PATH, DEFAULT_EVALUATIONS_PATH, DEFAULT_MEAN_REVERSION_REPORT_PATH, DEFAULT_PREDICTION_LOG_PATH, build_strategy_performance_audit
from nero_app.core.strategy_lab_agent import CANDIDATES, DEFAULT_REPORT_DIR as STRATEGY_LAB_REPORT_DIR, write_strategy_lab_summary
from nero_app.core.strategy_evolution import build_strategy_evolution_report
from nero_app.core.strategy_research_lab import build_strategy_research_report
from nero_app.core.strategy_repair_workbench import build_strategy_repair_workbench
from nero_app.core.strategy_repair_lab import build_strategy_repair_lab_report
from nero_app.core.strategy_quarantine import build_strategy_quarantine_report
from nero_app.core.strategy_verification import build_strategy_verification_report
from nero_app.core.live_trade_status import build_live_trade_status_report
from nero_app.core.profit_edge_engine import build_profit_edge_report
from nero_app.core.sunflower_profit_bridge import build_sunflower_profit_bridge_report
from nero_app.core.social_intelligence import (
    build_social_reliability_report,
    filter_watchlist_for_asset,
    load_social_call_ledger,
    load_social_watchlist,
    summarize_social_intel,
)
from nero_app.core.trade_desk import build_intraday_trade_plan
from nero_app.core.trade_opportunity_scanner import PaperTradeState, ScannerInputs, TechnicalSnapshot, scan_trade_opportunity
from nero_app.core.trade_path import TradePathInput, build_trade_path_report
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


def _render_opening_market_deck(asset: str, market_data, intraday_data, trade_plan, sentiment_result, active_timeframe: str) -> None:
    """TradingView-style opening command deck for the selected market."""
    candles = intraday_data.prices.copy()
    if candles.empty:
        candles = market_data.prices.copy()
    if candles.empty:
        st.info("Opening market deck is waiting for candle data.")
        return

    candles = candles.sort_values("date").tail(240).copy()
    for column in ["open", "high", "low", "close", "volume"]:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
    candles = candles.dropna(subset=["open", "high", "low", "close"])
    if candles.empty:
        st.info("Opening market deck could not parse candle data.")
        return

    latest = candles.iloc[-1]
    previous = candles.iloc[-2] if len(candles) > 1 else latest
    last_close = float(latest["close"])
    prev_close = float(previous["close"])
    candle_change = last_close - prev_close
    candle_change_pct = (candle_change / prev_close * 100) if prev_close else 0.0
    session_start = float(candles.iloc[0]["close"])
    session_change_pct = ((last_close - session_start) / session_start * 100) if session_start else 0.0
    price_color = "#26a69a" if candle_change >= 0 else "#ef5350"
    spread = max(last_close * 0.00035, 0.01)
    symbol = asset if asset in {"GOLD", "OIL", "FDX"} else f"{asset}USDT"
    default_ws_symbol = symbol if symbol.endswith("USDT") else "BTCUSDT"
    source_text = f"{intraday_data.source} ({intraday_data.status})"
    initial_rows = [
        {
            "t": str(row["date"])[5:16],
            "time": str(row["date"]),
            "o": round(float(row["open"]), 6),
            "h": round(float(row["high"]), 6),
            "l": round(float(row["low"]), 6),
            "c": round(float(row["close"]), 6),
            "v": round(float(row.get("volume", 0.0) or 0.0), 3),
        }
        for _, row in candles.iterrows()
    ]
    payload = json.dumps(initial_rows)
    stat_cards = "".join(
        f'<div class="nero-card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in [
            ("OPEN", f'{float(latest["open"]):,.4f}'),
            ("HIGH", f'{float(latest["high"]):,.4f}'),
            ("LOW", f'{float(latest["low"]):,.4f}'),
            ("CLOSE", f"{last_close:,.4f}"),
            ("VOLUME", f'{float(latest.get("volume", 0.0) or 0.0):,.2f}'),
            ("SESSION", f"{session_change_pct:+.2f}%"),
        ]
    )
    html = f"""
    <div id="tvxApp" class="tvx-shell" data-theme="dark">
      <div class="tvx-topbar">
        <div class="tvx-symbolbox"><select id="tvxSymbol"><option>BTCUSDT</option><option>ETHUSDT</option><option>SOLUSDT</option><option>BNBUSDT</option><option>XRPUSDT</option><option>DOGEUSDT</option><option>NEARUSDT</option><option>PAXGUSDT</option></select><span id="tvxSource">{source_text}</span><span id="tvxConnection" class="tvx-conn">server fallback ready</span></div>
        <div id="tvxFrames" class="tvx-frames"><button data-i="1m">1m</button><button data-i="5m">5m</button><button data-i="15m">15m</button><button data-i="1h">1H</button><button data-i="4h">4H</button><button data-i="1d">1D</button><button data-i="1w">1W</button></div>
        <div class="tvx-tools"><button id="tvxIndicatorBtn">+ Indicator</button><button id="tvxCompareBtn">Compare</button><button id="tvxAlertBtn">Alert</button><button id="tvxReplayBtn">Replay</button><button id="tvxScaleBtn">Scale Auto</button><button id="tvxUndoBtn">Undo</button><button id="tvxRedoBtn">Redo</button><button id="tvxFullBtn">Fullscreen</button><button id="tvxShotBtn">Screenshot</button><button id="tvxThemeBtn">Theme</button></div>
        <div class="tvx-actions"><button class="trade">Trade</button><button class="publish">Publish</button></div>
      </div>
      <div class="tvx-readout"><div id="tvxLivePrice" class="tvx-live-price" style="color:{price_color}">{last_close:,.4f}<small>{candle_change:+,.4f} / {candle_change_pct:+.2f}%</small></div>{stat_cards}</div>
      <div class="tvx-grid">
        <div id="tvxLeftTools" class="tvx-leftbar"><button data-tool="cursor">CUR</button><button data-tool="trend">TL</button><button data-tool="hline">HL</button><button data-tool="ray">RAY</button><button data-tool="fib">FIB</button><button data-tool="box">BOX</button><button data-tool="text">TXT</button><button data-tool="brush">BR</button><button data-tool="magnet">MAG</button><button data-tool="lock">LOCK</button><button data-tool="eye">EYE</button><button data-tool="trash">DEL</button></div>
        <div class="tvx-chart-stack">
          <div class="tvx-overlay"><div class="tvx-legend"><b id="tvxEma9" style="color:#f0b90b">EMA 9</b><b id="tvxEma21" style="color:#2962ff">EMA 21</b><span id="tvxHoverText">Move mouse over candles</span></div><div class="tvx-order"><div id="tvxSell" class="sell">SELL<br><b>{last_close-spread/2:,.4f}</b></div><div id="tvxBuy" class="buy">BUY<br><b>{last_close+spread/2:,.4f}</b></div><em id="tvxSpread">spread {spread:,.4f}</em></div><div id="tvxFloatTape" class="tvx-float-tape"></div><div id="tvxTip" class="tvx-tip"></div><div id="tvxToast" class="tvx-toast"></div></div>
          <canvas id="tvxPriceCanvas"></canvas>
          <div class="tvx-subpane"><div class="tvx-sublegend">MACD close 12 26 9 <b id="tvxMacdText">loading</b></div><canvas id="tvxMacdCanvas"></canvas></div>
        </div>
        <div class="tvx-rightbar"><button>WL</button><button>AL</button><button>CHAT</button><button>NEWS</button><button>CAL</button><button>SCR</button><button>HOT</button><button>BELL</button><button>TREE</button></div>
        <div class="tvx-sidepanel"><div class="tvx-mini"><span>NERO Bias</span><b>{trade_plan.bias}</b></div><div class="tvx-mini"><span>Action</span><b>{trade_plan.action.replace('_', ' ')}</b></div><div class="tvx-mini"><span>AI Sentiment</span><b>{sentiment_result.overall_sentiment} ({sentiment_result.sentiment_score}/10)</b></div><div class="tvx-mini"><span>Risk Guard</span><b>{trade_plan.status}</b></div><label class="tvx-qty">Order Type<select id="tvxOrderType"><option>Market</option><option>Limit</option></select></label><label class="tvx-qty">Risk %<input id="tvxRiskPct" type="number" value="1" min="0" step="0.1"></label><label class="tvx-qty">Qty<input id="tvxQty" type="number" value="1" min="0" step="0.01"></label><button id="tvxSubmitOrder">Submit simulated order</button><div id="tvxOrders" class="tvx-orders"><b>Open orders</b></div></div>
      </div>
      <div class="tvx-bottombar"><span>1D</span><span>5D</span><span>1M</span><span>3M</span><span>6M</span><span>YTD</span><span>1Y</span><span>5Y</span><span>All</span><div id="tvxReplay" class="tvx-replay"><button id="tvxBack">Step</button><button id="tvxPlay">Play</button><button id="tvxFwd">Next</button><select id="tvxSpeed"><option>1x</option><option>2x</option><option>4x</option></select></div><b id="tvxClock">UTC --:--:--</b><em>Auto scale A | Log L | Expand</em></div>
      <div id="tvxIndicatorModal" class="tvx-modal"><div><h3>Add Indicator</h3><input id="tvxIndicatorSearch" placeholder="Search MA, EMA, RSI, MACD, BB, Volume, Stochastic, ATR"><label><input type="checkbox" checked data-ind="ema9"> EMA 9</label><label><input type="checkbox" checked data-ind="ema21"> EMA 21</label><label><input type="checkbox" checked data-ind="macd"> MACD</label><label><input type="checkbox" data-ind="rsi"> RSI 14</label><label><input type="checkbox" data-ind="volume"> Volume</label><button id="tvxCloseIndicators">Close</button></div></div>
      <div id="tvxAlertModal" class="tvx-modal"><div><h3>Price Alert</h3><input id="tvxAlertPrice" type="number" step="0.01" placeholder="Alert price"><button id="tvxSaveAlert">Save alert</button><button id="tvxCloseAlert">Close</button></div></div>
    </div>
    <style>
      * {{ box-sizing:border-box; }} .tvx-shell {{ font-family:Inter,Arial,sans-serif; color:#d6dde8; background:#0b0f16; border:1px solid #1c2531; border-radius:10px; overflow:hidden; box-shadow:0 24px 70px rgba(0,0,0,.32); }} .tvx-shell[data-theme="light"] {{ background:#f6f8fb; color:#172033; }}
      .tvx-topbar,.tvx-readout,.tvx-bottombar {{ display:flex; align-items:center; gap:8px; border-bottom:1px solid #1c2531; background:#101722; padding:8px 10px; }} .tvx-shell[data-theme="light"] .tvx-topbar,.tvx-shell[data-theme="light"] .tvx-readout,.tvx-shell[data-theme="light"] .tvx-bottombar {{ background:#ffffff; }}
      .tvx-symbolbox {{ min-width:220px; }} .tvx-symbolbox select {{ background:#0d131c; color:#f4f7fb; border:1px solid #263244; border-radius:7px; padding:7px 9px; font-weight:900; }} .tvx-symbolbox span {{ display:block; color:#8fa1b7; font-size:10px; max-width:360px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:3px; }} .tvx-conn {{ color:#f0b90b!important; }}
      .tvx-frames,.tvx-tools,.tvx-actions,.tvx-replay {{ display:flex; gap:5px; flex-wrap:wrap; }} button,.tvx-frames button,.tvx-tools button,.tvx-actions button,.tvx-leftbar button,.tvx-rightbar button {{ background:#0d131c; color:#9eb0c5; border:1px solid #263244; border-radius:6px; padding:5px 8px; font-size:11px; transition:all .15s; }} button:hover {{ border-color:#f0b90b; color:#f0b90b; }} .tvx-frames .active,.tvx-leftbar .active {{ background:#f0b90b; color:#080c12; border-color:#f0b90b; font-weight:900; }} .tvx-actions {{ margin-left:auto; }} .tvx-actions .trade {{ background:#2962ff; color:white; border-color:#2962ff; }} .tvx-actions .publish {{ background:#26a69a; color:white; border-color:#26a69a; }}
      .tvx-live-price {{ min-width:185px; font-size:25px; font-weight:900; font-variant-numeric:tabular-nums; }} .tvx-live-price small {{ display:block; font-size:11px; }} .nero-card {{ background:#101722; border:1px solid #1c2531; border-radius:8px; padding:7px 10px; min-width:105px; }} .nero-card span {{ display:block; color:#8fa1b7; font-size:9px; font-weight:900; }} .nero-card strong {{ color:#f4f7fb; font-size:14px; font-variant-numeric:tabular-nums; }}
      .tvx-grid {{ display:grid; grid-template-columns:44px minmax(0,1fr) 40px 190px; min-height:1000px; }} .tvx-leftbar,.tvx-rightbar {{ display:flex; flex-direction:column; gap:6px; padding:8px 6px; background:#0d131c; border-right:1px solid #1c2531; }} .tvx-rightbar {{ border-right:0; border-left:1px solid #1c2531; }} .tvx-leftbar button,.tvx-rightbar button {{ padding:7px 4px; font-size:9px; }}
      .tvx-chart-stack {{ position:relative; min-height:1000px; background:#080c12; }} #tvxPriceCanvas {{ width:100%; height:780px; display:block; }} .tvx-subpane {{ height:220px; border-top:1px solid #1c2531; position:relative; background:#090e15; }} #tvxMacdCanvas {{ width:100%; height:220px; display:block; }} .tvx-sublegend {{ position:absolute; left:12px; top:8px; z-index:3; font-size:11px; color:#8fa1b7; background:rgba(8,12,18,.75); border:1px solid rgba(255,255,255,.07); padding:5px 8px; border-radius:7px; }}
      .tvx-overlay {{ position:absolute; inset:0 0 220px 0; z-index:4; pointer-events:none; }} .tvx-legend {{ position:absolute; left:12px; top:10px; display:flex; gap:10px; align-items:center; font-size:11px; background:rgba(8,12,18,.72); border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:6px 8px; backdrop-filter:blur(8px); }} .tvx-order {{ position:absolute; left:12px; top:54px; width:190px; display:grid; grid-template-columns:1fr 1fr; gap:7px; }} .tvx-order div {{ color:white; text-align:center; border-radius:8px; padding:8px 5px; font-size:11px; font-weight:900; }} .tvx-order .sell {{ background:#ef5350; }} .tvx-order .buy {{ background:#26a69a; }} .tvx-order em {{ grid-column:1/3; text-align:center; font-style:normal; font-size:10px; color:#8fa1b7; }}
      .tvx-float-tape {{ position:absolute; right:12px; top:52px; display:flex; flex-direction:column; gap:4px; font-size:11px; font-variant-numeric:tabular-nums; }} .tvx-float-tape div {{ background:rgba(17,23,34,.72); border:1px solid rgba(255,255,255,.06); padding:3px 6px; border-radius:6px; animation:floatPulse 1.8s infinite alternate; }} @keyframes floatPulse {{ from {{ opacity:.72; transform:translateX(0); }} to {{ opacity:1; transform:translateX(-3px); }} }} .tvx-tip,.tvx-toast {{ position:absolute; display:none; background:rgba(13,19,28,.95); border:1px solid #2a3443; border-radius:8px; padding:7px 9px; font-size:11px; color:#d6dde8; box-shadow:0 12px 26px rgba(0,0,0,.35); }} .tvx-toast {{ right:12px; bottom:14px; display:block; opacity:0; transition:.2s; }}
      .tvx-sidepanel {{ background:#0d131c; border-left:1px solid #1c2531; padding:10px; }} .tvx-mini {{ display:flex; justify-content:space-between; gap:10px; border-bottom:1px solid #1c2531; padding:9px 0; }} .tvx-mini span {{ color:#8fa1b7; font-size:11px; }} .tvx-mini b {{ color:#f4f7fb; font-size:12px; text-align:right; }} .tvx-qty {{ display:grid; gap:5px; margin-top:10px; font-size:11px; color:#8fa1b7; }} .tvx-qty input,.tvx-qty select,.tvx-modal input {{ background:#080c12; color:#d6dde8; border:1px solid #263244; border-radius:7px; padding:7px; }} .tvx-orders {{ margin-top:10px; display:grid; gap:5px; font-size:11px; }} .tvx-orders div {{ border:1px solid #263244; border-radius:7px; padding:5px; }}
      .tvx-bottombar {{ border-top:1px solid #1c2531; border-bottom:0; font-size:11px; color:#8fa1b7; }} .tvx-bottombar span {{ padding:5px 7px; border:1px solid #263244; border-radius:6px; }} .tvx-bottombar b {{ margin-left:auto; color:#d6dde8; }} .tvx-bottombar em {{ font-style:normal; color:#8fa1b7; }} .tvx-modal {{ position:absolute; inset:0; display:none; align-items:center; justify-content:center; z-index:9; background:rgba(0,0,0,.45); }} .tvx-modal>div {{ background:#101722; border:1px solid #263244; border-radius:10px; padding:16px; display:grid; gap:9px; min-width:280px; }} .tvx-modal label {{ display:block; font-size:12px; }}
      @media(max-width:1000px) {{ .tvx-grid {{ grid-template-columns:38px 1fr; }} .tvx-rightbar,.tvx-sidepanel {{ display:none; }} .tvx-topbar,.tvx-readout {{ flex-wrap:wrap; }} .tvx-actions {{ margin-left:0; }} }}
    </style>
    <script>
      let candles = {payload}; let symbol=localStorage.getItem('neroChartSymbol')||'{default_ws_symbol}'; let interval=localStorage.getItem('neroChartInterval')||'{active_timeframe.lower().replace('h','h').replace('H','h')}'; let ws=null; let replay=false; let replayIndex=null; let playTimer=null; let alertPrice=null; let compareSeries=[]; let compareSymbol=''; let scaleMode=localStorage.getItem('neroScaleMode')||'auto'; let wsRetry=0; let wsHeartbeat=null; let indicators=JSON.parse(localStorage.getItem('neroIndicators')||'{{"ema9":true,"ema21":true,"macd":true,"rsi":false,"volume":false}}'); let drawings=JSON.parse(localStorage.getItem('neroDrawings')||'[]'); let redoStack=[]; let orders=JSON.parse(localStorage.getItem('neroOrders')||'[]'); let activeTool='cursor'; const app=document.getElementById('tvxApp'); const priceCanvas=document.getElementById('tvxPriceCanvas'); const priceCtx=priceCanvas.getContext('2d'); const macdCanvas=document.getElementById('tvxMacdCanvas'); const macdCtx=macdCanvas.getContext('2d'); const tip=document.getElementById('tvxTip'); const hoverText=document.getElementById('tvxHoverText');
      function toast(msg){{ const t=document.getElementById('tvxToast'); t.textContent=msg; t.style.opacity=1; setTimeout(()=>t.style.opacity=0,2200); }} function setConn(msg,col='#f0b90b'){{const el=document.getElementById('tvxConnection'); el.textContent=msg; el.style.color=col;}} function rsi(values,p=14){{const out=new Array(values.length).fill(null); for(let i=p;i<values.length;i++){{let g=0,l=0; for(let j=i-p+1;j<=i;j++){{const d=values[j]-values[j-1]; if(d>=0)g+=d; else l-=d;}} const rs=l===0?100:g/l; out[i]=l===0?100:100-(100/(1+rs));}} return out;}} function ema(values,p){{const k=2/(p+1);const out=[];let prev=values[0]||0;values.forEach((v,i)=>{{prev=i===0?v:v*k+prev*(1-k);out.push(prev);}});return out;}} function fit(canvas,ctx){{const dpr=window.devicePixelRatio||1;const r=canvas.getBoundingClientRect();canvas.width=r.width*dpr;canvas.height=r.height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);return r;}}
      async function fetchKlines(sym, intv){{ localStorage.setItem('neroChartSymbol',sym); localStorage.setItem('neroChartInterval',intv); compareSeries=[]; compareSymbol=''; try{{ const r=await fetch(`https://api.binance.com/api/v3/klines?symbol=${{sym}}&interval=${{intv}}&limit=500`); if(!r.ok) throw new Error('REST '+r.status); const j=await r.json(); candles=j.map(a=>({{t:new Date(a[0]).toISOString().slice(5,16),time:new Date(a[0]).toISOString(),o:+a[1],h:+a[2],l:+a[3],c:+a[4],v:+a[5]}})); document.getElementById('tvxSource').textContent=`Binance REST + WS ${{sym}} ${{intv}}`; replayIndex=null; redraw(); connectWs(sym,intv); }}catch(e){{ toast('Binance fetch failed; using NERO server candles'); redraw(); }} }}
      function connectWs(sym,intv){{ if(ws) ws.close(); clearInterval(wsHeartbeat); setConn('connecting '+sym+' '+intv); ws=new WebSocket(`wss://stream.binance.com:9443/ws/${{sym.toLowerCase()}}@kline_${{intv}}`); ws.onopen=()=>{{wsRetry=0;setConn('live WebSocket connected','#26a69a');wsHeartbeat=setInterval(()=>{{if(ws&&ws.readyState===1) setConn('live WebSocket connected','#26a69a');}},15000);}}; ws.onmessage=e=>{{ const k=JSON.parse(e.data).k; const row={{t:new Date(k.t).toISOString().slice(5,16),time:new Date(k.t).toISOString(),o:+k.o,h:+k.h,l:+k.l,c:+k.c,v:+k.v}}; const last=candles[candles.length-1]; if(last&&last.time===row.time) candles[candles.length-1]=row; else if(!last||last.time!==row.time) candles.push(row); candles=candles.slice(-500); if(!replay) redraw(); checkAlert(row.c); }}; ws.onerror=()=>{{setConn('WebSocket delayed - using REST/server fallback','#f0b90b');toast('WebSocket delayed');}}; ws.onclose=()=>{{clearInterval(wsHeartbeat); const wait=Math.min(30000,1000*Math.pow(2,wsRetry++)); setConn('reconnecting in '+Math.round(wait/1000)+'s','#f0b90b'); setTimeout(()=>{{if(symbol===sym&&interval===intv) connectWs(sym,intv);}},wait);}}; }}
      function drawPrice(mx=null,my=null){{ const view=replay&&replayIndex?candles.slice(0,replayIndex):candles; if(!view.length) return; const r=fit(priceCanvas,priceCtx); const w=r.width,h=r.height,pL=28,pR=82,pT=24,pB=26; priceCtx.clearRect(0,0,w,h); priceCtx.fillStyle=app.dataset.theme==='light'?'#ffffff':'#080c12'; priceCtx.fillRect(0,0,w,h); const highs=view.map(x=>x.h),lows=view.map(x=>x.l),closes=view.map(x=>x.c); const max=Math.max(...highs),min=Math.min(...lows),range=Math.max(max-min,1e-9),step=(w-pL-pR)/view.length,logMin=Math.log(Math.max(min,1e-9)),logMax=Math.log(Math.max(max,1e-9)),logRange=Math.max(logMax-logMin,1e-9),y=p=>scaleMode==="log"?pT+(logMax-Math.log(Math.max(p,1e-9)))/logRange*(h-pT-pB):pT+(max-p)/range*(h-pT-pB); priceCtx.strokeStyle='#152030'; for(let i=0;i<8;i++){{const yy=pT+i*(h-pT-pB)/7;priceCtx.beginPath();priceCtx.moveTo(pL,yy);priceCtx.lineTo(w-pR,yy);priceCtx.stroke();}} view.forEach((c,i)=>{{const x=pL+i*step+step/2,up=c.c>=c.o,col=up?'#26a69a':'#ef5350';priceCtx.strokeStyle=col;priceCtx.fillStyle=col;priceCtx.beginPath();priceCtx.moveTo(x,y(c.h));priceCtx.lineTo(x,y(c.l));priceCtx.stroke();const top=y(Math.max(c.o,c.c)),bot=y(Math.min(c.o,c.c)),bw=Math.max(2,step*.62);priceCtx.fillRect(x-bw/2,top,bw,Math.max(2,bot-top));}}); const e9=ema(closes,9),e21=ema(closes,21); if(indicators.ema9) line(e9,'#f0b90b'); if(indicators.ema21) line(e21,'#2962ff'); if(compareSeries.length) drawCompare(); function line(s,col){{priceCtx.strokeStyle=col;priceCtx.lineWidth=1.5;priceCtx.beginPath();s.forEach((p,i)=>{{const x=pL+i*step+step/2,yy=y(p);if(i===0)priceCtx.moveTo(x,yy);else priceCtx.lineTo(x,yy);}});priceCtx.stroke();}} function drawCompare(){{const series=compareSeries.slice(-view.length); if(series.length<2) return; const vals=series.map(x=>x.c),cMax=Math.max(...vals),cMin=Math.min(...vals),cRange=Math.max(cMax-cMin,1e-9); priceCtx.strokeStyle='#b388ff'; priceCtx.lineWidth=1.4; priceCtx.setLineDash([5,4]); priceCtx.beginPath(); vals.forEach((p,i)=>{{const x=pL+i*step+step/2,yy=pT+(cMax-p)/cRange*(h-pT-pB); if(i===0)priceCtx.moveTo(x,yy); else priceCtx.lineTo(x,yy);}}); priceCtx.stroke(); priceCtx.setLineDash([]); priceCtx.fillStyle='#b388ff'; priceCtx.fillText('Compare '+compareSymbol,pL+12,pT+18);}} drawings.forEach(d=>{{priceCtx.strokeStyle='#f0b90b'; priceCtx.setLineDash([6,4]); priceCtx.beginPath(); priceCtx.moveTo(pL,y(d.price)); priceCtx.lineTo(w-pR,y(d.price)); priceCtx.stroke(); priceCtx.setLineDash([]);}}); const last=view.at(-1); if(last){{const yy=y(last.c),col=last.c>=last.o?'#26a69a':'#ef5350';priceCtx.setLineDash([4,5]);priceCtx.strokeStyle=col;priceCtx.beginPath();priceCtx.moveTo(pL,yy);priceCtx.lineTo(w-pR,yy);priceCtx.stroke();priceCtx.setLineDash([]);priceCtx.fillStyle=col;priceCtx.fillRect(w-pR+8,yy-11,70,22);priceCtx.fillStyle='#081018';priceCtx.font='bold 11px Inter,Arial';priceCtx.fillText(last.c.toFixed(2),w-pR+13,yy+4);updateReadout(last,e9.at(-1),e21.at(-1));}} if(mx!==null){{const idx=Math.max(0,Math.min(view.length-1,Math.floor((mx-pL)/step)));const c=view[idx],x=pL+idx*step+step/2;priceCtx.strokeStyle='rgba(214,221,232,.45)';priceCtx.setLineDash([3,4]);priceCtx.beginPath();priceCtx.moveTo(x,pT);priceCtx.lineTo(x,h-pB);priceCtx.moveTo(pL,my);priceCtx.lineTo(w-pR,my);priceCtx.stroke();priceCtx.setLineDash([]);hoverText.textContent=`${{c.t}} O ${{c.o}} H ${{c.h}} L ${{c.l}} C ${{c.c}}`;tip.style.display='block';tip.style.left=(Math.min(mx+16,w-245))+'px';tip.style.top=(Math.max(my-62,10))+'px';tip.innerHTML=`<b>${{c.t}}</b><br>O ${{c.o}} H ${{c.h}}<br>L ${{c.l}} C ${{c.c}}`;}} }}
      function drawMacd(){{const view=replay&&replayIndex?candles.slice(0,replayIndex):candles;const r=fit(macdCanvas,macdCtx),w=r.width,h=r.height,pL=28,pR=82,pT=22,pB=18;macdCtx.clearRect(0,0,w,h);macdCtx.fillStyle='#090e15';macdCtx.fillRect(0,0,w,h); const close=view.map(x=>x.c),step=(w-pL-pR)/Math.max(close.length,1); if(indicators.rsi){{const vals=rsi(close,14),yR=v=>pT+(100-(v??50))/100*(h-pT-pB); macdCtx.strokeStyle='#1c2531'; [30,50,70].forEach(level=>{{const yy=yR(level); macdCtx.beginPath(); macdCtx.moveTo(pL,yy); macdCtx.lineTo(w-pR,yy); macdCtx.stroke();}}); macdCtx.strokeStyle='#b388ff'; macdCtx.beginPath(); vals.forEach((v,i)=>{{if(v===null)return; const x=pL+i*step+step/2,yy=yR(v); if(i===14)macdCtx.moveTo(x,yy); else macdCtx.lineTo(x,yy);}}); macdCtx.stroke(); document.getElementById('tvxMacdText').textContent='RSI '+(vals.at(-1)?.toFixed(1) || 'n/a'); return;}} if(!indicators.macd){{macdCtx.fillStyle='#8fa1b7';macdCtx.fillText('MACD hidden from indicator menu',pL+10,pT+22);document.getElementById('tvxMacdText').textContent='hidden';return;}} const e12=ema(close,12),e26=ema(close,26),macd=e12.map((v,i)=>v-e26[i]),signal=ema(macd,9),hist=macd.map((v,i)=>v-signal[i]),scale=Math.max(...macd.map(Math.abs),...signal.map(Math.abs),...hist.map(Math.abs),1e-9),y=v=>pT+(scale-v)/(scale*2)*(h-pT-pB);macdCtx.strokeStyle='#1c2531';macdCtx.beginPath();macdCtx.moveTo(pL,y(0));macdCtx.lineTo(w-pR,y(0));macdCtx.stroke();hist.forEach((v,i)=>{{const x=pL+i*step+step/2;macdCtx.fillStyle=v>=0?'#26a69a':'#ef5350';macdCtx.fillRect(x-Math.max(2,step*.35)/2,y(Math.max(v,0)),Math.max(2,step*.35),Math.abs(y(v)-y(0)));}}); if(indicators.volume){{const maxV=Math.max(...view.map(x=>x.v),1); view.forEach((c,i)=>{{const x=pL+i*step+step/2,vh=(c.v/maxV)*(h-pT-pB)*.28; macdCtx.fillStyle='rgba(143,161,183,.28)'; macdCtx.fillRect(x-Math.max(1,step*.25)/2,h-pB-vh,Math.max(1,step*.25),vh);}});}} l(macd,'#2962ff');l(signal,'#f0b90b');function l(s,c){{macdCtx.strokeStyle=c;macdCtx.beginPath();s.forEach((v,i)=>{{const x=pL+i*step+step/2,yy=y(v);if(i===0)macdCtx.moveTo(x,yy);else macdCtx.lineTo(x,yy);}});macdCtx.stroke();}} document.getElementById('tvxMacdText').textContent=`${{macd.at(-1)?.toFixed(2)}}  ${{signal.at(-1)?.toFixed(2)}}  ${{hist.at(-1)?.toFixed(2)}}`;}}
      function updateReadout(c,e9,e21){{const ch=c.c-c.o,p=c.o?ch/c.o*100:0,col=ch>=0?'#26a69a':'#ef5350';document.getElementById('tvxLivePrice').style.color=col;document.getElementById('tvxLivePrice').innerHTML=`${{c.c.toFixed(4)}}<small>${{ch.toFixed(4)}} / ${{p.toFixed(2)}}%</small>`;const spr=Math.max(c.c*.00035,.01);document.getElementById('tvxSell').innerHTML=`SELL<br><b>${{(c.c-spr/2).toFixed(4)}}</b>`;document.getElementById('tvxBuy').innerHTML=`BUY<br><b>${{(c.c+spr/2).toFixed(4)}}</b>`;document.getElementById('tvxSpread').textContent='spread '+spr.toFixed(4);document.getElementById('tvxEma9').textContent='EMA 9 '+(e9||0).toFixed(2);document.getElementById('tvxEma21').textContent='EMA 21 '+(e21||0).toFixed(2);}}
      function floats(){{document.getElementById('tvxFloatTape').innerHTML=candles.slice(-10).reverse().map(c=>'<div style="color:'+(c.c>=c.o?'#26a69a':'#ef5350')+'">'+c.c.toFixed(4)+' | '+(((c.c-c.o)/c.o)*100).toFixed(2)+'%</div>').join('');}} function redraw(){{drawPrice();drawMacd();floats();renderOrders();}} function clock(){{document.getElementById('tvxClock').textContent='UTC '+new Date().toISOString().slice(11,19);}}
      function checkAlert(px){{if(alertPrice&&((px>=alertPrice&&candles.at(-2)?.c<alertPrice)||(px<=alertPrice&&candles.at(-2)?.c>alertPrice))){{toast('Price alert triggered '+alertPrice);alertPrice=null;}}}}
      async function toggleCompare(){{ if(compareSeries.length){{compareSeries=[];compareSymbol='';toast('Compare off');redraw();return;}} const map={{BTCUSDT:'ETHUSDT',ETHUSDT:'BTCUSDT',SOLUSDT:'BTCUSDT',BNBUSDT:'BTCUSDT',XRPUSDT:'BTCUSDT',DOGEUSDT:'BTCUSDT',NEARUSDT:'BTCUSDT',PAXGUSDT:'BTCUSDT'}}; compareSymbol=map[symbol]||'BTCUSDT'; try{{const r=await fetch(`https://api.binance.com/api/v3/klines?symbol=${{compareSymbol}}&interval=${{interval}}&limit=500`); if(!r.ok) throw new Error('REST '+r.status); const j=await r.json(); compareSeries=j.map(a=>({{c:+a[4]}})); toast('Compare '+compareSymbol+' on'); redraw();}}catch(e){{compareSeries=[];compareSymbol='';toast('Compare fetch failed');}} }}
      function saveIndicatorState(){{localStorage.setItem('neroIndicators',JSON.stringify(indicators));}}
      function snapshotChart(){{const a=document.createElement('a'); a.download=`nero-${{symbol}}-${{interval}}.png`; a.href=priceCanvas.toDataURL('image/png'); a.click();}}
      document.getElementById('tvxSymbol').value=symbol; document.querySelectorAll('#tvxFrames button').forEach(b=>{{if(b.dataset.i===interval)b.classList.add('active'); b.onclick=()=>{{interval=b.dataset.i;document.querySelectorAll('#tvxFrames button').forEach(x=>x.classList.remove('active'));b.classList.add('active');fetchKlines(symbol,interval);}}}}); document.getElementById('tvxSymbol').onchange=e=>{{symbol=e.target.value;fetchKlines(symbol,interval);}}; document.getElementById('tvxThemeBtn').onclick=()=>{{app.dataset.theme=app.dataset.theme==='dark'?'light':'dark';localStorage.setItem('neroChartTheme',app.dataset.theme);redraw();}}; const savedTheme=localStorage.getItem('neroChartTheme'); if(savedTheme) app.dataset.theme=savedTheme; document.getElementById('tvxIndicatorBtn').onclick=()=>document.getElementById('tvxIndicatorModal').style.display='flex'; document.getElementById('tvxCloseIndicators').onclick=()=>document.getElementById('tvxIndicatorModal').style.display='none'; document.querySelectorAll('#tvxIndicatorModal input[data-ind]').forEach(cb=>{{cb.checked=!!indicators[cb.dataset.ind]; cb.onchange=()=>{{indicators[cb.dataset.ind]=cb.checked;saveIndicatorState();redraw();}};}}); document.getElementById('tvxCompareBtn').onclick=toggleCompare; document.getElementById('tvxScaleBtn').textContent='Scale '+scaleMode.toUpperCase(); document.getElementById('tvxScaleBtn').onclick=()=>{{scaleMode=scaleMode==='auto'?'log':scaleMode==='log'?'linear':'auto';localStorage.setItem('neroScaleMode',scaleMode);document.getElementById('tvxScaleBtn').textContent='Scale '+scaleMode.toUpperCase();redraw();}}; document.getElementById('tvxFullBtn').onclick=()=>{{ if(app.requestFullscreen) app.requestFullscreen(); }}; document.getElementById('tvxShotBtn').onclick=snapshotChart; document.getElementById('tvxUndoBtn').onclick=()=>{{const d=drawings.pop(); if(d){{redoStack.push(d);localStorage.setItem('neroDrawings',JSON.stringify(drawings));redraw();}}}}; document.getElementById('tvxRedoBtn').onclick=()=>{{const d=redoStack.pop(); if(d){{drawings.push(d);localStorage.setItem('neroDrawings',JSON.stringify(drawings));redraw();}}}}; document.getElementById('tvxAlertBtn').onclick=()=>document.getElementById('tvxAlertModal').style.display='flex'; document.getElementById('tvxCloseAlert').onclick=()=>document.getElementById('tvxAlertModal').style.display='none'; document.getElementById('tvxSaveAlert').onclick=()=>{{alertPrice=+document.getElementById('tvxAlertPrice').value;document.getElementById('tvxAlertModal').style.display='none';toast('Alert saved '+alertPrice);}};
      document.getElementById('tvxLeftTools').onclick=e=>{{if(e.target.dataset.tool){{activeTool=e.target.dataset.tool;document.querySelectorAll('#tvxLeftTools button').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');if(activeTool==='trash'){{drawings=[];redoStack=[];localStorage.setItem('neroDrawings','[]');redraw();}}}}}}; priceCanvas.onclick=e=>{{if(activeTool==='hline'){{const r=priceCanvas.getBoundingClientRect(),yClick=e.clientY-r.top,view=candles,max=Math.max(...view.map(x=>x.h)),min=Math.min(...view.map(x=>x.l));const price=max-(yClick-24)/(r.height-50)*(max-min);drawings.push({{type:'hline',price}});redoStack=[];localStorage.setItem('neroDrawings',JSON.stringify(drawings));redraw();}}}};
      document.getElementById('tvxSubmitOrder').onclick=()=>{{const c=candles.at(-1);orders.unshift({{symbol,side:'SIM',type:document.getElementById('tvxOrderType').value,risk:document.getElementById('tvxRiskPct').value+'%',qty:document.getElementById('tvxQty').value,price:c.c.toFixed(4),time:new Date().toLocaleTimeString()}});orders=orders.slice(0,6);localStorage.setItem('neroOrders',JSON.stringify(orders));renderOrders();toast('Simulated order added');}}; function renderOrders(){{document.getElementById('tvxOrders').innerHTML='<b>Open orders</b>'+orders.map(o=>`<div>${{o.time}} ${{o.symbol}} ${{o.type||'Market'}} risk ${{o.risk||'-'}} qty ${{o.qty}} @ ${{o.price}}</div>`).join('');}}
      document.getElementById('tvxReplayBtn').onclick=()=>{{replay=!replay;replayIndex=replay?Math.max(30,Math.floor(candles.length*.65)):null;toast(replay?'Replay on':'Replay off');redraw();}}; document.getElementById('tvxBack').onclick=()=>{{if(replay){{replayIndex=Math.max(30,replayIndex-1);redraw();}}}}; document.getElementById('tvxFwd').onclick=()=>{{if(replay){{replayIndex=Math.min(candles.length,replayIndex+1);redraw();}}}}; document.getElementById('tvxPlay').onclick=()=>{{if(!replay){{replay=true;replayIndex=30;}} clearInterval(playTimer); playTimer=setInterval(()=>{{replayIndex=Math.min(candles.length,replayIndex+1);redraw();if(replayIndex>=candles.length)clearInterval(playTimer);}}, document.getElementById('tvxSpeed').value==='4x'?180:document.getElementById('tvxSpeed').value==='2x'?350:700);}};
      priceCanvas.addEventListener('mousemove',e=>{{const r=priceCanvas.getBoundingClientRect();drawPrice(e.clientX-r.left,e.clientY-r.top);}}); priceCanvas.addEventListener('mouseleave',()=>{{tip.style.display='none';hoverText.textContent='Move mouse over candles';drawPrice();}}); window.addEventListener('resize',redraw); setInterval(clock,1000); redraw(); clock(); if(symbol.endsWith('USDT')) connectWs(symbol,interval);
    </script>
    """
    components.html(html, height=1190, scrolling=False)
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



def _render_strategy_research_lab_tab() -> None:
    st.subheader("Strategy Research Lab")
    st.caption("NERO's controlled self-improvement layer: proposes alternate paper-test candidates without changing live rules.")
    report = build_strategy_research_report()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Lab Score", f"{report.lab_score:.0f}/100")
    col_b.metric("Label", report.label)
    col_c.metric("Sample", report.sample_status)
    col_d.metric("Current Edge", report.current_edge)
    for note in report.notes:
        st.info(note)
    if report.top_blockers:
        st.subheader("Top Strategy Blockers")
        st.dataframe(pd.DataFrame(report.top_blockers), use_container_width=True, hide_index=True)
    st.subheader("Candidate Algos For Forward Testing")
    if report.candidates:
        st.dataframe(pd.DataFrame(report.candidate_rows()), use_container_width=True, hide_index=True)
        with st.expander("Candidate Detail"):
            for candidate in report.candidates:
                st.markdown(f"**{candidate.candidate_id}: {candidate.title}**")
                st.caption(candidate.hypothesis)
                st.caption("Changes: " + "; ".join(candidate.proposed_changes))
                st.caption("Evidence: " + "; ".join(candidate.evidence))
                st.caption("Risks: " + "; ".join(candidate.risks))
    else:
        st.info("No candidates yet. NERO needs more paper-trade and rejection data.")
    st.warning("Safety rule: candidates are RESEARCH_ONLY. NERO must not auto-change strategy parameters without manual version approval.")


def _render_profit_edge_engine_tab() -> None:
    st.subheader("Profit Edge Engine")
    st.caption("Profit-seeking evidence filter. It ranks paper candidates, blocks capital drains, and tracks whether positive edges can recover historical strategy drag.")
    try:
        edge_report, edge_summary = build_profit_edge_report()
    except Exception as exc:  # pragma: no cover - dashboard should survive report errors
        st.warning(f"Profit Edge Engine could not be built: {exc}")
        return
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Status", edge_summary.status)
    col_b.metric("Profit Candidates", str(edge_summary.profit_candidates))
    col_c.metric("Capital Drains", str(edge_summary.capital_drains))
    col_d.metric("Top Edge", edge_summary.top_candidate)
    col_e.metric("Recovery Ratio", f"{edge_summary.recovery_ratio:.0%}")
    col_f, col_g, col_h = st.columns(3)
    col_f.metric("Positive Pool", f"${edge_summary.evidence_pool_pnl:,.2f}")
    col_g.metric("Blocked Drag", f"${edge_summary.blocked_drag_pnl:,.2f}")
    col_h.metric("Recovery Gap", f"${edge_summary.recovery_gap:,.2f}")
    for note in edge_summary.notes:
        st.info(note)
    if edge_report.empty:
        st.info("No profit-edge report yet. Run the Strategy Lab workflow first.")
        return
    display = edge_report.copy()
    preferred_columns = [
        "display_label",
        "role",
        "decision",
        "paper_weight",
        "recovery_priority",
        "total_trades",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "net_pnl",
        "edge_score",
        "reason",
    ]
    display = display[[column for column in preferred_columns if column in display.columns]]
    for column in ["win_rate", "paper_weight"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.0%}")
    for column in ["expectancy_r", "profit_factor", "net_pnl", "edge_score"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.warning("This engine is paper-only. It is a focus and loss-containment tool, not a real-money trade instruction.")


def _render_strategy_test_lab_tab() -> None:
    st.subheader("Strategy TEST Lab")
    st.caption("Old strategies plus new evidence candidates. No real orders. GitHub Actions records evidence so NERO can rank strategies after enough trades.")
    report_dir = STRATEGY_LAB_REPORT_DIR
    summary_path = report_dir / "strategy_lab_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = write_strategy_lab_summary(report_dir, list(CANDIDATES.values()))
    if summary.empty:
        st.info("No Strategy Lab records yet. GitHub runner will create them after first run.")
        return
    best = summary.sort_values(["rating_score", "total_trades"], ascending=False).iloc[0]
    total_trades = int(pd.to_numeric(summary.get("total_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    active_count = int(summary.get("enabled", pd.Series([True] * len(summary))).astype(str).str.lower().ne("false").sum()) if "enabled" in summary else len(summary)
    col_a.metric("Algos Testing", str(active_count))
    col_b.metric("Total Paper Trades", str(total_trades))
    col_c.metric("Best Algo", str(best.get("display_label", best.get("candidate_id", "-"))))
    col_d.metric("Best Rating", str(best.get("rating", "-")))
    col_e.metric("Best Score", f"{float(best.get('rating_score', 0.0)):.0f}/100")
    if total_trades < 30:
        st.warning("Still early. Reliable ranking needs about 30-50 closed trades per algo.")
    st.info("Ratings combine win rate, expectancy, profit factor, drawdown, and sample size. Use this as evidence collection, not a standalone trade command.")
    try:
        live_status, live_summary = build_live_trade_status_report()
    except Exception as exc:  # pragma: no cover - dashboard should survive report errors
        live_status = pd.DataFrame()
        live_summary = None
        st.warning(f"Live trade status report could not be built: {exc}")
    if live_summary is not None:
        st.subheader("Live Trade Reconciliation")
        l_col_a, l_col_b, l_col_c, l_col_d = st.columns(4)
        l_col_a.metric("State Open", str(live_summary.state_open_trades))
        l_col_b.metric("Trusted Live", str(live_summary.trusted_live_trades))
        l_col_c.metric("Stale / Blocked", str(live_summary.stale_or_blocked_trades))
        l_col_d.metric("Heartbeat Mismatch", str(live_summary.heartbeat_mismatches))
        if live_summary.state_open_trades and live_summary.trusted_live_trades == 0:
            st.warning("Open state files exist, but none qualify as trusted live paper trades. They are stale, blocked, or heartbeat-mismatched.")
        elif live_summary.stale_or_blocked_trades:
            st.info("Some open states are excluded from live status because heartbeat/quarantine checks failed.")
        if not live_status.empty:
            status_display = live_status.copy()
            status_columns = [
                "system",
                "strategy_id",
                "asset",
                "side",
                "opened_at",
                "entry_price",
                "target",
                "stop_loss",
                "heartbeat_at",
                "heartbeat_age_minutes",
                "heartbeat_open_trade",
                "trusted_live",
                "issue",
                "action",
            ]
            status_display = status_display[[column for column in status_columns if column in status_display.columns]]
            st.dataframe(status_display, use_container_width=True, hide_index=True)
    try:
        verification = build_strategy_verification_report()
    except Exception as exc:  # pragma: no cover - dashboard should survive report errors
        verification = pd.DataFrame()
        st.warning(f"Strategy verification report could not be built: {exc}")
    if not verification.empty:
        st.subheader("Verification Verdict")
        verdict_counts = verification["verdict"].value_counts().to_dict() if "verdict" in verification else {}
        q_col, w_col, p_col, s_col = st.columns(4)
        q_col.metric("Quarantine", str(verdict_counts.get("QUARANTINE", 0)))
        w_col.metric("Watchlist", str(verdict_counts.get("WATCHLIST", 0)))
        p_col.metric("Promote Paper", str(verdict_counts.get("PROMOTE_PAPER", 0)))
        s_col.metric("Too Early", str(verdict_counts.get("INSUFFICIENT_SAMPLE", 0)))
        st.caption("Verification is stricter than the lab rating: it applies sample-size, profit-factor, expectancy, drawdown, and data-trust gates.")
        verification_display = verification.copy()
        verification_columns = [
            "display_label",
            "verdict",
            "evidence_score",
            "sample_status",
            "total_trades",
            "win_rate",
            "expectancy_r",
            "profit_factor",
            "max_drawdown",
            "net_pnl",
            "primary_reason",
            "action",
        ]
        verification_display = verification_display[[column for column in verification_columns if column in verification_display.columns]]
        for column in ["win_rate", "max_drawdown"]:
            if column in verification_display:
                verification_display[column] = pd.to_numeric(verification_display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.0%}")
        for column in ["expectancy_r", "profit_factor", "net_pnl", "evidence_score"]:
            if column in verification_display:
                verification_display[column] = pd.to_numeric(verification_display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.2f}")
        st.dataframe(verification_display, use_container_width=True, hide_index=True)
        quarantine = build_strategy_quarantine_report()
        if not quarantine.empty:
            st.subheader("Quarantine Automation")
            st.warning("These strategies are blocked from opening new paper trades until reviewed. Existing history remains preserved.")
            q_display = quarantine.copy()
            for column in ["expectancy_r", "profit_factor", "net_pnl"]:
                if column in q_display:
                    q_display[column] = pd.to_numeric(q_display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.2f}")
            st.dataframe(q_display, use_container_width=True, hide_index=True)
    if "bucket" in summary:
        st.caption("Lab groups: OLD_TEST = existing NERO algos, NEW_TEST = Claude-sweep candidates, RESEARCH_ONLY = not executed until a real engine is wired.")
        bucket_view = summary.groupby("bucket", dropna=False).agg(
            algos=("candidate_id", "count"),
            trades=("total_trades", "sum"),
            avg_score=("rating_score", "mean"),
        ).reset_index()
        st.dataframe(bucket_view, use_container_width=True, hide_index=True)
    try:
        sunflower_report, sunflower_summary = build_sunflower_profit_bridge_report()
    except Exception as exc:  # pragma: no cover - dashboard should survive report errors
        sunflower_report = pd.DataFrame()
        sunflower_summary = None
        st.warning(f"Sunflower profit discipline could not be built: {exc}")
    if sunflower_summary is not None:
        st.subheader("Sunflower Profit Discipline")
        st.caption("Merged Sunflower discipline: profit must pass data, sample, drawdown, quarantine, and cost-aware gates before NERO trusts it.")
        sf_a, sf_b, sf_c, sf_d, sf_e = st.columns(5)
        sf_a.metric("Status", sunflower_summary.status)
        sf_b.metric("Disciplined", str(sunflower_summary.disciplined_profit_candidates))
        sf_c.metric("Early Profit", str(sunflower_summary.early_profit_watchlist))
        sf_d.metric("Blocked", str(sunflower_summary.capital_drains_blocked))
        sf_e.metric("Top Candidate", sunflower_summary.top_candidate)
        sf_f, sf_g = st.columns(2)
        sf_f.metric("Positive Pool", f"${sunflower_summary.positive_pool_pnl:,.2f}")
        sf_g.metric("Blocked Pool", f"${sunflower_summary.blocked_pool_pnl:,.2f}")
        for note in sunflower_summary.notes:
            st.info(note)
        if not sunflower_report.empty:
            sf_display = sunflower_report.copy()
            sf_columns = [
                "display_label",
                "sunflower_gate",
                "decision",
                "discipline_score",
                "total_trades",
                "win_rate",
                "expectancy_r",
                "profit_factor",
                "max_drawdown",
                "net_pnl",
                "reason",
            ]
            sf_display = sf_display[[column for column in sf_columns if column in sf_display.columns]]
            for column in ["win_rate", "max_drawdown"]:
                if column in sf_display:
                    sf_display[column] = pd.to_numeric(sf_display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.0%}")
            for column in ["discipline_score", "expectancy_r", "profit_factor", "net_pnl"]:
                if column in sf_display:
                    sf_display[column] = pd.to_numeric(sf_display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.2f}")
            st.dataframe(sf_display, use_container_width=True, hide_index=True)
    display = summary.copy()
    preferred_columns = [
        "display_label",
        "bucket",
        "interval",
        "asset_filter",
        "family",
        "total_trades",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "max_drawdown",
        "net_pnl",
        "rating_score",
        "rating",
        "evidence_note",
    ]
    display = display[[column for column in preferred_columns if column in display.columns] + [column for column in display.columns if column not in preferred_columns]]
    for column in ["win_rate", "max_drawdown"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.0%}")
    for column in ["expectancy_r", "profit_factor", "net_pnl", "rating_score"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).map(lambda value: f"{value:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    with st.expander("Per-algo report files"):
        rows = []
        for candidate_id in CANDIDATES:
            csv_path = report_dir / f"strategy_lab_{candidate_id}.csv"
            json_path = report_dir / f"strategy_lab_{candidate_id}.json"
            spec = CANDIDATES[candidate_id]
            rows.append({"Algo": getattr(spec, "display_label", "") or candidate_id, "Bucket": getattr(spec, "bucket", "OLD_TEST"), "Interval": getattr(spec, "interval", "1h"), "CSV": str(csv_path), "CSV Exists": csv_path.exists(), "JSON Exists": json_path.exists()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_strategy_evolution_tab() -> None:
    st.subheader("NERO Self-Evolution Lab")
    st.caption("Loss autopsy, Strategy Doctor recommendations, and versioned shadow-test variants. No live rules are changed silently.")
    report = build_strategy_evolution_report()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Maturity", f"{report.maturity_score:.0f}/100")
    col_b.metric("Label", report.label)
    col_c.metric("Trades Read", str(report.total_trades))
    col_d.metric("Losses Autopsied", str(report.total_losses))
    for note in report.notes:
        st.info(note)
    if report.autopsy_rows:
        st.subheader("Loss Autopsy")
        st.dataframe(pd.DataFrame(report.autopsy_rows), use_container_width=True, hide_index=True)
    if report.recommendation_rows:
        st.subheader("Strategy Doctor")
        st.dataframe(pd.DataFrame(report.recommendation_rows), use_container_width=True, hide_index=True)
    repair_workbench = build_strategy_repair_workbench()
    if not repair_workbench.empty:
        st.subheader("Active Repair Operation Theater")
        st.caption(
            "Quarantined strategies are mapped to versioned repair candidates. "
            "Repairs stay paper-only until release gates are met."
        )
        st.dataframe(repair_workbench, use_container_width=True, hide_index=True)
    repair_lab = build_strategy_repair_lab_report()
    if not repair_lab.empty:
        st.subheader("Repair Lab Fresh-Data Guard")
        st.caption("Every repair attempt must use a genuinely unseen historical window or forward paper tracking from today. Same-window retests are blocked.")
        decision_counts = repair_lab["promotion_decision"].value_counts().to_dict() if "promotion_decision" in repair_lab else {}
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Repair Lines", str(len(repair_lab)))
        col_b.metric("Collecting", str(decision_counts.get("COLLECT_FRESH_DATA", 0)))
        col_c.metric("Design Needed", str(decision_counts.get("DESIGN_REQUIRED", 0)))
        col_d.metric("Dead After 4", str(decision_counts.get("PERMANENTLY_DEAD", 0)))
        if {"dashboard_lineage_label", "repair_vs_parent_net_delta"}.issubset(repair_lab.columns):
            chart_rows = repair_lab[["dashboard_lineage_label", "repair_vs_parent_net_delta"]].copy()
            chart_rows["repair_vs_parent_net_delta"] = pd.to_numeric(chart_rows["repair_vs_parent_net_delta"], errors="coerce").fillna(0.0)
            chart_rows = chart_rows.set_index("dashboard_lineage_label")
            st.bar_chart(chart_rows, use_container_width=True)
        preferred = [
            "parent_label",
            "repair_label",
            "attempt_number",
            "failure_reason_code",
            "repair_trades",
            "repair_net_pnl",
            "repair_vs_parent_net_delta",
            "sample_milestone",
            "promotion_decision",
            "lineage_status",
            "random_baseline_status",
            "anti_overfit_guard",
        ]
        visible = repair_lab[[column for column in preferred if column in repair_lab.columns] + [column for column in repair_lab.columns if column not in preferred]]
        st.dataframe(visible, use_container_width=True, hide_index=True)
    if report.asset_action_rows:
        st.subheader("Asset Failure Correction")
        st.caption("NERO separates promising assets from weak or data-blocked areas before proposing new hypotheses.")
        st.dataframe(pd.DataFrame(report.asset_action_rows), use_container_width=True, hide_index=True)
    if report.variant_rows:
        st.subheader("Shadow-Test Variant Proposals")
        st.dataframe(pd.DataFrame(report.variant_rows), use_container_width=True, hide_index=True)
    st.warning("Autonomy safety: NERO can propose and shadow-test improvements, but production strategy changes must be versioned and audited.")


def _render_hypothesis_quality_gate_tab() -> None:
    st.subheader("Hypothesis Quality Gate")
    st.caption("Pre-deployment brain filter: scores new and repair hypotheses before they receive fresh paper-test budget.")
    try:
        gate_report, gate_summary = build_hypothesis_quality_gate()
    except Exception as exc:
        st.warning(f"Hypothesis Quality Gate could not be built: {exc}")
        return

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Status", gate_summary.status)
    col_b.metric("Approved", str(gate_summary.approved_shadow_tests))
    col_c.metric("Repair First", str(gate_summary.repair_first))
    col_d.metric("Rejected", str(gate_summary.rejected))
    col_e.metric("Avg Score", f"{gate_summary.average_score:.0f}/100")
    st.metric("Top Hypothesis", gate_summary.top_hypothesis)
    for note in gate_summary.notes:
        st.info(note)
    if gate_report.empty:
        st.info("No hypothesis gate report yet. Run Strategy Evolution first.")
        return

    display = gate_report.copy()
    preferred = [
        "proposed_variant",
        "decision",
        "gate_score",
        "parent",
        "family",
        "parent_trades",
        "parent_expectancy_r",
        "parent_profit_factor",
        "parent_net_pnl",
        "fixes_known_failure",
        "evidence_quality",
        "overfit_risk",
        "reason",
    ]
    display = display[[column for column in preferred if column in display.columns]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.warning("Gate decisions are paper-research controls, not live trade instructions.")
def _load_strategy_lab_rows() -> list[dict[str, object]]:
    summary_path = STRATEGY_LAB_REPORT_DIR / "strategy_lab_summary.csv"
    if not summary_path.exists():
        return []
    try:
        return pd.read_csv(summary_path).to_dict("records")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return []


def _render_nero_chat_tab(context: NeroChatContext) -> None:
    st.subheader("NERO Chat")
    st.caption("Ask NERO in simple language. It explains current dashboard readings; it does not place real orders.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Asset", context.asset)
    col_b.metric("Consensus", context.consensus_class.replace("_", " "), f"{context.consensus_quality:.0f}/100")
    col_c.metric("Trade Desk", context.trade_action.replace("_", " "))
    if "nero_chat_messages" not in st.session_state:
        st.session_state.nero_chat_messages = [
            {"role": "assistant", "content": answer_nero_chat("summary", context)}
        ]
    with st.expander("Suggested questions", expanded=False):
        st.dataframe(pd.DataFrame({"Question": SUGGESTED_QUESTIONS}), use_container_width=True, hide_index=True)
    for message in st.session_state.nero_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask NERO: trade lena chahiye, risk kya hai, next trigger kya hai...")
    if question:
        st.session_state.nero_chat_messages.append({"role": "user", "content": question})
        answer = answer_nero_chat(question, context)
        st.session_state.nero_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()
    if st.button("Reset NERO Chat"):
        st.session_state.nero_chat_messages = [{"role": "assistant", "content": answer_nero_chat("summary", context)}]
        st.rerun()

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



def _render_trade_path_tab(asset: str, price_history: pd.DataFrame, source: str, sentiment_score: float | None = None) -> None:
    st.subheader("Next Trade Path")
    st.caption("Plain trader roadmap: what blocks the setup, what must improve, and when NERO should check again.")
    snapshot = build_quant_snapshot(price_history, asset=asset, source=source)
    garch_report = build_garch_volatility_report(price_history, asset)
    local_consensus = build_quant_consensus_report(snapshot, garch_report)

    external_score = None
    external_label = "not loaded"
    external_notes: list[str] = []
    if asset == "BTC":
        if st.button("Refresh ETF flow for trade path", key="trade_path_refresh_etf"):
            etf_report = fetch_etf_flow_score()
            external_score = etf_report.etf_flow_score if etf_report.etf_flow_label != "DATA_INSUFFICIENT" else None
            external_label = etf_report.etf_flow_label
            external_notes = etf_report.notes
            st.caption("ETF flow source: " + ("actual net-flow CSV/API" if not etf_report.is_proxy else "price/volume proxy fallback"))
    elif asset == "GOLD":
        if st.button("Refresh real-yield for trade path", key="trade_path_refresh_real_yield"):
            real_yield_report = fetch_gold_real_yield_score()
            external_score = real_yield_report.real_yield_score if real_yield_report.real_yield_label != "DATA_INSUFFICIENT" else None
            external_label = real_yield_report.real_yield_label
            external_notes = real_yield_report.notes
            st.caption("Real-yield source: " + ("official CSV/API" if not real_yield_report.is_proxy else "yfinance proxy fallback"))

    for note in external_notes:
        st.info(str(note))

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
    trade_path = build_trade_path_report(
        TradePathInput(
            asset=asset,
            readiness_label=readiness.label,
            readiness_score=readiness.readiness_score,
            opportunity_decision=scanner.decision,
            opportunity_score=scanner.opportunity_score,
            direction_bias=scanner.direction_bias,
            quant_score=local_consensus.score,
            external_score=external_score,
            external_label=external_label,
            sentiment_score=_scanner_sentiment_score(sentiment_score),
            volatility_regime=garch_report.regime,
            blockers=readiness.blockers,
            failed_conditions=scanner.failed_conditions,
            near_miss_conditions=scanner.near_miss_conditions,
            has_active_paper_trade=scanner_inputs.paper_trade_state.has_open_position or scanner_inputs.paper_trade_state.has_pending_order,
        )
    )

    pcol_a, pcol_b, pcol_c, pcol_d = st.columns(4)
    pcol_a.metric("Path", trade_path.path_label)
    pcol_b.metric("Readiness", readiness.label)
    pcol_c.metric("Readiness Score", f"{readiness.readiness_score:.0f}/100")
    pcol_d.metric("Opportunity", f"{scanner.opportunity_score:.0f}/100")
    st.info(trade_path.action)
    st.warning("Direction: " + scanner.direction_bias + " | Next check: " + trade_path.next_check)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Required Confirmations**")
        for item in trade_path.missing_confirmations or ["none"]:
            st.caption(item)
    with col_b:
        st.markdown("**Watch Triggers**")
        for item in trade_path.watch_triggers or ["none"]:
            st.caption(item)
    with col_c:
        st.markdown("**Why Not Yet**")
        for item in trade_path.blocker_explanations or readiness.blockers or scanner.failed_conditions or ["none"]:
            st.caption(item)

    st.subheader("Underlying Gate Scores")
    st.dataframe(
        pd.DataFrame(
            [
                {"Gate": "Quant Consensus", "Reading": f"{local_consensus.score:.0f}/100", "Status": local_consensus.label},
                {"Gate": "Volatility", "Reading": garch_report.regime, "Status": f"shock {garch_report.shock_score:.0f}/100"},
                {"Gate": "External", "Reading": "not loaded" if external_score is None else f"{external_score:.0f}/100", "Status": external_label},
                {"Gate": "Opportunity", "Reading": f"{scanner.opportunity_score:.0f}/100", "Status": scanner.decision},
                {"Gate": "Readiness", "Reading": f"{readiness.readiness_score:.0f}/100", "Status": readiness.label},
            ]
        ),
        use_container_width=True,
        hide_index=True,
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
        if st.button("Refresh Gold real-yield", key="refresh_gold_real_yield_proxy"):
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
            st.caption("Real-yield source: " + ("official CSV/API" if not real_yield_report.is_proxy else "yfinance proxy fallback"))
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

    if asset == "BTC":
        st.subheader("BTC Structural Models")
        latest_btc_price = None
        if isinstance(price_history, pd.DataFrame) and not price_history.empty:
            for price_col in ("close", "Close"):
                if price_col in price_history.columns:
                    closes = pd.to_numeric(price_history[price_col], errors="coerce").dropna()
                    if not closes.empty:
                        latest_btc_price = float(closes.iloc[-1])
                    break
        structural = build_btc_structural_report(current_price=latest_btc_price)
        bcol_a, bcol_b, bcol_c, bcol_d = st.columns(4)
        bcol_a.metric("Structural Score", f"{structural.structural_score:.0f}/100")
        bcol_b.metric("Label", structural.structural_label)
        bcol_c.metric("Stock-to-Flow", "n/a" if structural.stock_to_flow is None else f"{structural.stock_to_flow:.1f}")
        bcol_d.metric("Blocks to Halving", f"{structural.blocks_to_halving:,}")
        for note in structural.notes:
            st.info(note)
        st.dataframe(pd.DataFrame(structural.rows()), use_container_width=True, hide_index=True)
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


def _render_cycle_intelligence_tab(asset: str, price_history: pd.DataFrame, source: str) -> None:
    st.subheader("Cycle Intelligence")
    st.caption("Mayer Multiple and cycle context from real daily closes only. Missing feeds are reported as unavailable, not converted into fake scores.")
    cycle_asset = "PAXG" if asset == "GOLD" else asset
    if asset == "GOLD":
        report = build_cycle_intelligence_report(assets=["PAXG"], prefer_live=True)
    else:
        price_frames = {cycle_asset: price_history}
        report = build_cycle_intelligence_report(
            assets=[cycle_asset],
            price_frames=price_frames,
            provided_sources={cycle_asset: (source, "dashboard")},
            prefer_live=False,
        )
    row = report.rows[0] if report.rows else {}

    if row.get("status") == "OK":
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Mayer Multiple", _display_number(row.get("mayer_multiple"), 3))
        col_b.metric("MM Percentile", _display_percent(row.get("mm_percentile_rank")))
        col_c.metric("SMA200 Slope", str(row.get("sma200_slope_label") or "n/a"))
        col_d.metric("Drawdown", _display_percent(row.get("drawdown_from_high_pct")))
        st.info("Cycle context is available, but this is not a standalone trade signal and does not alter NERO live-entry rules.")
    else:
        st.warning(f"Cycle Intelligence unavailable: {row.get('unavailable_reason', 'unknown reason')}")

    if row.get("paxg_caveat"):
        st.warning(str(row["paxg_caveat"]))
    for note in report.notes:
        st.info(note)

    st.dataframe(pd.DataFrame(cycle_dashboard_rows(row)), use_container_width=True, hide_index=True)
    st.caption(
        f"Source: {row.get('source') or source} | Method: {row.get('data_method', 'n/a')} | "
        f"Daily closes: {row.get('total_daily_closes', 0)} | Computable MM history: {row.get('computable_history_days', 0)}"
    )

    st.subheader("Layer Availability")
    st.dataframe(pd.DataFrame(report.availability_rows), use_container_width=True, hide_index=True)

    st.subheader("Field Provenance")
    provenance = pd.DataFrame(
        [
            {"Field": "latest_close", "Source": "daily close", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "sma_200", "Source": "200-day rolling mean of close", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "mayer_multiple", "Source": "latest_close / sma_200", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "distance_from_sma200_pct", "Source": "(latest_close - sma_200) / sma_200", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "mm_percentile_rank", "Source": "rank of current MM versus computable MM history", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "sma200_slope_pct_30d", "Source": "30-day percentage change in SMA200", "Consumer": row.get("consumer", ""), "Tier": "A"},
            {"Field": "drawdown_from_high_pct", "Source": "latest close versus highest available close", "Consumer": row.get("consumer", ""), "Tier": "A"},
        ]
    )
    st.dataframe(provenance, use_container_width=True, hide_index=True)

    if report.correlation_rows:
        st.subheader("Redundancy Check")
        st.caption("MM and distance from SMA200 are algebraically related; NERO reports both for readability but should not double-count them.")
        st.dataframe(pd.DataFrame(report.correlation_rows), use_container_width=True, hide_index=True)


def _display_number(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def _display_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"


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


    OPENING_TIMEFRAME_OPTIONS = {
        "1m": {"binance": "1m", "twelve": "1min", "candles": 500},
        "5m": {"binance": "5m", "twelve": "5min", "candles": 500},
        "15m": {"binance": "15m", "twelve": "15min", "candles": 500},
        "1H": {"binance": "1h", "twelve": "1h", "candles": 500},
        "4H": {"binance": "4h", "twelve": "4h", "candles": 500},
        "1D": {"binance": "1d", "twelve": "1day", "candles": 365},
        "1W": {"binance": "1w", "twelve": "1week", "candles": 260},
    }
    opening_timeframe_label = st.radio(
        "Opening chart timeframe",
        list(OPENING_TIMEFRAME_OPTIONS.keys()),
        index=3,
        horizontal=True,
        key="opening_chart_timeframe",
    )
    opening_timeframe_config = OPENING_TIMEFRAME_OPTIONS[opening_timeframe_label]
    opening_interval = opening_timeframe_config["twelve" if asset in {"GOLD", "OIL", "FDX"} else "binance"]
    opening_candles = int(opening_timeframe_config["candles"])
    market_data = market_client.load(
        asset=asset,
        prefer_live=prefer_live,
        days=365,
        twelve_data_api_key=twelve_data_api_key,
    )
    intraday_data = market_client.load_intraday(
        asset=asset,
        prefer_live=prefer_live,
        interval=opening_interval,
        candles=opening_candles,
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

    _render_opening_market_deck(asset, market_data, intraday_data, trade_plan, sentiment_result, opening_timeframe_label)

    verdict_tab, trade_tab, accountability_tab, mean_reversion_tab, strategy_audit_tab, research_lab_tab, evolution_tab, hypothesis_gate_tab, test_lab_tab, profit_edge_tab, market_memory_tab, cycle_intel_tab, quant_intel_tab, trade_path_tab, chat_tab, social_intel_tab, structure_tab, news_tab, knowledge_tab, backtest_tab, log_tab = st.tabs(
        ["Verdict", "Trade Desk", "Accountability", "Mean Reversion", "Strategy Audit", "Research Lab", "Evolution", "Hypothesis Gate", "TEST Lab", "Profit Engine", "Market Memory", "Cycle Intel", "Quant Intel", "Trade Path", "NERO Chat", "Social Intel", "Market Structure", "News", "Knowledge Store", "Backtest", "Prediction Log"]
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

    with research_lab_tab:
        _render_strategy_research_lab_tab()

    with evolution_tab:
        _render_strategy_evolution_tab()

    with hypothesis_gate_tab:
        _render_hypothesis_quality_gate_tab()

    with test_lab_tab:
        _render_strategy_test_lab_tab()

    with profit_edge_tab:
        _render_profit_edge_engine_tab()

    with market_memory_tab:
        _render_market_memory_tab(asset, enriched_headline)

    with cycle_intel_tab:
        _render_cycle_intelligence_tab(asset, price_history, f"{market_data.source} ({market_data.status})")

    with quant_intel_tab:
        _render_quant_intelligence_tab(asset, price_history, f"{market_data.source} ({market_data.status})", sentiment_result.sentiment_score)

    with trade_path_tab:
        _render_trade_path_tab(asset, price_history, f"{market_data.source} ({market_data.status})", sentiment_result.sentiment_score)

    chat_context = NeroChatContext(
        asset=asset,
        data_status=market_data.status,
        verdict_direction=result.verdict.direction,
        verdict_confidence=result.verdict.confidence,
        risk_score=result.verdict.risk_score,
        trade_action=trade_plan.action,
        trade_bias=trade_plan.bias,
        trade_confidence=trade_plan.confidence,
        entry_trigger=trade_plan.entry_trigger,
        stop_loss=trade_plan.stop_loss,
        take_profit_1=trade_plan.take_profit_1,
        take_profit_2=trade_plan.take_profit_2,
        invalidation=trade_plan.invalidation,
        consensus_class=consensus_decision.decision_class,
        consensus_quality=consensus_decision.trade_quality,
        consensus_direction=consensus_decision.direction,
        sentiment=sentiment_result.overall_sentiment,
        sentiment_score=sentiment_result.sentiment_score,
        confluence_score=result.assessment.confluence_score,
        market_regime=result.assessment.market_regime,
        volatility_regime=result.assessment.volatility_regime,
        blockers=consensus_decision.blockers,
        reasons=consensus_decision.reasons,
        test_lab=_load_strategy_lab_rows(),
    )
    with chat_tab:
        _render_nero_chat_tab(chat_context)



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
