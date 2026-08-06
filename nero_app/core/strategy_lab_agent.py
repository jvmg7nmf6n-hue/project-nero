"""Parallel Strategy Lab Agent for Project Nero.

Runs multiple research-only paper strategies side by side. No real orders,
no exchange trading keys, no auto-promotion of strategy rules.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from nero_app.core.mean_reversion_agent import (
    AgentRunSummary,
    MeanReversionAgent,
    MeanReversionConfig,
    add_indicators,
    apply_slippage,
    load_assets_from_env,
    report_row,
)
from nero_app.core.strategy_quarantine import DEFAULT_QUARANTINE_CSV, load_quarantined_strategy_ids
from nero_app.core.strategy_contracts import write_strategy_architecture_manifests
from nero_app.core.range_mean_reversion import (
    RangeMRConfig,
    _confirmation_entry_state,
    _entry_rejection_reasons,
    _entry_side,
    add_range_mr_indicators,
)

STRATEGY_LAB_VERSION = "strategy-lab-v1.0.0"
DEFAULT_LAB_DIR = Path(__file__).resolve().parents[1] / "data" / "strategy_lab"
DEFAULT_REPORT_DIR = Path("reports")

STRATEGY_LAB_DEFAULT_ASSETS = {
    # Crypto / 24-7 instruments
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "NEAR": "NEARUSDT",
    "BNB": "BNBUSDT",
    "PAXG": "PAXGUSDT",
    # Metals and energy instruments
    "GOLD": "XAU/USD",
    "SILVER": "XAG/USD",
    "OIL": "WTI/USD",
    "GOLD_FUT": "GC=F",
    "SILVER_FUT": "SI=F",
    "COPPER_FUT": "HG=F",
    "OIL_FUT": "CL=F",
    "BRENT_FUT": "BZ=F",
    # Stocks, ETFs and market proxies via yfinance
    "SPY": "SPY",
    "QQQ": "QQQ",
    "NVDA": "NVDA",
    "MSTR": "MSTR",
    "COIN": "COIN",
    "MARA": "MARA",
    "RIOT": "RIOT",
    "GLD": "GLD",
    "GDX": "GDX",
    "NEM": "NEM",
    # Dollar and FX pairs via yfinance
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
}

MARKET_HOURS_ASSETS = {
    "GOLD", "SILVER", "OIL", "GOLD_FUT", "SILVER_FUT", "COPPER_FUT", "OIL_FUT", "BRENT_FUT",
    "SPY", "QQQ", "NVDA", "MSTR", "COIN", "MARA", "RIOT", "GLD", "GDX", "NEM",
    "DXY", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
}

DEFAULT_QUARANTINED_ASSETS = {"ETH", "NEAR", "EURUSD", "GBPUSD", "AUDUSD", "COPPER_FUT"}

MANUAL_BLOCKED_STRATEGIES = {
    "BREAKOUT_MOMENTUM_V1",
    "MR_DEEP_VALUE_V1",
    "MR_REGIME_FILTER_V1",
    "MR_TARGET_1R_V1",
    "MR_RELAXED_PULLBACK_V1",
}


class TargetMode(str, Enum):
    FROZEN_MA20 = "FROZEN_MA20"
    FIXED_1R = "FIXED_1R"
    FIXED_125R = "FIXED_125R"
    FIXED_150R = "FIXED_150R"


class StrategyFamily(str, Enum):
    MEAN_REVERSION = "Mean Reversion"
    MOMENTUM = "Momentum"
    SHORT_MOMENTUM = "Short Momentum"
    EXIT_LOGIC = "Exit Logic"
    RANGE_MEAN_REVERSION = "Range Mean Reversion"
    PAIRS_RESEARCH = "Pairs Research"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    title: str
    display_label: str = ""
    bucket: str = "OLD_TEST"
    asset_filter: tuple[str, ...] = ()
    asset_exclude: tuple[str, ...] = ()
    interval: str = "1h"
    evidence_note: str = ""
    enabled: bool = True
    rsi_entry_below: float = 35.0
    lower_bb_buffer_atr: float = 0.0
    require_ma200: bool = True
    target_mode: str = "FROZEN_MA20"  # FROZEN_MA20 / FIXED_1R / FIXED_125R / FIXED_150R
    atr_stop_multiple: float = 1.5
    breakout_lookback: int = 20
    quant_gate: float | None = None
    require_rsi_recovery: bool = False
    require_breakout_retest: bool = False
    require_trend_support: bool = False
    min_planned_reward_r: float = 0.0
    max_atr_pct: float | None = None
    entry_side: str = "LONG"
    range_entry_mode: str = "BAND_EXTREME"
    range_min_band_atr: float = 0.0
    range_require_adx_falling: bool = False
    range_long_only: bool = False


CANDIDATES: dict[str, CandidateSpec] = {
    "MR_RELAXED_PULLBACK_V1": CandidateSpec(
        candidate_id="MR_RELAXED_PULLBACK_V1",
        family="Mean Reversion",
        title="Relaxed pullback",
        display_label="OLD_MR_RELAXED",
        rsi_entry_below=40.0,
        lower_bb_buffer_atr=0.25,
    ),
    "MR_REGIME_FILTER_V1": CandidateSpec(
        candidate_id="MR_REGIME_FILTER_V1",
        family="Mean Reversion",
        title="Regime-filtered pullback",
        display_label="OLD_MR_REGIME",
        rsi_entry_below=35.0,
        lower_bb_buffer_atr=0.1,
        require_ma200=True,
    ),
    "MR_DEEP_VALUE_V1": CandidateSpec(
        candidate_id="MR_DEEP_VALUE_V1",
        family="Mean Reversion",
        title="Deep value pullback",
        display_label="OLD_MR_DEEP",
        rsi_entry_below=30.0,
        lower_bb_buffer_atr=0.0,
    ),
    "MR_TARGET_1R_V1": CandidateSpec(
        candidate_id="MR_TARGET_1R_V1",
        family="Exit Logic",
        title="Fixed 1R target",
        display_label="OLD_MR_1R",
        rsi_entry_below=35.0,
        lower_bb_buffer_atr=0.0,
        target_mode="FIXED_1R",
    ),
    "BREAKOUT_MOMENTUM_V1": CandidateSpec(
        candidate_id="BREAKOUT_MOMENTUM_V1",
        family="Momentum",
        title="20-bar breakout momentum",
        display_label="OLD_BREAKOUT",
        rsi_entry_below=100.0,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.2,
        breakout_lookback=20,
    ),
    "NEW_BTC_12H_MR": CandidateSpec(
        candidate_id="NEW_BTC_12H_MR",
        family="Mean Reversion",
        title="BTC 12h relaxed mean reversion",
        display_label="NEW_BTC_12H_MR",
        bucket="NEW_TEST",
        asset_filter=("BTC",),
        interval="12h",
        evidence_note="Claude sweep: BTC/12h relaxed pullback positive in train and test.",
        rsi_entry_below=40.0,
        lower_bb_buffer_atr=0.25,
    ),
    "NEW_BNB_12H_TREND": CandidateSpec(
        candidate_id="NEW_BNB_12H_TREND",
        family="Momentum",
        title="BNB 12h trend pullback proxy",
        display_label="NEW_BNB_12H_TREND",
        bucket="NEW_TEST",
        asset_filter=("BNB",),
        interval="12h",
        evidence_note="Claude sweep: BNB/12h trend pullback positive in train and test. Running as current momentum proxy until dedicated pullback engine is ported.",
        rsi_entry_below=100.0,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.2,
        breakout_lookback=20,
    ),
    "NEW_BNB_12H_MR": CandidateSpec(
        candidate_id="NEW_BNB_12H_MR",
        family="Mean Reversion",
        title="BNB 12h relaxed mean reversion",
        display_label="NEW_BNB_12H_MR",
        bucket="NEW_TEST",
        asset_filter=("BNB",),
        interval="12h",
        evidence_note="Claude sweep: BNB/12h relaxed pullback positive in train and test.",
        rsi_entry_below=40.0,
        lower_bb_buffer_atr=0.25,
    ),
    "NEW_XRP_2H_MR": CandidateSpec(
        candidate_id="NEW_XRP_2H_MR",
        family="Mean Reversion",
        title="XRP 2h deep-value mean reversion",
        display_label="NEW_XRP_2H_MR",
        bucket="NEW_TEST",
        asset_filter=("XRP",),
        interval="2h",
        evidence_note="Claude sweep: XRP/2h deep-value positive in train and test.",
        rsi_entry_below=30.0,
        lower_bb_buffer_atr=0.0,
    ),
    "NEW_NEAR_2H_MR": CandidateSpec(
        candidate_id="NEW_NEAR_2H_MR",
        family="Mean Reversion",
        title="NEAR 2h deep-value mean reversion",
        display_label="NEW_NEAR_2H_MR",
        bucket="NEW_TEST",
        asset_filter=("NEAR",),
        interval="2h",
        evidence_note="Claude sweep: NEAR/2h deep-value positive in train and test.",
        rsi_entry_below=30.0,
        lower_bb_buffer_atr=0.0,
    ),

    "HYP_OIL_TREND_V1": CandidateSpec(
        candidate_id="HYP_OIL_TREND_V1",
        family="Momentum",
        title="Oil futures trend continuation",
        display_label="HYP_OIL_TREND",
        bucket="HYPOTHESIS_TEST",
        asset_filter=("OIL_FUT", "BRENT_FUT"),
        interval="1h",
        evidence_note="Asset Failure Correction: OIL_FUT and BRENT_FUT are the strongest current positive asset cluster. Test only clean 1h trend continuation in energy futures.",
        rsi_entry_below=100.0,
        require_ma200=True,
        target_mode="FIXED_150R",
        atr_stop_multiple=1.1,
        breakout_lookback=18,
        require_breakout_retest=True,
        require_trend_support=True,
        min_planned_reward_r=1.35,
        max_atr_pct=0.05,
    ),
    "HYP_OIL_MR_V1": CandidateSpec(
        candidate_id="HYP_OIL_MR_V1",
        family="Mean Reversion",
        title="Oil futures recovery mean reversion",
        display_label="HYP_OIL_MR",
        bucket="HYPOTHESIS_TEST",
        asset_filter=("OIL_FUT", "BRENT_FUT"),
        interval="1h",
        evidence_note="Asset Failure Correction: test whether 1h oil pullbacks work better after RSI recovery and a minimum reward gate.",
        rsi_entry_below=38.0,
        lower_bb_buffer_atr=0.2,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.25,
        require_rsi_recovery=True,
        require_trend_support=True,
        min_planned_reward_r=1.2,
        max_atr_pct=0.045,
    ),
    "RMR_LONG_ONLY_EURUSD_4H": CandidateSpec(
        candidate_id="RMR_LONG_ONLY_EURUSD_4H",
        family="Range Mean Reversion",
        title="EURUSD 4h range MR long-only",
        display_label="RMR_LONG_EURUSD_4H",
        bucket="RANGE_MR_WATCHLIST",
        asset_filter=("EURUSD",),
        interval="4h",
        evidence_note="Range MR hypothesis split: EURUSD 4h long-only showed strongest watchlist edge. Forward-test only until 30-50 trades.",
        atr_stop_multiple=2.0,
        range_long_only=True,
    ),
    "RMR_ADX_FALLING_ETH_4H": CandidateSpec(
        candidate_id="RMR_ADX_FALLING_ETH_4H",
        family="Range Mean Reversion",
        title="ETH 4h range MR with falling ADX",
        display_label="RMR_ADX_ETH_4H",
        bucket="RANGE_MR_WATCHLIST",
        asset_filter=("ETH",),
        interval="4h",
        evidence_note="Range MR hypothesis split: ETH 4h improved when ADX was falling into range conditions.",
        atr_stop_multiple=2.0,
        range_require_adx_falling=True,
    ),
    "RMR_LONG_ONLY_BTC_1D": CandidateSpec(
        candidate_id="RMR_LONG_ONLY_BTC_1D",
        family="Range Mean Reversion",
        title="BTC daily range MR long-only",
        display_label="RMR_LONG_BTC_1D",
        bucket="RANGE_MR_WATCHLIST",
        asset_filter=("BTC",),
        interval="1d",
        evidence_note="Range MR hypothesis split: BTC daily long-only is promising but sample-limited.",
        atr_stop_multiple=2.0,
        range_long_only=True,
    ),
    "RMR_CONFIRMATION_BTC_1D": CandidateSpec(
        candidate_id="RMR_CONFIRMATION_BTC_1D",
        family="Range Mean Reversion",
        title="BTC daily range MR confirmation entry",
        display_label="RMR_CONFIRM_BTC_1D",
        bucket="RANGE_MR_WATCHLIST",
        asset_filter=("BTC",),
        interval="1d",
        evidence_note="Range MR hypothesis split: BTC daily confirmation entry tests whether waiting after the band breach reduces weak entries.",
        atr_stop_multiple=2.0,
        range_entry_mode="CONFIRMATION",
    ),

    "NEW_BTC_ETH_12H_PAIR": CandidateSpec(
        candidate_id="NEW_BTC_ETH_12H_PAIR",
        family="Pairs Research",
        title="BTC-ETH 12h cointegration pair",
        display_label="NEW_BTC_ETH_12H_PAIR",
        bucket="RESEARCH_ONLY",
        asset_filter=("BTC", "ETH"),
        interval="12h",
        evidence_note="Claude sweep: BTC-ETH/12h pair positive but weak. Research-only report cycle until real pair execution is wired.",
        enabled=True,
    ),

    "V2_BREAKOUT_RETEST": CandidateSpec(
        candidate_id="V2_BREAKOUT_RETEST",
        family="Momentum",
        title="Breakout retest with volatility block",
        display_label="V2_BREAKOUT_RETEST",
        bucket="V2_SHADOW",
        interval="1h",
        evidence_note="Loss autopsy: old breakout suffered fakeouts. V2 requires a cleaner retest, trend support, normal volatility, and a 1.5R target.",
        rsi_entry_below=100.0,
        require_ma200=True,
        target_mode="FIXED_150R",
        atr_stop_multiple=1.2,
        breakout_lookback=20,
        require_breakout_retest=True,
        require_trend_support=True,
        max_atr_pct=0.045,
    ),
    "V2_MR_RECOVERY": CandidateSpec(
        candidate_id="V2_MR_RECOVERY",
        family="Mean Reversion",
        title="Mean reversion after RSI recovery",
        display_label="V2_MR_RECOVERY",
        bucket="V2_SHADOW",
        interval="1h",
        evidence_note="Loss autopsy: relaxed pullback entered falling knives. V2 waits for RSI recovery and minimum reward.",
        rsi_entry_below=40.0,
        lower_bb_buffer_atr=0.25,
        require_rsi_recovery=True,
        min_planned_reward_r=1.2,
        max_atr_pct=0.055,
    ),
    "V2_MR_REGIME": CandidateSpec(
        candidate_id="V2_MR_REGIME",
        family="Mean Reversion",
        title="Mean reversion with tighter regime filter",
        display_label="V2_MR_REGIME",
        bucket="V2_SHADOW",
        interval="1h",
        evidence_note="Loss autopsy: regime-filtered MR still bought weak structure. V2 requires MA20 above MA200, RSI recovery, and reward quality.",
        rsi_entry_below=35.0,
        lower_bb_buffer_atr=0.1,
        require_ma200=True,
        require_trend_support=True,
        require_rsi_recovery=True,
        min_planned_reward_r=1.2,
        max_atr_pct=0.05,
    ),
    "V2_MR_DEEP": CandidateSpec(
        candidate_id="V2_MR_DEEP",
        family="Mean Reversion",
        title="Deep value with recovery confirmation",
        display_label="V2_MR_DEEP",
        bucket="V2_SHADOW",
        interval="2h",
        evidence_note="Loss autopsy: deep value had too few but sharp losses. V2 keeps deep RSI but waits for recovery and avoids volatility shock.",
        rsi_entry_below=30.0,
        lower_bb_buffer_atr=0.0,
        require_rsi_recovery=True,
        min_planned_reward_r=1.2,
        max_atr_pct=0.06,
    ),
    "V2_MR_REWARD": CandidateSpec(
        candidate_id="V2_MR_REWARD",
        family="Exit Logic",
        title="Mean reversion with 1.25R reward gate",
        display_label="V2_MR_REWARD",
        bucket="V2_SHADOW",
        interval="1h",
        evidence_note="Loss autopsy: fixed 1R did not pay enough after fees. V2 requires 1.25R target and rejects weak planned reward.",
        rsi_entry_below=35.0,
        lower_bb_buffer_atr=0.0,
        target_mode="FIXED_125R",
        require_rsi_recovery=True,
        min_planned_reward_r=1.2,
        max_atr_pct=0.055,
    ),
    "REPAIR_BREAKOUT_QUALITY_V1": CandidateSpec(
        candidate_id="REPAIR_BREAKOUT_QUALITY_V1",
        family="Momentum",
        title="Breakout repair with retest and quality gates",
        display_label="FIX_BREAKOUT_QUALITY",
        bucket="LOSS_REPAIR_TEST",
        interval="4h",
        evidence_note="Loss autopsy: OLD_BREAKOUT had too many fakeouts and SLs. Repair waits for trend support, breakout retest, reward quality, and calmer volatility.",
        asset_filter=("BTC", "XRP", "BNB", "PAXG"),
        rsi_entry_below=100.0,
        require_ma200=True,
        target_mode="FIXED_150R",
        atr_stop_multiple=1.0,
        breakout_lookback=30,
        require_breakout_retest=True,
        require_trend_support=True,
        min_planned_reward_r=1.35,
        max_atr_pct=0.04,
    ),
    "REPAIR_MR_REGIME_LATE_V1": CandidateSpec(
        candidate_id="REPAIR_MR_REGIME_LATE_V1",
        family="Mean Reversion",
        title="Regime MR repair with late confirmation",
        display_label="FIX_MR_LATE",
        bucket="LOSS_REPAIR_TEST",
        interval="4h",
        evidence_note="Loss autopsy: OLD_MR_REGIME entered weak falling candles too early. Repair requires RSI/close recovery, trend support, reward quality, and rejects known weak-loss assets.",
        asset_filter=("BTC", "SOL", "XRP", "DOGE", "BNB", "PAXG"),
        rsi_entry_below=32.0,
        lower_bb_buffer_atr=0.15,
        require_ma200=True,
        require_trend_support=True,
        require_rsi_recovery=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.0,
        min_planned_reward_r=1.2,
        max_atr_pct=0.045,
    ),
    "REPAIR_MR_1R_ASYMMETRIC_V1": CandidateSpec(
        candidate_id="REPAIR_MR_1R_ASYMMETRIC_V1",
        family="Exit Logic",
        title="1R repair with asymmetric reward",
        display_label="FIX_MR_1R_ASYM",
        bucket="LOSS_REPAIR_TEST",
        interval="4h",
        evidence_note="Loss autopsy: OLD_MR_1R could not overcome fees and outsized SLs. Repair uses 1.5R target, 1.0 ATR stop, and recovery confirmation.",
        asset_filter=("BTC", "SOL", "XRP", "DOGE", "BNB", "PAXG"),
        rsi_entry_below=35.0,
        lower_bb_buffer_atr=0.0,
        require_rsi_recovery=True,
        target_mode="FIXED_150R",
        atr_stop_multiple=1.0,
        min_planned_reward_r=1.4,
        max_atr_pct=0.045,
    ),
    "SHORT_BTC_BREAKDOWN_4H": CandidateSpec(
        candidate_id="SHORT_BTC_BREAKDOWN_4H",
        family="Short Momentum",
        title="BTC 4h breakdown short",
        display_label="SHORT_BTC_4H",
        bucket="SHORT_SIDE_TEST",
        asset_filter=("BTC",),
        interval="4h",
        evidence_note="Short-side maturity: tests whether BTC breakdowns below recent lows produce cleaner paper shorts than forced long-only entries.",
        rsi_entry_below=45.0,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.3,
        breakout_lookback=20,
        require_trend_support=True,
        min_planned_reward_r=1.1,
        max_atr_pct=0.06,
        entry_side="SHORT",
    ),
    "SHORT_ETH_BREAKDOWN_4H": CandidateSpec(
        candidate_id="SHORT_ETH_BREAKDOWN_4H",
        family="Short Momentum",
        title="ETH 4h breakdown short",
        display_label="SHORT_ETH_4H",
        bucket="SHORT_SIDE_TEST",
        asset_filter=("ETH",),
        interval="4h",
        evidence_note="Short-side maturity: ETH breakdown paper test with trend, volatility, and reward gates.",
        rsi_entry_below=45.0,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.3,
        breakout_lookback=20,
        require_trend_support=True,
        min_planned_reward_r=1.1,
        max_atr_pct=0.065,
        entry_side="SHORT",
    ),
    "SHORT_SOL_BREAKDOWN_4H": CandidateSpec(
        candidate_id="SHORT_SOL_BREAKDOWN_4H",
        family="Short Momentum",
        title="SOL 4h breakdown short",
        display_label="SHORT_SOL_4H",
        bucket="SHORT_SIDE_TEST",
        asset_filter=("SOL",),
        interval="4h",
        evidence_note="Short-side maturity: SOL is higher beta, so this validates whether breakdown shorts survive after fees and stops.",
        rsi_entry_below=45.0,
        require_ma200=True,
        target_mode="FIXED_150R",
        atr_stop_multiple=1.25,
        breakout_lookback=18,
        require_trend_support=True,
        min_planned_reward_r=1.25,
        max_atr_pct=0.08,
        entry_side="SHORT",
    ),
    "SHORT_OIL_BREAKDOWN_1H": CandidateSpec(
        candidate_id="SHORT_OIL_BREAKDOWN_1H",
        family="Short Momentum",
        title="Oil futures 1h breakdown short",
        display_label="SHORT_OIL_1H",
        bucket="SHORT_SIDE_TEST",
        asset_filter=("OIL_FUT", "BRENT_FUT"),
        interval="1h",
        evidence_note="Short-side maturity: oil already has live futures data in the lab; test downside continuation instead of only recovery longs.",
        rsi_entry_below=45.0,
        require_ma200=True,
        target_mode="FIXED_125R",
        atr_stop_multiple=1.2,
        breakout_lookback=18,
        require_trend_support=True,
        min_planned_reward_r=1.1,
        max_atr_pct=0.055,
        entry_side="SHORT",
    ),
}


class SignalValidator:
    """Family-aware entry validator for Strategy Lab candidates.

    Mean-reversion and momentum candidates intentionally look for opposite
    market behavior, so their entry checks stay separated here.
    """

    def __init__(self, spec: CandidateSpec, slippage_bps: float = 0.0) -> None:
        self.spec = spec
        self.slippage_bps = slippage_bps

    def validate(self, candle: pd.Series, state: dict[str, Any], daily_loss_guard_r: float) -> tuple[list[str], float]:
        reasons: list[str] = []
        if state.get("open_trade"):
            reasons.append("OPEN_TRADE_EXISTS")
        if float(state.get("daily_r", 0.0)) <= daily_loss_guard_r:
            reasons.append("DAILY_LOSS_GUARD")

        close = float(candle["close"])
        ma20 = float(candle["ma20"])
        ma200 = float(candle["ma200"])
        atr = float(candle["atr"])
        atr_pct = atr / close if close else 0.0
        planned_reward_r = self.planned_reward_r(candle)

        if self.spec.max_atr_pct is not None and atr_pct > self.spec.max_atr_pct:
            reasons.append("VOLATILITY_SHOCK")
        if self.spec.require_trend_support and not self._is_short_momentum() and ma20 <= ma200:
            reasons.append("TREND_SUPPORT_NOT_CONFIRMED")
        if self.spec.min_planned_reward_r and planned_reward_r < self.spec.min_planned_reward_r:
            reasons.append("PLANNED_REWARD_TOO_LOW")

        if self._is_momentum():
            reasons.extend(self._validate_momentum(candle))
        elif self._is_short_momentum():
            reasons.extend(self._validate_short_momentum(candle))
        elif self._is_mean_reversion_like():
            reasons.extend(self._validate_mean_reversion(candle))
        else:
            reasons.append(f"UNKNOWN_FAMILY:{self.spec.family}")

        return reasons, planned_reward_r

    def planned_reward_r(self, candle: pd.Series) -> float:
        side = self._entry_side()
        entry_price = apply_slippage(float(candle["close"]), self.slippage_bps, "sell" if side == "SHORT" else "buy")
        risk_per_unit = self.risk_per_unit(candle)
        if risk_per_unit <= 0:
            return 0.0
        target = self.target_price(candle, entry_price, risk_per_unit)
        reward_per_unit = entry_price - target if side == "SHORT" else target - entry_price
        return max(0.0, reward_per_unit / risk_per_unit)

    def risk_per_unit(self, candle: pd.Series) -> float:
        side = self._entry_side()
        entry_price = apply_slippage(float(candle["close"]), self.slippage_bps, "sell" if side == "SHORT" else "buy")
        stop_loss = entry_price + self.spec.atr_stop_multiple * float(candle["atr"]) if side == "SHORT" else entry_price - self.spec.atr_stop_multiple * float(candle["atr"])
        return abs(entry_price - stop_loss)

    def target_price(self, candle: pd.Series, entry_price: float, risk_per_unit: float) -> float:
        target_mode = _target_mode_value(self.spec.target_mode)
        side = self._entry_side()
        direction = -1 if side == "SHORT" else 1
        if target_mode == TargetMode.FIXED_1R.value:
            return entry_price + direction * risk_per_unit
        if target_mode == TargetMode.FIXED_125R.value:
            return entry_price + direction * 1.25 * risk_per_unit
        if target_mode == TargetMode.FIXED_150R.value:
            return entry_price + direction * 1.5 * risk_per_unit
        return float(candle["ma20"])

    def _validate_momentum(self, candle: pd.Series) -> list[str]:
        reasons: list[str] = []
        breakout_high = candle.get("breakout_high")
        close = float(candle["close"])
        if pd.isna(breakout_high) or close <= float(breakout_high):
            reasons.append("CLOSE_NOT_ABOVE_BREAKOUT_HIGH")
        elif self.spec.require_breakout_retest and float(candle["low"]) > float(breakout_high) * 1.003:
            reasons.append("BREAKOUT_RETEST_NOT_CONFIRMED")
        if self.spec.require_ma200 and close <= float(candle["ma200"]):
            reasons.append("CLOSE_NOT_ABOVE_MA200")
        if float(candle["rsi"]) < 50:
            reasons.append("RSI_NOT_MOMENTUM_SUPPORTIVE")
        return reasons

    def _validate_mean_reversion(self, candle: pd.Series) -> list[str]:
        reasons: list[str] = []
        close = float(candle["close"])
        if float(candle["rsi"]) >= self.spec.rsi_entry_below:
            reasons.append(f"RSI_NOT_BELOW_{int(self.spec.rsi_entry_below)}")
        lower_threshold = float(candle["bb_lower"]) + self.spec.lower_bb_buffer_atr * float(candle["atr"])
        if close >= lower_threshold:
            reasons.append("CLOSE_NOT_NEAR_OR_BELOW_LOWER_BB")
        if self.spec.require_ma200 and close <= float(candle["ma200"]):
            reasons.append("CLOSE_NOT_ABOVE_MA200")
        if _target_mode_value(self.spec.target_mode) == TargetMode.FROZEN_MA20.value and float(candle["ma20"]) <= close:
            reasons.append("TARGET_NOT_ABOVE_ENTRY")
        if self.spec.require_rsi_recovery:
            rsi_prev = candle.get("rsi_prev")
            close_prev = candle.get("close_prev")
            if pd.isna(rsi_prev) or pd.isna(close_prev) or not (float(candle["rsi"]) > float(rsi_prev) and close >= float(close_prev)):
                reasons.append("RSI_RECOVERY_NOT_CONFIRMED")
        return reasons

    def _validate_short_momentum(self, candle: pd.Series) -> list[str]:
        reasons: list[str] = []
        breakdown_low = candle.get("breakdown_low")
        close = float(candle["close"])
        ma20 = float(candle["ma20"])
        ma200 = float(candle["ma200"])
        if pd.isna(breakdown_low) or close >= float(breakdown_low):
            reasons.append("CLOSE_NOT_BELOW_BREAKDOWN_LOW")
        elif self.spec.require_breakout_retest and float(candle["high"]) < float(breakdown_low) * 0.997:
            reasons.append("BREAKDOWN_RETEST_NOT_CONFIRMED")
        if self.spec.require_ma200 and close >= ma200:
            reasons.append("CLOSE_NOT_BELOW_MA200")
        if self.spec.require_trend_support and ma20 >= ma200:
            reasons.append("DOWNTREND_SUPPORT_NOT_CONFIRMED")
        if float(candle["rsi"]) > self.spec.rsi_entry_below:
            reasons.append(f"RSI_NOT_BELOW_{int(self.spec.rsi_entry_below)}")
        return reasons
    def _is_momentum(self) -> bool:
        return self.spec.family == StrategyFamily.MOMENTUM.value

    def _is_short_momentum(self) -> bool:
        return self.spec.family == StrategyFamily.SHORT_MOMENTUM.value

    def _is_mean_reversion_like(self) -> bool:
        return self.spec.family in {StrategyFamily.MEAN_REVERSION.value, StrategyFamily.EXIT_LOGIC.value}

    def _entry_side(self) -> str:
        return str(self.spec.entry_side or "LONG").upper()


def _target_mode_value(target_mode: str | TargetMode) -> str:
    return target_mode.value if isinstance(target_mode, TargetMode) else str(target_mode)


@dataclass(frozen=True)
class StrategyLabRunSummary:
    evaluated: int
    entries: int
    exits: int
    alerts: list[str]
    candidate_count: int


class CandidatePaperAgent(MeanReversionAgent):
    def __init__(self, spec: CandidateSpec, config: MeanReversionConfig, data_dir: Path, report_dir: Path, now: datetime | None = None) -> None:
        self.spec = spec
        super().__init__(config=config, data_dir=data_dir, report_dir=report_dir, now=now)

    def process_asset(self, asset: str, symbol: str, candles: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
        enriched = add_indicators(candles, self.config)
        enriched["breakout_high"] = enriched["high"].shift(1).rolling(self.spec.breakout_lookback).max()
        enriched["breakdown_low"] = enriched["low"].shift(1).rolling(self.spec.breakout_lookback).min()
        enriched["rsi_prev"] = enriched["rsi"].shift(1)
        enriched["close_prev"] = enriched["close"].shift(1)
        enriched["atr_pct"] = enriched["atr"] / enriched["close"]
        last_seen = int(state.get("last_evaluated_close_time", 0))
        rows = enriched[enriched["close_time"] > last_seen].copy()
        rows = rows.dropna(subset=["rsi", "bb_lower", "ma20", "ma200", "atr"])
        rows = rows.sort_values("close_time")
        entries = 0
        exits = 0
        evaluated = 0
        alerts: list[str] = []

        for _, candle in rows.iterrows():
            candle_time = int(candle["close_time"])
            state = self._reset_daily_guard_if_needed(state, candle)
            exit_event = self._maybe_exit(asset, symbol, candle, state)
            if exit_event:
                exits += 1
                alerts.append(f"{self.spec.candidate_id} {asset}: {exit_event['exit_reason']} net={exit_event['net_pnl']:.2f} R={exit_event['r_multiple']:.2f}")

            evaluation = self._evaluate_entry(asset, symbol, candle, state)
            self._append_rows(self.trade_dir / "evaluations.csv", [evaluation])
            evaluated += 1
            if evaluation["passed"]:
                entry_event = self._enter_trade(asset, symbol, candle, state)
                if entry_event:
                    entries += 1
                    alerts.append(f"{self.spec.candidate_id} {asset}: PAPER_ENTRY entry={entry_event['entry_price']:.4f} target={entry_event['target']:.4f}")
            state["last_evaluated_close_time"] = candle_time

        return {"state": state, "evaluated": evaluated, "entries": entries, "exits": exits, "alerts": alerts}

    def _evaluate_entry(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any]:
        validator = SignalValidator(self.spec, slippage_bps=self.config.slippage_bps)
        reasons, planned_reward_r = validator.validate(candle, state, self.config.daily_loss_guard_r)
        atr_pct = float(candle.get("atr_pct", 0.0) or 0.0)
        passed = not reasons
        return {
            "timestamp": self.now.isoformat(),
            "candidate_id": self.spec.candidate_id,
            "asset": asset,
            "symbol": symbol,
            "strategy_version": self.config.strategy_version,
            "candle_close_time": int(candle["close_time"]),
            "candle_time": candle["date"].isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "rsi": float(candle["rsi"]),
            "ma20": float(candle["ma20"]),
            "bb_lower": float(candle["bb_lower"]),
            "ma200": float(candle["ma200"]),
            "atr": float(candle["atr"]),
            "atr_pct": atr_pct,
            "rsi_prev": float(candle["rsi_prev"]) if not pd.isna(candle.get("rsi_prev")) else "",
            "planned_reward_r": planned_reward_r,
            "passed": passed,
            "rejection_reasons": "|".join(reasons),
        }

    def _planned_reward_r(self, candle: pd.Series) -> float:
        return SignalValidator(self.spec, slippage_bps=self.config.slippage_bps).planned_reward_r(candle)

    def _enter_trade(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("open_trade"):
            return None
        equity = float(state.get("equity", self.config.initial_equity))
        side = str(self.spec.entry_side or "LONG").upper()
        raw_entry = float(candle["close"])
        entry_price = apply_slippage(raw_entry, self.config.slippage_bps, "sell" if side == "SHORT" else "buy")
        atr = float(candle["atr"])
        stop_loss = entry_price + self.spec.atr_stop_multiple * atr if side == "SHORT" else entry_price - self.spec.atr_stop_multiple * atr
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return None
        target_mode = _target_mode_value(self.spec.target_mode)
        target = SignalValidator(self.spec, slippage_bps=self.config.slippage_bps).target_price(candle, entry_price, risk_per_unit)
        reward_per_unit = entry_price - target if side == "SHORT" else target - entry_price
        if reward_per_unit <= 0:
            return None
        risk_dollars = equity * self.config.risk_per_trade
        quantity = risk_dollars / risk_per_unit
        max_notional = equity * self.config.max_notional_pct
        notional = quantity * entry_price
        if notional > max_notional:
            quantity = max_notional / entry_price
            notional = max_notional
            risk_dollars = quantity * risk_per_unit
        fees = notional * self.config.fee_bps / 10000.0
        trade_id = f"SLAB-{self.spec.candidate_id}-{asset}-{int(candle['close_time'])}"
        trade = {
            "trade_id": trade_id,
            "candidate_id": self.spec.candidate_id,
            "family": self.spec.family,
            "asset": asset,
            "symbol": symbol,
            "side": side,
            "strategy_version": self.config.strategy_version,
            "status": "OPEN",
            "opened_at": candle["date"].isoformat(),
            "open_close_time": int(candle["close_time"]),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "target_mode": target_mode,
            "quantity": quantity,
            "notional": notional,
            "risk_dollars": risk_dollars,
            "entry_fee": fees,
            "entry_slippage_bps": self.config.slippage_bps,
            "planned_reward_r": reward_per_unit / risk_per_unit,
            "entry_rsi": float(candle["rsi"]),
            "entry_ma20": float(candle["ma20"]),
            "entry_bb_lower": float(candle["bb_lower"]),
            "entry_ma200": float(candle["ma200"]),
            "entry_atr": atr,
        }
        state["open_trade"] = trade
        self._append_rows(self.trade_dir / "trade_events.csv", [{**trade, "event": "ENTRY", "timestamp": self.now.isoformat()}])
        return trade

    def _maybe_exit(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        trade = state.get("open_trade")
        if not trade:
            return None
        side = str(trade.get("side", "LONG")).upper()
        candle_time = int(candle["close_time"])
        hours_held = (candle_time - int(trade["open_close_time"])) / 3600000.0
        stop_loss = float(trade["stop_loss"])
        target = float(trade["target"])
        low = float(candle["low"])
        high = float(candle["high"])
        close = float(candle["close"])
        exit_reason = ""
        raw_exit = close
        if side == "SHORT":
            if high >= stop_loss and low <= target:
                exit_reason, raw_exit = "SL", stop_loss
            elif high >= stop_loss:
                exit_reason, raw_exit = "SL", stop_loss
            elif low <= target:
                exit_reason, raw_exit = "TARGET", target
        else:
            if low <= stop_loss and high >= target:
                exit_reason, raw_exit = "SL", stop_loss
            elif low <= stop_loss:
                exit_reason, raw_exit = "SL", stop_loss
            elif high >= target:
                exit_reason, raw_exit = "TARGET", target
        if not exit_reason and hours_held >= self.config.max_holding_hours:
            exit_reason, raw_exit = "TIME", close
        if not exit_reason:
            return None

        exit_price = apply_slippage(raw_exit, self.config.slippage_bps, "buy" if side == "SHORT" else "sell")
        quantity = float(trade["quantity"])
        gross_pnl = (float(trade["entry_price"]) - exit_price) * quantity if side == "SHORT" else (exit_price - float(trade["entry_price"])) * quantity
        exit_fee = exit_price * quantity * self.config.fee_bps / 10000.0
        total_fees = float(trade["entry_fee"]) + exit_fee
        net_pnl = gross_pnl - total_fees
        risk_dollars = max(float(trade["risk_dollars"]), 1e-9)
        r_multiple = net_pnl / risk_dollars
        equity = float(state.get("equity", self.config.initial_equity)) + net_pnl
        state["equity"] = equity
        state["daily_r"] = float(state.get("daily_r", 0.0)) + r_multiple
        state["open_trade"] = None
        event = {
            **trade,
            "event": "EXIT",
            "timestamp": self.now.isoformat(),
            "status": "CLOSED",
            "closed_at": candle["date"].isoformat(),
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "exit_fee": exit_fee,
            "fees": total_fees,
            "slippage_bps": self.config.slippage_bps,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
            "holding_hours": hours_held,
        }
        self._append_rows(self.trade_dir / "trade_events.csv", [event])
        self._append_rows(self.trade_dir / "closed_trades.csv", [event])
        return event

    def write_report(self) -> None:
        trades_path = self.trade_dir / "closed_trades.csv"
        evaluations_path = self.trade_dir / "evaluations.csv"
        trades = _safe_read_csv(trades_path)
        evaluations = _safe_read_csv(evaluations_path)
        rows = []
        assets = sorted(set(self.config.assets.keys()) | _csv_asset_values(trades))
        for asset in assets:
            rows.append(_candidate_report_row(self.spec, asset, trades[trades["asset"] == asset] if not trades.empty and "asset" in trades else pd.DataFrame(), evaluations))
        rows.append(_candidate_report_row(self.spec, "COMBINED", trades, evaluations))
        report = pd.DataFrame(rows)
        report.to_csv(self.report_dir / f"strategy_lab_{self.spec.candidate_id}.csv", index=False)
        (self.report_dir / f"strategy_lab_{self.spec.candidate_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")



class RangeMRPaperAgent(CandidatePaperAgent):
    def process_asset(self, asset: str, symbol: str, candles: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
        range_config = _range_config_from_spec(self.spec)
        enriched = add_range_mr_indicators(candles, range_config)
        last_seen = int(state.get("last_evaluated_close_time", 0))
        rows = enriched[enriched["close_time"] > last_seen].copy()
        rows = rows.dropna(subset=["ma20", "bb_upper", "bb_lower", "atr", "adx"])
        rows = rows.sort_values("close_time")
        entries = 0
        exits = 0
        evaluated = 0
        alerts: list[str] = []

        for _, candle in rows.iterrows():
            candle_time = int(candle["close_time"])
            state = self._reset_daily_guard_if_needed(state, candle)
            exit_event = self._maybe_exit(asset, symbol, candle, state)
            if exit_event:
                exits += 1
                alerts.append(f"{self.spec.candidate_id} {asset}: {exit_event['exit_reason']} net={exit_event['net_pnl']:.2f} R={exit_event['r_multiple']:.2f}")

            evaluation = self._evaluate_entry(asset, symbol, candle, state)
            self._append_rows(self.trade_dir / "evaluations.csv", [evaluation])
            evaluated += 1
            if evaluation["passed"]:
                entry_event = self._enter_trade(asset, symbol, candle, state)
                if entry_event:
                    entries += 1
                    alerts.append(f"{self.spec.candidate_id} {asset}: PAPER_ENTRY {entry_event['side']} entry={entry_event['entry_price']:.4f} target={entry_event['target']:.4f}")
            state["last_evaluated_close_time"] = candle_time

        return {"state": state, "evaluated": evaluated, "entries": entries, "exits": exits, "alerts": alerts}

    def _evaluate_entry(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any]:
        range_config = _range_config_from_spec(self.spec)
        pending = state.get("range_pending_setup") if isinstance(state.get("range_pending_setup"), dict) else None
        if range_config.entry_mode == "CONFIRMATION":
            reasons, side, pending = _confirmation_entry_state(candle, state.get("open_trade"), pending, range_config)
            state["range_pending_setup"] = pending
        else:
            reasons = _entry_rejection_reasons(candle, state.get("open_trade"), range_config)
            side = _entry_side(candle, range_config) if not reasons else ""
        planned_reward_r = _range_planned_reward_r(candle, side, self.config.slippage_bps, range_config) if side else 0.0
        return {
            "timestamp": self.now.isoformat(),
            "candidate_id": self.spec.candidate_id,
            "asset": asset,
            "symbol": symbol,
            "strategy_version": self.config.strategy_version,
            "candle_close_time": int(candle["close_time"]),
            "candle_time": candle["date"].isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "side": side,
            "adx": float(candle["adx"]),
            "adx_falling_3": bool(candle.get("adx_falling_3", False)),
            "ma20": float(candle["ma20"]),
            "bb_lower": float(candle["bb_lower"]),
            "bb_upper": float(candle["bb_upper"]),
            "bb_width_pct": float(candle["bb_width_pct"]),
            "band_distance_atr": float(candle.get("band_distance_atr", 0.0)),
            "atr": float(candle["atr"]),
            "planned_reward_r": planned_reward_r,
            "passed": not reasons,
            "rejection_reasons": "|".join(reasons),
        }

    def _enter_trade(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("open_trade"):
            return None
        range_config = _range_config_from_spec(self.spec)
        side = _entry_side(candle, range_config)
        equity = float(state.get("equity", self.config.initial_equity))
        raw_entry = float(candle["close"])
        entry_price = apply_slippage(raw_entry, self.config.slippage_bps, "buy" if side == "LONG" else "sell")
        atr = float(candle["atr"])
        stop_loss = entry_price - range_config.atr_stop_multiple * atr if side == "LONG" else entry_price + range_config.atr_stop_multiple * atr
        target = float(candle["ma20"])
        risk_per_unit = abs(entry_price - stop_loss)
        reward_per_unit = target - entry_price if side == "LONG" else entry_price - target
        if risk_per_unit <= 0 or reward_per_unit <= 0:
            return None
        risk_dollars = equity * self.config.risk_per_trade
        quantity = risk_dollars / risk_per_unit
        max_notional = equity * self.config.max_notional_pct
        notional = quantity * entry_price
        if notional > max_notional:
            quantity = max_notional / entry_price
            notional = max_notional
            risk_dollars = quantity * risk_per_unit
        fees = notional * self.config.fee_bps / 10000.0
        trade_id = f"SLAB-{self.spec.candidate_id}-{asset}-{int(candle['close_time'])}"
        trade = {
            "trade_id": trade_id,
            "candidate_id": self.spec.candidate_id,
            "family": self.spec.family,
            "asset": asset,
            "symbol": symbol,
            "side": side,
            "strategy_version": self.config.strategy_version,
            "status": "OPEN",
            "opened_at": candle["date"].isoformat(),
            "open_close_time": int(candle["close_time"]),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "target_mode": "DYNAMIC_MA20",
            "quantity": quantity,
            "notional": notional,
            "risk_dollars": risk_dollars,
            "entry_fee": fees,
            "entry_slippage_bps": self.config.slippage_bps,
            "planned_reward_r": reward_per_unit / risk_per_unit,
            "entry_adx": float(candle["adx"]),
            "entry_ma20": float(candle["ma20"]),
            "entry_bb_lower": float(candle["bb_lower"]),
            "entry_bb_upper": float(candle["bb_upper"]),
            "entry_band_distance_atr": float(candle.get("band_distance_atr", 0.0)),
            "entry_atr": atr,
        }
        state["open_trade"] = trade
        state["range_pending_setup"] = None
        state["range_adx_break_count"] = 0
        self._append_rows(self.trade_dir / "trade_events.csv", [{**trade, "event": "ENTRY", "timestamp": self.now.isoformat()}])
        return trade

    def _maybe_exit(self, asset: str, symbol: str, candle: pd.Series, state: dict[str, Any]) -> dict[str, Any] | None:
        trade = state.get("open_trade")
        if not trade:
            return None
        side = str(trade.get("side", "LONG")).upper()
        stop_loss = float(trade["stop_loss"])
        low = float(candle["low"])
        high = float(candle["high"])
        close = float(candle["close"])
        ma20 = float(candle["ma20"])
        adx_break_count = int(state.get("range_adx_break_count", 0))
        adx_break_count = adx_break_count + 1 if float(candle["adx"]) >= 28.0 else 0
        state["range_adx_break_count"] = adx_break_count
        exit_reason = ""
        raw_exit = close
        if side == "LONG":
            if low <= stop_loss:
                exit_reason, raw_exit = "SL", stop_loss
            elif close >= ma20:
                exit_reason, raw_exit = "TARGET", ma20
        else:
            if high >= stop_loss:
                exit_reason, raw_exit = "SL", stop_loss
            elif close <= ma20:
                exit_reason, raw_exit = "TARGET", ma20
        if not exit_reason and adx_break_count >= 2:
            exit_reason, raw_exit = "REGIME_BREAK", close
        if not exit_reason:
            return None

        exit_price = apply_slippage(raw_exit, self.config.slippage_bps, "sell" if side == "LONG" else "buy")
        quantity = float(trade["quantity"])
        gross_pnl = (exit_price - float(trade["entry_price"])) * quantity if side == "LONG" else (float(trade["entry_price"]) - exit_price) * quantity
        exit_fee = exit_price * quantity * self.config.fee_bps / 10000.0
        total_fees = float(trade["entry_fee"]) + exit_fee
        net_pnl = gross_pnl - total_fees
        risk_dollars = max(float(trade["risk_dollars"]), 1e-9)
        r_multiple = net_pnl / risk_dollars
        equity = float(state.get("equity", self.config.initial_equity)) + net_pnl
        hours_held = (int(candle["close_time"]) - int(trade["open_close_time"])) / 3600000.0
        state["equity"] = equity
        state["daily_r"] = float(state.get("daily_r", 0.0)) + r_multiple
        state["open_trade"] = None
        state["range_adx_break_count"] = 0
        event = {
            **trade,
            "event": "EXIT",
            "timestamp": self.now.isoformat(),
            "status": "CLOSED",
            "closed_at": candle["date"].isoformat(),
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "exit_fee": exit_fee,
            "fees": total_fees,
            "slippage_bps": self.config.slippage_bps,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
            "holding_hours": hours_held,
        }
        self._append_rows(self.trade_dir / "trade_events.csv", [event])
        self._append_rows(self.trade_dir / "closed_trades.csv", [event])
        return event



class PairResearchAgent(CandidatePaperAgent):
    """Research-only pair monitor.

    This intentionally does not paper-trade. Pair P/L needs two-leg execution,
    hedge-ratio sizing, and separate fee accounting. Until that is wired, the
    agent records live BTC/ETH spread evidence and produces normal Strategy Lab
    report files with zero trades.
    """

    def run(self, assets: list[str] | None = None) -> AgentRunSummary:
        selected_assets = assets or list(self.config.assets.keys())
        if len(selected_assets) < 2:
            self._append_error("PAIR", "PAIR_ASSETS_MISSING", "Pair research requires at least two assets.")
            self.write_report()
            return AgentRunSummary(evaluated=0, entries=0, exits=0, alerts=["PAIR: missing assets"], missed_runs=0)

        frames: dict[str, pd.DataFrame] = {}
        alerts: list[str] = []
        heartbeat_rows: list[dict[str, Any]] = []
        missed_runs = 0

        for asset in selected_assets[:2]:
            symbol = self.config.assets.get(asset, asset)
            state = self._load_state(asset)
            try:
                candles = self.fetch_closed_candles(asset, symbol)
                if self._is_stale(candles):
                    self._append_error(asset, "STALE_FEED", f"Latest closed candle is stale for {symbol}")
                    alerts.append(f"{asset}: stale feed")
                    state["missed_run_count"] = int(state.get("missed_run_count", 0)) + 1
                else:
                    missed = self._missed_run_count(state, candles)
                    missed_runs += missed
                    state["last_evaluated_close_time"] = int(candles.iloc[-1]["close_time"])
                    state["missed_run_count"] = int(state.get("missed_run_count", 0)) + missed
                    frames[asset] = candles
                self._save_state(asset, state)
            except Exception as exc:  # noqa: BLE001 - runtime audit should catch all failures.
                self._append_error(asset, "ERROR", f"{exc.__class__.__name__}: {exc}")
                alerts.append(f"{asset}: ERROR {exc.__class__.__name__}")
            heartbeat_rows.append(self._heartbeat_row(asset, symbol, state))

        evaluated = 0
        if len(frames) == 2:
            evaluated = self._write_pair_evaluation(frames)
            alerts.append(f"{self.spec.candidate_id}: pair research evaluated={evaluated}")

        self._append_rows(self.heartbeat_dir / "heartbeats.csv", heartbeat_rows)
        self.write_report()
        return AgentRunSummary(evaluated=evaluated, entries=0, exits=0, alerts=alerts, missed_runs=missed_runs)

    def _write_pair_evaluation(self, frames: dict[str, pd.DataFrame]) -> int:
        left_asset, right_asset = list(frames.keys())[:2]
        left = frames[left_asset][["close_time", "date", "close"]].rename(columns={"close": "left_close"})
        right = frames[right_asset][["close_time", "close"]].rename(columns={"close": "right_close"})
        pair = pd.merge(left, right, on="close_time", how="inner").sort_values("close_time")
        pair = pair.dropna(subset=["left_close", "right_close"]).tail(self.config.candle_limit)
        if len(pair) < 60:
            self._append_error("PAIR", "PAIR_SAMPLE_TOO_SMALL", f"Only {len(pair)} aligned candles available.")
            return 0

        left_close = pair["left_close"].astype(float)
        right_close = pair["right_close"].astype(float)
        variance = float(right_close.var(ddof=0))
        hedge_beta = float(left_close.cov(right_close) / variance) if variance else 0.0
        spread = left_close - hedge_beta * right_close
        spread_mean = spread.rolling(60).mean()
        spread_std = spread.rolling(60).std(ddof=0)
        zscore = (spread - spread_mean) / spread_std.replace(0, pd.NA)
        latest = pair.iloc[-1]
        latest_z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0.0
        signal = "PAIR_STRETCHED" if abs(latest_z) >= 2.0 else "PAIR_NORMAL"
        row = {
            "timestamp": self.now.isoformat(),
            "candidate_id": self.spec.candidate_id,
            "asset": "BTC_ETH_PAIR",
            "symbol": f"{self.config.assets.get(left_asset, left_asset)}/{self.config.assets.get(right_asset, right_asset)}",
            "strategy_version": self.config.strategy_version,
            "candle_close_time": int(latest["close_time"]),
            "candle_time": pd.to_datetime(latest["date"], utc=True).isoformat(),
            "left_asset": left_asset,
            "right_asset": right_asset,
            "left_close": float(latest["left_close"]),
            "right_close": float(latest["right_close"]),
            "hedge_beta": hedge_beta,
            "spread": float(spread.iloc[-1]),
            "spread_zscore_60": latest_z,
            "passed": False,
            "rejection_reasons": "RESEARCH_ONLY_NO_PAIR_EXECUTION",
            "signal": signal,
        }
        self._append_rows(self.trade_dir / "evaluations.csv", [row])
        return int(len(pair))

    def write_report(self) -> None:
        evaluations = _safe_read_csv(self.trade_dir / "evaluations.csv")
        rows = [_candidate_report_row(self.spec, "BTC_ETH_PAIR", pd.DataFrame(), evaluations)]
        rows.append(_candidate_report_row(self.spec, "COMBINED", pd.DataFrame(), evaluations))
        report = pd.DataFrame(rows)
        report.to_csv(self.report_dir / f"strategy_lab_{self.spec.candidate_id}.csv", index=False)
        (self.report_dir / f"strategy_lab_{self.spec.candidate_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

def _range_config_from_spec(spec: CandidateSpec) -> RangeMRConfig:
    return RangeMRConfig(
        hypothesis_id=spec.candidate_id,
        entry_mode=spec.range_entry_mode,
        min_band_atr=spec.range_min_band_atr,
        require_adx_falling=spec.range_require_adx_falling,
        long_only=spec.range_long_only,
        atr_stop_multiple=spec.atr_stop_multiple if spec.atr_stop_multiple else 2.0,
    )


def _range_planned_reward_r(candle: pd.Series, side: str, slippage_bps: float, cfg: RangeMRConfig) -> float:
    if not side:
        return 0.0
    entry = apply_slippage(float(candle["close"]), slippage_bps, "buy" if side == "LONG" else "sell")
    stop = entry - cfg.atr_stop_multiple * float(candle["atr"]) if side == "LONG" else entry + cfg.atr_stop_multiple * float(candle["atr"])
    target = float(candle["ma20"])
    risk = abs(entry - stop)
    reward = target - entry if side == "LONG" else entry - target
    return max(0.0, reward / risk) if risk > 0 else 0.0


def run_strategy_lab(assets: dict[str, str] | None = None, now: datetime | None = None) -> StrategyLabRunSummary:
    now = now or datetime.now(timezone.utc)
    assets = assets or load_assets_from_env(default=STRATEGY_LAB_DEFAULT_ASSETS)
    evaluated = entries = exits = 0
    alerts: list[str] = []
    base_config = MeanReversionConfig(
        assets=assets,
        fee_bps=float(os.getenv("SLAB_FEE_BPS", os.getenv("MR_FEE_BPS", "10"))),
        slippage_bps=float(os.getenv("SLAB_SLIPPAGE_BPS", os.getenv("MR_SLIPPAGE_BPS", "2"))),
        risk_per_trade=float(os.getenv("SLAB_RISK_PER_TRADE", "0.01")),
        daily_loss_guard_r=float(os.getenv("SLAB_DAILY_LOSS_GUARD_R", "-3")),
        max_notional_pct=float(os.getenv("SLAB_MAX_NOTIONAL_PCT", "1")),
    )
    lab_dir = Path(os.getenv("STRATEGY_LAB_DATA_DIR", str(DEFAULT_LAB_DIR)))
    report_dir = Path(os.getenv("STRATEGY_LAB_REPORT_DIR", str(DEFAULT_REPORT_DIR)))
    selected = _selected_candidates()
    quarantine_enabled = os.getenv("STRATEGY_QUARANTINE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    quarantine_path = Path(os.getenv("STRATEGY_QUARANTINE_REPORT", str(DEFAULT_QUARANTINE_CSV)))
    quarantined = load_quarantined_strategy_ids(quarantine_path) if quarantine_enabled else set()
    quarantined |= MANUAL_BLOCKED_STRATEGIES
    for spec in selected:
        if not spec.enabled:
            continue
        if spec.candidate_id in quarantined:
            alerts.append(f"{spec.candidate_id}: QUARANTINED_BY_VERIFICATION; skipped new paper entries")
            continue
        candidate_assets = _candidate_assets(spec, assets)
        if not candidate_assets:
            continue
        config = dataclasses.replace(
            base_config,
            assets=candidate_assets,
            stale_after_minutes=_stale_after_minutes(candidate_assets, spec.interval),
            interval=spec.interval,
            strategy_version=f"{STRATEGY_LAB_VERSION}:{spec.candidate_id}",
            rsi_entry_below=spec.rsi_entry_below,
            atr_stop_multiple=spec.atr_stop_multiple,
        )
        if spec.family == StrategyFamily.RANGE_MEAN_REVERSION.value:
            agent_class = RangeMRPaperAgent
        elif spec.family == StrategyFamily.PAIRS_RESEARCH.value:
            agent_class = PairResearchAgent
        else:
            agent_class = CandidatePaperAgent
        agent = agent_class(spec=spec, config=config, data_dir=lab_dir / spec.candidate_id, report_dir=report_dir, now=now)
        summary = agent.run(list(candidate_assets.keys()))
        evaluated += summary.evaluated
        entries += summary.entries
        exits += summary.exits
        alerts.extend(summary.alerts)
    run_summary = StrategyLabRunSummary(evaluated=evaluated, entries=entries, exits=exits, alerts=alerts, candidate_count=len(selected))
    write_strategy_lab_summary(report_dir, selected)
    write_strategy_architecture_manifests(
        selected,
        assets=assets,
        run_summary=run_summary,
        report_dir=report_dir,
        lab_dir=lab_dir,
        now=now,
        workflow_name=os.getenv("GITHUB_WORKFLOW", "local-strategy-lab"),
    )
    return run_summary


def write_strategy_lab_summary(report_dir: Path = DEFAULT_REPORT_DIR, candidates: list[CandidateSpec] | None = None) -> pd.DataFrame:
    report_dir = Path(report_dir)
    candidates = candidates or list(CANDIDATES.values())
    rows: list[dict[str, Any]] = []
    for spec in candidates:
        path = report_dir / f"strategy_lab_{spec.candidate_id}.csv"
        frame = _safe_read_csv(path)
        if frame.empty:
            rows.append(_empty_summary_row(spec))
            continue
        combined = frame[frame["asset"].astype(str).str.upper() == "COMBINED"] if "asset" in frame else pd.DataFrame()
        row = combined.iloc[0].to_dict() if not combined.empty else frame.iloc[0].to_dict()
        rows.append(_summary_row(spec, row))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["rating_score", "total_trades"], ascending=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "strategy_lab_summary.csv", index=False)
    (report_dir / "strategy_lab_summary.json").write_text(json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8")
    return summary


def _candidate_report_row(spec: CandidateSpec, asset: str, trades: pd.DataFrame, evaluations: pd.DataFrame) -> dict[str, Any]:
    row = report_row(asset, trades, evaluations)
    row["candidate_id"] = spec.candidate_id
    row["display_label"] = spec.display_label or spec.candidate_id
    row["bucket"] = spec.bucket
    row["family"] = spec.family
    row["title"] = spec.title
    row["interval"] = spec.interval
    row["asset_filter"] = ",".join(spec.asset_filter) if spec.asset_filter else "ALL"
    row["asset_exclude"] = ",".join(spec.asset_exclude) if spec.asset_exclude else "AUTO_QUARANTINE"
    row["evidence_note"] = spec.evidence_note
    row["enabled"] = spec.enabled
    row["rating"] = _rating(row)
    row["rating_score"] = _rating_score(row)
    return row


def _rating_score(row: dict[str, Any]) -> float:
    trades = int(row.get("total_trades", 0) or 0)
    expectancy = float(row.get("expectancy_r", 0.0) or 0.0)
    pf = float(row.get("profit_factor", 0.0) or 0.0)
    drawdown = abs(float(row.get("max_drawdown", 0.0) or 0.0))
    score = 40 + min(20, trades) + max(-20, min(25, expectancy * 15)) + max(-10, min(20, (pf - 1) * 10 if pf else -10)) - min(20, drawdown * 100)
    if trades < 20:
        score -= 10
    return round(max(0.0, min(100.0, score)), 2)


def _rating(row: dict[str, Any]) -> str:
    trades = int(row.get("total_trades", 0) or 0)
    score = _rating_score(row)
    if trades < 20:
        return "INSUFFICIENT_SAMPLE"
    if score >= 75:
        return "PROMOTE_CANDIDATE"
    if score >= 60:
        return "KEEP_TESTING"
    if score >= 45:
        return "WATCHLIST"
    return "REJECT_OR_REWORK"


def _summary_row(spec: CandidateSpec, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": spec.candidate_id,
        "display_label": spec.display_label or spec.candidate_id,
        "bucket": spec.bucket,
        "family": spec.family,
        "title": spec.title,
        "interval": spec.interval,
        "asset_filter": ",".join(spec.asset_filter) if spec.asset_filter else "ALL",
        "asset_exclude": ",".join(spec.asset_exclude) if spec.asset_exclude else "AUTO_QUARANTINE",
        "evidence_note": spec.evidence_note,
        "enabled": spec.enabled,
        "total_trades": int(float(row.get("total_trades", 0) or 0)),
        "win_rate": float(row.get("win_rate", 0.0) or 0.0),
        "expectancy_r": float(row.get("expectancy_r", 0.0) or 0.0),
        "profit_factor": float(row.get("profit_factor", 0.0) or 0.0),
        "max_drawdown": float(row.get("max_drawdown", 0.0) or 0.0),
        "net_pnl": float(row.get("net_pnl", 0.0) or 0.0),
        "rating_score": float(row.get("rating_score", 0.0) or 0.0),
        "rating": str(row.get("rating", "INSUFFICIENT_SAMPLE")),
        "insufficient_sample": _bool_value(row.get("insufficient_sample", True)),
    }


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _empty_summary_row(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": spec.candidate_id,
        "display_label": spec.display_label or spec.candidate_id,
        "bucket": spec.bucket,
        "family": spec.family,
        "title": spec.title,
        "interval": spec.interval,
        "asset_filter": ",".join(spec.asset_filter) if spec.asset_filter else "ALL",
        "asset_exclude": ",".join(spec.asset_exclude) if spec.asset_exclude else "AUTO_QUARANTINE",
        "evidence_note": spec.evidence_note,
        "enabled": spec.enabled,
        "total_trades": 0,
        "win_rate": 0.0,
        "expectancy_r": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "net_pnl": 0.0,
        "rating_score": 0.0,
        "rating": "INSUFFICIENT_SAMPLE",
        "insufficient_sample": True,
    }



def _stale_after_minutes(assets: dict[str, str], interval: str = "1h") -> int:
    asset_names = {asset.upper() for asset in assets}
    if asset_names and asset_names.issubset(MARKET_HOURS_ASSETS):
        return int(os.getenv("SLAB_MARKET_HOURS_STALE_AFTER_MINUTES", "4320"))
    if asset_names & MARKET_HOURS_ASSETS:
        return int(os.getenv("SLAB_MIXED_MARKET_STALE_AFTER_MINUTES", "4320"))
    interval_minutes = _interval_minutes(interval)
    default_minutes = max(180, int(interval_minutes * 1.5))
    return int(os.getenv("SLAB_STALE_AFTER_MINUTES", str(default_minutes)))


def _interval_minutes(interval: str) -> int:
    value = str(interval).strip().lower()
    if value.endswith("m"):
        return int(value[:-1] or "1")
    if value.endswith("h"):
        return int(value[:-1] or "1") * 60
    if value.endswith("d"):
        return int(value[:-1] or "1") * 1440
    if value.endswith("w"):
        return int(value[:-1] or "1") * 10080
    return 60


def _candidate_assets(spec: CandidateSpec, assets: dict[str, str]) -> dict[str, str]:
    if spec.asset_filter:
        allowed = {asset.upper() for asset in spec.asset_filter}
        selected = {asset: symbol for asset, symbol in assets.items() if asset.upper() in allowed}
    else:
        selected = dict(assets)
    excluded = set(_quarantined_assets())
    excluded.update(asset.upper() for asset in spec.asset_exclude)
    if spec.asset_filter:
        explicit = {asset.upper() for asset in spec.asset_filter}
        excluded -= explicit
    return {asset: symbol for asset, symbol in selected.items() if asset.upper() not in excluded}


def _quarantined_assets() -> set[str]:
    raw = os.getenv("STRATEGY_LAB_QUARANTINED_ASSETS", "").strip()
    if raw:
        return {item.strip().upper() for item in raw.split(",") if item.strip()}
    return set(DEFAULT_QUARANTINED_ASSETS)


def _selected_candidates() -> list[CandidateSpec]:
    raw = os.getenv("STRATEGY_LAB_CANDIDATES", "").strip()
    if not raw:
        return list(CANDIDATES.values())
    selected = []
    for item in raw.split(","):
        spec = CANDIDATES.get(item.strip().upper())
        if spec:
            selected.append(spec)
    return selected or list(CANDIDATES.values())


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _csv_asset_values(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "asset" not in frame:
        return set()
    values: set[str] = set()
    for value in frame["asset"].dropna().unique():
        text = str(value).strip()
        if text:
            values.add(text)
    return values




