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
    MeanReversionAgent,
    MeanReversionConfig,
    add_indicators,
    apply_slippage,
    load_assets_from_env,
    report_row,
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


class TargetMode(str, Enum):
    FROZEN_MA20 = "FROZEN_MA20"
    FIXED_1R = "FIXED_1R"
    FIXED_125R = "FIXED_125R"
    FIXED_150R = "FIXED_150R"


class StrategyFamily(str, Enum):
    MEAN_REVERSION = "Mean Reversion"
    MOMENTUM = "Momentum"
    EXIT_LOGIC = "Exit Logic"
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
    "NEW_BTC_ETH_12H_PAIR": CandidateSpec(
        candidate_id="NEW_BTC_ETH_12H_PAIR",
        family="Pairs Research",
        title="BTC-ETH 12h cointegration pair",
        display_label="NEW_BTC_ETH_12H_PAIR",
        bucket="RESEARCH_ONLY",
        asset_filter=("BTC", "ETH"),
        interval="12h",
        evidence_note="Claude sweep: BTC-ETH/12h pair positive but weak. Kept research-only until real pair execution is wired.",
        enabled=False,
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
        if self.spec.require_trend_support and ma20 <= ma200:
            reasons.append("TREND_SUPPORT_NOT_CONFIRMED")
        if self.spec.min_planned_reward_r and planned_reward_r < self.spec.min_planned_reward_r:
            reasons.append("PLANNED_REWARD_TOO_LOW")

        if self._is_momentum():
            reasons.extend(self._validate_momentum(candle))
        elif self._is_mean_reversion_like():
            reasons.extend(self._validate_mean_reversion(candle))
        else:
            reasons.append(f"UNKNOWN_FAMILY:{self.spec.family}")

        return reasons, planned_reward_r

    def planned_reward_r(self, candle: pd.Series) -> float:
        entry_price = apply_slippage(float(candle["close"]), self.slippage_bps, "buy")
        risk_per_unit = self.risk_per_unit(candle)
        if risk_per_unit <= 0:
            return 0.0
        target = self.target_price(candle, entry_price, risk_per_unit)
        return max(0.0, (target - entry_price) / risk_per_unit)

    def risk_per_unit(self, candle: pd.Series) -> float:
        entry_price = apply_slippage(float(candle["close"]), self.slippage_bps, "buy")
        stop_loss = entry_price - self.spec.atr_stop_multiple * float(candle["atr"])
        return entry_price - stop_loss

    def target_price(self, candle: pd.Series, entry_price: float, risk_per_unit: float) -> float:
        target_mode = _target_mode_value(self.spec.target_mode)
        if target_mode == TargetMode.FIXED_1R.value:
            return entry_price + risk_per_unit
        if target_mode == TargetMode.FIXED_125R.value:
            return entry_price + 1.25 * risk_per_unit
        if target_mode == TargetMode.FIXED_150R.value:
            return entry_price + 1.5 * risk_per_unit
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

    def _is_momentum(self) -> bool:
        return self.spec.family == StrategyFamily.MOMENTUM.value

    def _is_mean_reversion_like(self) -> bool:
        return self.spec.family in {StrategyFamily.MEAN_REVERSION.value, StrategyFamily.EXIT_LOGIC.value}


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
        raw_entry = float(candle["close"])
        entry_price = apply_slippage(raw_entry, self.config.slippage_bps, "buy")
        stop_loss = entry_price - self.spec.atr_stop_multiple * float(candle["atr"])
        risk_per_unit = entry_price - stop_loss
        if risk_per_unit <= 0:
            return None
        target_mode = _target_mode_value(self.spec.target_mode)
        target = SignalValidator(self.spec, slippage_bps=self.config.slippage_bps).target_price(candle, entry_price, risk_per_unit)
        reward_per_unit = target - entry_price
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
            "entry_atr": float(candle["atr"]),
        }
        state["open_trade"] = trade
        self._append_rows(self.trade_dir / "trade_events.csv", [{**trade, "event": "ENTRY", "timestamp": self.now.isoformat()}])
        return trade

    def write_report(self) -> None:
        trades_path = self.trade_dir / "closed_trades.csv"
        evaluations_path = self.trade_dir / "evaluations.csv"
        trades = _safe_read_csv(trades_path)
        evaluations = _safe_read_csv(evaluations_path)
        rows = []
        assets = sorted(set(self.config.assets.keys()) | set(trades["asset"].unique() if not trades.empty and "asset" in trades else []))
        for asset in assets:
            rows.append(_candidate_report_row(self.spec, asset, trades[trades["asset"] == asset] if not trades.empty and "asset" in trades else pd.DataFrame(), evaluations))
        rows.append(_candidate_report_row(self.spec, "COMBINED", trades, evaluations))
        report = pd.DataFrame(rows)
        report.to_csv(self.report_dir / f"strategy_lab_{self.spec.candidate_id}.csv", index=False)
        (self.report_dir / f"strategy_lab_{self.spec.candidate_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


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
    for spec in selected:
        if not spec.enabled:
            continue
        candidate_assets = _candidate_assets(spec, assets)
        if not candidate_assets:
            continue
        config = dataclasses.replace(
            base_config,
            assets=candidate_assets,
            stale_after_minutes=_stale_after_minutes(candidate_assets),
            interval=spec.interval,
            strategy_version=f"{STRATEGY_LAB_VERSION}:{spec.candidate_id}",
            rsi_entry_below=spec.rsi_entry_below,
            atr_stop_multiple=spec.atr_stop_multiple,
        )
        agent = CandidatePaperAgent(spec=spec, config=config, data_dir=lab_dir / spec.candidate_id, report_dir=report_dir, now=now)
        summary = agent.run(list(candidate_assets.keys()))
        evaluated += summary.evaluated
        entries += summary.entries
        exits += summary.exits
        alerts.extend(summary.alerts)
    write_strategy_lab_summary(report_dir, selected)
    return StrategyLabRunSummary(evaluated=evaluated, entries=entries, exits=exits, alerts=alerts, candidate_count=len(selected))


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



def _stale_after_minutes(assets: dict[str, str]) -> int:
    asset_names = {asset.upper() for asset in assets}
    if asset_names and asset_names.issubset(MARKET_HOURS_ASSETS):
        return int(os.getenv("SLAB_MARKET_HOURS_STALE_AFTER_MINUTES", "4320"))
    if asset_names & MARKET_HOURS_ASSETS:
        return int(os.getenv("SLAB_MIXED_MARKET_STALE_AFTER_MINUTES", "4320"))
    return int(os.getenv("SLAB_STALE_AFTER_MINUTES", "180"))


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


