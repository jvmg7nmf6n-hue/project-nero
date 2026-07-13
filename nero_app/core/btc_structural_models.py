"""
BTC Structural Models
=====================

Supply-side context for Bitcoin research. These models are explanatory
filters, not price predictions: stock-to-flow and miner-cost floors are
contested and must never be treated as deterministic trading signals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import os
from typing import Dict, List, Optional

import numpy as np

HALVING_INTERVAL_BLOCKS = 210_000
GENESIS_UTC = datetime(2009, 1, 3, 18, 15, tzinfo=timezone.utc)
BLOCKS_PER_DAY = 144.0
DAYS_PER_YEAR = 365.25
MAX_SUPPLY_BTC = 21_000_000.0
INITIAL_SUBSIDY_BTC = 50.0


@dataclass
class BtcStructuralReport:
    structural_score: float
    structural_label: str
    block_height: int
    halving_era: int
    block_subsidy_btc: float
    estimated_stock_btc: float
    annual_issuance_btc: float
    stock_to_flow: Optional[float]
    next_halving_height: int
    blocks_to_halving: int
    miner_cost_floor: Optional[float] = None
    miner_floor_ratio: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def rows(self) -> List[Dict[str, object]]:
        return [
            {"Signal": "Block Height", "Reading": self.block_height, "Meaning": "Estimated or configured BTC chain height."},
            {"Signal": "Halving Era", "Reading": self.halving_era, "Meaning": "Number of completed reward halvings."},
            {"Signal": "Block Subsidy", "Reading": f"{self.block_subsidy_btc:.8f} BTC", "Meaning": "New BTC created per block before fees."},
            {"Signal": "Annual Issuance", "Reading": f"{self.annual_issuance_btc:,.0f} BTC", "Meaning": "Approximate new BTC supply per year at current subsidy."},
            {"Signal": "Stock-to-Flow", "Reading": "n/a" if self.stock_to_flow is None else f"{self.stock_to_flow:.1f}", "Meaning": "Existing stock divided by annual new issuance; contested scarcity metric."},
            {"Signal": "Blocks to Halving", "Reading": self.blocks_to_halving, "Meaning": "Lower value means supply-cut narrative is closer."},
            {"Signal": "Miner Floor Ratio", "Reading": "n/a" if self.miner_floor_ratio is None else f"{self.miner_floor_ratio:.2f}x", "Meaning": "BTC price divided by optional miner cost floor; below 1.0 suggests miner stress."},
        ]


def estimate_block_height(now: Optional[datetime] = None) -> int:
    """Estimate BTC block height from genesis at a 10-minute cadence."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elapsed_minutes = max(0.0, (now - GENESIS_UTC).total_seconds() / 60.0)
    return int(elapsed_minutes // 10.0)


def block_subsidy_for_height(block_height: int) -> float:
    era = max(0, int(block_height) // HALVING_INTERVAL_BLOCKS)
    if era >= 33:
        return 0.0
    return INITIAL_SUBSIDY_BTC / (2 ** era)


def issued_supply_for_height(block_height: int) -> float:
    height = max(0, int(block_height))
    era = height // HALVING_INTERVAL_BLOCKS
    issued = 0.0
    for completed_era in range(min(era, 33)):
        issued += HALVING_INTERVAL_BLOCKS * (INITIAL_SUBSIDY_BTC / (2 ** completed_era))
    if era < 33:
        issued += (height % HALVING_INTERVAL_BLOCKS) * block_subsidy_for_height(height)
    return float(min(MAX_SUPPLY_BTC, issued))


def build_btc_structural_report(
    current_price: Optional[float] = None,
    block_height: Optional[int] = None,
    miner_cost_floor: Optional[float] = None,
) -> BtcStructuralReport:
    """Build a BTC supply-structure report.

    `block_height` can be configured through `BTC_BLOCK_HEIGHT`. If absent,
    NERO estimates it from calendar time so the dashboard remains usable.
    `miner_cost_floor` can be supplied directly or through `BTC_MINER_COST_FLOOR`.
    """
    notes: List[str] = []
    if block_height is None:
        env_height = os.getenv("BTC_BLOCK_HEIGHT", "").strip()
        if env_height:
            try:
                block_height = int(env_height)
                notes.append("BTC block height loaded from BTC_BLOCK_HEIGHT.")
            except ValueError:
                block_height = None
                notes.append("BTC_BLOCK_HEIGHT could not be parsed; using calendar estimate.")
    if block_height is None:
        block_height = estimate_block_height()
        notes.append("BTC block height estimated from genesis and 10-minute cadence; configure BTC_BLOCK_HEIGHT for precision.")

    if miner_cost_floor is None:
        env_floor = os.getenv("BTC_MINER_COST_FLOOR", "").strip()
        if env_floor:
            try:
                miner_cost_floor = float(env_floor.replace(",", ""))
                notes.append("Miner cost floor loaded from BTC_MINER_COST_FLOOR.")
            except ValueError:
                notes.append("BTC_MINER_COST_FLOOR could not be parsed; miner floor omitted.")

    subsidy = block_subsidy_for_height(block_height)
    annual_issuance = subsidy * BLOCKS_PER_DAY * DAYS_PER_YEAR
    stock = issued_supply_for_height(block_height)
    stock_to_flow = None if annual_issuance <= 0 else stock / annual_issuance
    era = block_height // HALVING_INTERVAL_BLOCKS
    next_halving = (era + 1) * HALVING_INTERVAL_BLOCKS
    blocks_to_halving = max(0, next_halving - block_height)

    miner_floor_ratio = None
    if current_price is not None and miner_cost_floor and miner_cost_floor > 0:
        miner_floor_ratio = float(current_price) / float(miner_cost_floor)

    score = 50.0
    if stock_to_flow is not None:
        if stock_to_flow >= 100:
            score += 25
            notes.append("Stock-to-flow is high; structural scarcity lens is supportive, but contested.")
        elif stock_to_flow >= 50:
            score += 15
            notes.append("Stock-to-flow is moderately supportive by scarcity lens.")
        else:
            score += 5
            notes.append("Stock-to-flow is below mature-scarcity regimes.")
    if blocks_to_halving <= 52_560:
        score += 10
        notes.append("Next halving is within roughly one year; supply-cut narrative can strengthen.")
    elif blocks_to_halving >= 160_000:
        score -= 5
        notes.append("Halving is far away; supply-cut narrative is less immediate.")
    if miner_floor_ratio is not None:
        if miner_floor_ratio < 1.0:
            score -= 20
            notes.append("BTC is below supplied miner cost floor; miner stress risk is elevated.")
        elif miner_floor_ratio < 1.2:
            score -= 8
            notes.append("BTC is close to supplied miner cost floor; miner stress risk deserves caution.")
        elif miner_floor_ratio > 2.0:
            score += 5
            notes.append("BTC trades well above supplied miner cost floor; miner stress is lower.")

    score = float(np.clip(score, 0, 100))
    if score >= 70:
        label = "STRUCTURAL_SUPPORTIVE"
    elif score >= 45:
        label = "STRUCTURAL_NEUTRAL"
    else:
        label = "STRUCTURAL_PRESSURE"
    notes.append("Stock-to-flow and miner-floor models are explanatory context, not deterministic price laws.")

    return BtcStructuralReport(
        structural_score=round(score, 2),
        structural_label=label,
        block_height=int(block_height),
        halving_era=int(era),
        block_subsidy_btc=round(subsidy, 8),
        estimated_stock_btc=round(stock, 2),
        annual_issuance_btc=round(annual_issuance, 2),
        stock_to_flow=None if stock_to_flow is None or math.isinf(stock_to_flow) else round(stock_to_flow, 2),
        next_halving_height=int(next_halving),
        blocks_to_halving=int(blocks_to_halving),
        miner_cost_floor=miner_cost_floor,
        miner_floor_ratio=None if miner_floor_ratio is None else round(miner_floor_ratio, 3),
        notes=notes,
    )
