"""Self-evolution analysis for NERO strategy research.

This module does not change live or paper-trading rules. It reads closed
paper trades and strategy reports, explains losses, and proposes versioned
shadow-test variants for human/audited promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_LAB_DIR = PROJECT_ROOT / "nero_app" / "data" / "strategy_lab"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class EvolutionReport:
    maturity_score: float
    label: str
    total_trades: int
    total_losses: int
    promote_ready: int
    notes: list[str]
    autopsy_rows: list[dict[str, Any]]
    recommendation_rows: list[dict[str, Any]]
    variant_rows: list[dict[str, Any]]
    asset_action_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "maturity_score": self.maturity_score,
            "label": self.label,
            "total_trades": self.total_trades,
            "total_losses": self.total_losses,
            "promote_ready": self.promote_ready,
            "notes": self.notes,
            "autopsy_rows": self.autopsy_rows,
            "recommendation_rows": self.recommendation_rows,
            "variant_rows": self.variant_rows,
            "asset_action_rows": self.asset_action_rows,
        }


def build_strategy_evolution_report(
    lab_dir: Path = DEFAULT_STRATEGY_LAB_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    min_trades_for_promotion: int = 100,
) -> EvolutionReport:
    lab_dir = Path(lab_dir)
    report_dir = Path(report_dir)
    summary = _safe_read_csv(report_dir / "strategy_lab_summary.csv")
    closed = _load_closed_strategy_trades(lab_dir)
    if summary.empty and closed.empty:
        return EvolutionReport(
            maturity_score=0.0,
            label="NO_EVIDENCE_YET",
            total_trades=0,
            total_losses=0,
            promote_ready=0,
            notes=["No strategy lab evidence found yet. Run the Strategy TEST Lab first."],
            autopsy_rows=[],
            recommendation_rows=[],
            variant_rows=[],
            asset_action_rows=[],
        )

    autopsy_rows = _build_autopsy_rows(summary, closed)
    recommendation_rows = [_recommend_for_row(row, min_trades_for_promotion) for row in _summary_records(summary)]
    variant_rows = [_variant_for_recommendation(row) for row in recommendation_rows if row["Action"] != "PROMOTE_READY"]
    asset_action_rows = _build_asset_action_rows(closed, lab_dir)
    total_trades = _sum_numeric(summary, "total_trades") if not summary.empty else len(closed)
    total_losses = int((pd.to_numeric(closed.get("r_multiple", pd.Series(dtype=float)), errors="coerce") < 0).sum()) if not closed.empty else 0
    promote_ready = sum(1 for row in recommendation_rows if row["Action"] == "PROMOTE_READY")
    maturity_score = _maturity_score(summary, total_trades, promote_ready, min_trades_for_promotion)
    label = _maturity_label(maturity_score, promote_ready)
    notes = _report_notes(total_trades, total_losses, promote_ready, min_trades_for_promotion)
    return EvolutionReport(
        maturity_score=maturity_score,
        label=label,
        total_trades=total_trades,
        total_losses=total_losses,
        promote_ready=promote_ready,
        notes=notes,
        autopsy_rows=autopsy_rows,
        recommendation_rows=recommendation_rows,
        variant_rows=variant_rows,
        asset_action_rows=asset_action_rows,
    )


def write_strategy_evolution_report(report: EvolutionReport, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.autopsy_rows).to_csv(report_dir / "strategy_evolution_autopsy.csv", index=False)
    pd.DataFrame(report.recommendation_rows).to_csv(report_dir / "strategy_evolution_recommendations.csv", index=False)
    pd.DataFrame(report.variant_rows).to_csv(report_dir / "strategy_evolution_variants.csv", index=False)
    pd.DataFrame(report.asset_action_rows).to_csv(report_dir / "strategy_evolution_asset_actions.csv", index=False)
    (report_dir / "strategy_evolution_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _build_autopsy_rows(summary: pd.DataFrame, closed: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _summary_records(summary):
        candidate_id = str(record.get("candidate_id", "UNKNOWN"))
        trades = closed[closed["candidate_id"].astype(str) == candidate_id] if not closed.empty and "candidate_id" in closed else pd.DataFrame()
        losses = trades[pd.to_numeric(trades.get("r_multiple", pd.Series(dtype=float)), errors="coerce") < 0] if not trades.empty else pd.DataFrame()
        top_exit = _top_value(losses, "exit_reason")
        avg_loss_r = _mean_numeric(losses, "r_multiple")
        avg_entry_rsi = _mean_numeric(losses, "entry_rsi")
        avg_planned_reward = _mean_numeric(losses, "planned_reward_r")
        rows.append(
            {
                "Candidate": candidate_id,
                "Trades": int(float(record.get("total_trades", 0) or 0)),
                "Losses": len(losses),
                "Top Loss Exit": top_exit or "none",
                "Avg Loss R": round(avg_loss_r, 3),
                "Avg Losing Entry RSI": round(avg_entry_rsi, 2),
                "Avg Planned Reward R": round(avg_planned_reward, 2),
                "Likely Mistake": _loss_reason(record, losses, top_exit, avg_planned_reward),
            }
        )
    return rows


def _recommend_for_row(row: dict[str, Any], min_trades_for_promotion: int) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", "UNKNOWN"))
    family = str(row.get("family", "Unknown"))
    trades = int(float(row.get("total_trades", 0) or 0))
    win_rate = float(row.get("win_rate", 0.0) or 0.0)
    expectancy = float(row.get("expectancy_r", 0.0) or 0.0)
    profit_factor = float(row.get("profit_factor", 0.0) or 0.0)
    drawdown = abs(float(row.get("max_drawdown", 0.0) or 0.0))
    if trades >= min_trades_for_promotion and expectancy > 0 and profit_factor >= 1.25 and drawdown <= 0.08:
        action = "PROMOTE_READY"
        recommendation = "Candidate has enough evidence for controlled promotion review."
    elif trades < 30:
        action = "COLLECT_MORE_DATA"
        recommendation = "Keep in shadow testing; sample is too small for strategy judgement."
    elif expectancy <= 0 or profit_factor < 1.0:
        action = "REWORK"
        recommendation = _rework_recommendation(candidate_id, family)
    else:
        action = "KEEP_TESTING"
        recommendation = "Edge is not rejected; keep testing until 100 trades and monitor drawdown."
    return {
        "Candidate": candidate_id,
        "Family": family,
        "Trades": trades,
        "Win Rate": round(win_rate, 3),
        "Expectancy R": round(expectancy, 3),
        "Profit Factor": round(profit_factor, 3),
        "Max Drawdown": round(drawdown, 3),
        "Action": action,
        "Recommendation": recommendation,
    }


def _variant_for_recommendation(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row["Candidate"])
    family = str(row["Family"])
    if family == "Momentum":
        changes = "Add retest confirmation; require 60D trend support; block high-volatility shock regimes; target at 1.5R."
        hypothesis = "Breakout losses may shrink if NERO avoids first-break fakeouts and only accepts cleaner trend regimes."
    elif family == "Exit Logic":
        changes = "Compare fixed 1R, 1.25R, and frozen MA20 exits; reject setups with planned reward below 1.2R."
        hypothesis = "Exit quality may improve if reward-to-risk is validated before entry."
    else:
        changes = "Tighten regime filter; require RSI recovery hook; avoid entries when MA20 target is too close after fees."
        hypothesis = "Mean-reversion losses may reduce if entries wait for oversold exhaustion instead of catching falling candles."
    return {
        "Parent": candidate_id,
        "Proposed Variant": candidate_id.replace("_V1", "_V2") if "_V1" in candidate_id else f"{candidate_id}_V2",
        "Mode": "SHADOW_TEST_ONLY",
        "Hypothesis": hypothesis,
        "Proposed Changes": changes,
        "Promotion Rule": "100 trades, expectancy > 0, profit factor >= 1.25, max drawdown <= 8%, live-data only.",
    }



def _build_asset_action_rows(closed: pd.DataFrame, lab_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closed_rows = _asset_rows_from_closed_trades(closed)
    rows.extend(closed_rows)
    rows.extend(_asset_rows_from_runtime_errors(lab_dir, {row["Asset"] for row in closed_rows}))
    rows.sort(key=lambda row: (str(row["Action"]), str(row["Asset"])))
    return rows


def _asset_rows_from_closed_trades(closed: pd.DataFrame) -> list[dict[str, Any]]:
    if closed.empty or "asset" not in closed:
        return []
    rows: list[dict[str, Any]] = []
    for asset, group in closed.groupby(closed["asset"].astype(str)):
        r_series = pd.to_numeric(group.get("r_multiple", pd.Series(dtype=float)), errors="coerce").dropna()
        pnl_series = pd.to_numeric(group.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        total = int(len(group))
        wins = int((r_series > 0).sum())
        losses = int((r_series < 0).sum())
        expectancy = float(r_series.mean()) if not r_series.empty else 0.0
        net_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0
        profit_factor = _profit_factor(r_series)
        action, next_hypothesis = _asset_action(asset, total, wins, losses, expectancy, profit_factor, net_pnl)
        rows.append(
            {
                "Asset": asset,
                "Class": _asset_class(asset),
                "Trades": total,
                "Wins": wins,
                "Losses": losses,
                "Win Rate": round(wins / total, 3) if total else 0.0,
                "Expectancy R": round(expectancy, 3),
                "Profit Factor": round(profit_factor, 3),
                "Net PnL": round(net_pnl, 2),
                "Action": action,
                "Next Hypothesis": next_hypothesis,
            }
        )
    return rows


def _asset_rows_from_runtime_errors(lab_dir: Path, assets_seen: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocked: dict[str, str] = {}
    for path in Path(lab_dir).glob("*/trades/runtime_errors.csv"):
        frame = _safe_read_csv(path)
        if frame.empty or "asset" not in frame:
            continue
        for record in frame.to_dict("records"):
            asset = str(record.get("asset", "") or "").strip()
            if not asset or asset in assets_seen:
                continue
            error_text = " ".join(str(value) for value in record.values()).lower()
            if any(token in error_text for token in ["twelve", "api", "httperror", "delisted", "stale"]):
                blocked[asset] = str(record.get("error", record.get("message", "feed failure")) or "feed failure")
    for asset, reason in sorted(blocked.items()):
        rows.append(
            {
                "Asset": asset,
                "Class": _asset_class(asset),
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Win Rate": 0.0,
                "Expectancy R": 0.0,
                "Profit Factor": 0.0,
                "Net PnL": 0.0,
                "Action": "DATA_BLOCKED",
                "Next Hypothesis": f"Do not trust signals until feed quality is fixed: {reason}",
            }
        )
    return rows


def _asset_action(asset: str, total: int, wins: int, losses: int, expectancy: float, profit_factor: float, net_pnl: float) -> tuple[str, str]:
    asset_class = _asset_class(asset)
    if total < 5:
        return "COLLECT_MORE_DATA", "Sample is too small; keep shadow testing before changing rules."
    if expectancy > 0 and profit_factor >= 1.25 and net_pnl > 0:
        return "PROMISING_WATCH", f"Build a {asset_class.lower()}-specific variant and keep collecting until 100 trades."
    if wins == 0 or expectancy <= -0.5 or profit_factor < 0.75:
        return "QUARANTINE", _quarantine_hypothesis(asset_class)
    if expectancy <= 0 or net_pnl < 0:
        return "REWORK_ASSET_RULES", _rework_asset_hypothesis(asset_class)
    return "KEEP_TESTING", "Edge is not proven, but not rejected; continue live-data shadow testing."


def _quarantine_hypothesis(asset_class: str) -> str:
    if asset_class == "FX":
        return "Pause this asset class; current crypto-style entries do not fit FX volatility/fee behavior."
    if asset_class == "CRYPTO":
        return "Reduce risk and require stronger regime/volatility confirmation before any new entry."
    if asset_class in {"ENERGY_FUTURES", "METALS_FUTURES"}:
        return "Retest with commodity-specific session, gap, and volatility filters before allowing entries."
    if asset_class == "EQUITY":
        return "Require market-index alignment and earnings/event filters before paper entries."
    return "Pause new entries until the loss pattern is explained with a larger clean sample."


def _rework_asset_hypothesis(asset_class: str) -> str:
    if asset_class == "FX":
        return "Use smaller targets, tighter stale-feed checks, and session-aware entry filters."
    if asset_class == "CRYPTO":
        return "Add trend-strength and liquidity filters; avoid repeating entries into high-volatility drawdowns."
    if asset_class in {"ENERGY_FUTURES", "METALS_FUTURES"}:
        return "Separate commodity logic from crypto logic; validate ATR target distance after fees/slippage."
    return "Create an asset-specific shadow variant instead of applying one generic rule set."


def _asset_class(asset: str) -> str:
    symbol = asset.upper()
    if symbol in {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "DXY"}:
        return "FX"
    if symbol in {"GOLD", "SILVER", "OIL"}:
        return "SPOT_COMMODITY"
    if symbol in {"GOLD_FUT", "SILVER_FUT", "COPPER_FUT"}:
        return "METALS_FUTURES"
    if symbol in {"OIL_FUT", "BRENT_FUT"}:
        return "ENERGY_FUTURES"
    if symbol in {"SPY", "QQQ", "NVDA", "MSTR", "COIN", "MARA", "RIOT", "GLD", "GDX", "NEM"}:
        return "EQUITY"
    return "CRYPTO"


def _profit_factor(r_series: pd.Series) -> float:
    if r_series.empty:
        return 0.0
    gross_win = float(r_series[r_series > 0].sum())
    gross_loss = abs(float(r_series[r_series < 0].sum()))
    if gross_loss == 0:
        return gross_win if gross_win > 0 else 0.0
    return gross_win / gross_loss

def _loss_reason(record: dict[str, Any], losses: pd.DataFrame, top_exit: str, avg_planned_reward: float) -> str:
    expectancy = float(record.get("expectancy_r", 0.0) or 0.0)
    profit_factor = float(record.get("profit_factor", 0.0) or 0.0)
    if losses.empty:
        return "No losses recorded yet; keep collecting evidence."
    if top_exit == "SL" and avg_planned_reward < 1.2:
        return "Stop-loss hits with weak planned reward; entry needs stricter reward-to-risk gate."
    if top_exit == "SL":
        return "Stop-loss cluster; likely regime/volatility filter problem."
    if expectancy <= 0 and profit_factor < 1:
        return "Negative edge; wins are not paying enough for losses."
    return "Loss pattern needs more sample before diagnosis."


def _rework_recommendation(candidate_id: str, family: str) -> str:
    if family == "Momentum":
        return "Rework breakout entry: require retest/close confirmation and volatility shock filter before paper entry."
    if family == "Exit Logic":
        return "Rework exits: compare target logic and block trades with planned reward below realistic fees/slippage threshold."
    if "DEEP_VALUE" in candidate_id:
        return "Deep value is too early; add recovery candle confirmation before entry."
    return "Rework mean-reversion entry: require oversold recovery hook and stronger trend/regime confirmation."


def _summary_records(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    return summary.to_dict("records")


def _load_closed_strategy_trades(lab_dir: Path) -> pd.DataFrame:
    frames = []
    for path in Path(lab_dir).glob("*/trades/closed_trades.csv"):
        frame = _safe_read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _top_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    counts = frame[column].astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.mean())


def _sum_numeric(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _maturity_score(summary: pd.DataFrame, total_trades: int, promote_ready: int, min_trades_for_promotion: int) -> float:
    if summary.empty:
        return 0.0
    best_pf = float(pd.to_numeric(summary.get("profit_factor", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
    best_expectancy = float(pd.to_numeric(summary.get("expectancy_r", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
    sample_score = min(40.0, (total_trades / max(1, min_trades_for_promotion * max(1, len(summary)))) * 40.0)
    edge_score = max(0.0, min(35.0, best_expectancy * 25.0 + (best_pf - 1.0) * 15.0))
    promote_score = min(25.0, promote_ready * 25.0)
    return round(max(0.0, min(100.0, sample_score + edge_score + promote_score)), 2)


def _maturity_label(score: float, promote_ready: int) -> str:
    if promote_ready:
        return "PROMOTION_REVIEW_READY"
    if score >= 65:
        return "LEARNING_WITH_EVIDENCE"
    if score >= 35:
        return "EARLY_SELF_IMPROVEMENT"
    return "DATA_COLLECTION_MODE"


def _report_notes(total_trades: int, total_losses: int, promote_ready: int, min_trades_for_promotion: int) -> list[str]:
    notes = [
        f"NERO has {total_trades} strategy-lab trades and {total_losses} recorded losses for autopsy.",
        "No strategy is changed silently; variants are SHADOW_TEST_ONLY until promotion rules are met.",
        "Live-data-only rule remains mandatory for any decision-quality analysis.",
    ]
    if total_trades < min_trades_for_promotion:
        notes.append(f"Still below {min_trades_for_promotion} trades per candidate; use recommendations as research guidance, not proof.")
    if promote_ready:
        notes.append("At least one candidate is ready for controlled promotion review.")
    return notes
