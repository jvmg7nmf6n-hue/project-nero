from __future__ import annotations

import pandas as pd

from nero_app.core.schema import BacktestResult, HistoricalMatch


def run_event_backtest(
    matches: list[HistoricalMatch],
    prices: pd.DataFrame,
    forward_days: int = 14,
) -> BacktestResult:
    if not matches:
        return BacktestResult(average_forward_return=0.0, win_rate=0.0, sample_count=0, trades=[])

    frame = prices.sort_values("date").reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    trades: list[dict[str, object]] = []

    for match in matches:
        index = _nearest_index(frame, match.event_date)
        exit_index = min(index + forward_days, len(frame) - 1)
        if exit_index <= index:
            continue
        entry = float(frame.loc[index, "close"])
        exit_ = float(frame.loc[exit_index, "close"])
        forward_return = (exit_ - entry) / entry
        expected_sign = 1 if match.forward_bias >= 0 else -1
        trades.append(
            {
                "event": match.title,
                "event_date": match.event_date.isoformat(),
                "similarity": match.similarity,
                "forward_return": round(forward_return, 4),
                "aligned": forward_return * expected_sign > 0,
            }
        )

    if not trades:
        return BacktestResult(average_forward_return=0.0, win_rate=0.0, sample_count=0, trades=[])

    returns = [float(trade["forward_return"]) for trade in trades]
    wins = [bool(trade["aligned"]) for trade in trades]
    return BacktestResult(
        average_forward_return=sum(returns) / len(returns),
        win_rate=sum(wins) / len(wins),
        sample_count=len(trades),
        trades=trades,
    )


def _nearest_index(frame: pd.DataFrame, event_date) -> int:
    distances = frame["date"].map(lambda value: abs((value - event_date).days))
    return int(distances.idxmin())
