from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nero_app.core.schema import MacroEvent


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_macro_events() -> list[MacroEvent]:
    frame = pd.read_csv(DATA_DIR / "macro_events.csv", parse_dates=["event_date"])
    events: list[MacroEvent] = []
    for row in frame.to_dict("records"):
        tags = [tag.strip() for tag in str(row["tags"]).split("|") if tag.strip()]
        asset_bias = {
            "BTC": float(row["btc_bias"]),
            "SPY": float(row["spy_bias"]),
            "ETH": float(row["eth_bias"]),
            "GOLD": float(row["gold_bias"]),
        }
        events.append(
            MacroEvent(
                event_id=str(row["event_id"]),
                event_date=row["event_date"].date(),
                title=str(row["title"]),
                narrative=str(row["narrative"]),
                tags=tags,
                asset_bias=asset_bias,
            )
        )
    return events


def load_price_history() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=520, freq="D")
    rng = np.random.default_rng(7)
    drift = np.linspace(0, 0.55, len(dates))
    cycle = np.sin(np.linspace(0, 20, len(dates))) * 0.08
    noise = rng.normal(0, 0.018, len(dates)).cumsum()
    close = 100 * (1 + drift + cycle + noise).clip(min=0.2)
    open_ = close * (1 + rng.normal(0, 0.006, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.025, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.025, len(dates)))
    volume = rng.integers(900_000, 5_000_000, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
