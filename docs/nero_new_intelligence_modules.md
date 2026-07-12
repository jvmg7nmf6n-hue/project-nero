# NERO — New Intelligence Modules

Three additions to Project NERO: **ETF Flow Intelligence**, **Gold Real
Yield Engine**, and **Trade Opportunity Scanner**. All three are
research/paper-trading decision-support tools. **None of them place real
orders, and none of them guarantee any outcome.**

```
nero_app/
  core/
    etf_flow_intelligence.py
    gold_real_yield.py
    trade_opportunity_scanner.py
  ui/
    opportunity_scanner_tab.py
tests/
  test_etf_flow_intelligence.py
  test_gold_real_yield.py
  test_trade_opportunity_scanner.py
app_integration_snippet.py   # copy into app.py
```

## 1. ETF Flow Intelligence Engine

**What it does:** Estimates whether BTC's price action is backed by
institutional spot-ETF demand (IBIT, FBTC, GBTC, BITB, ARKB, HODL).

**Why a proxy:** there is no free real-time API for ETF
creations/redemptions. The engine instead builds a transparent proxy
signal from price + volume + correlation:
- abnormal volume (20-day rolling z-score)
- 5-day return
- rolling correlation with BTC

Each ETF gets a `flow_proxy_reading` (`INFLOW_LIKELY` /
`OUTFLOW_LIKELY` / `NEUTRAL` / `INSUFFICIENT`); these roll up into an
`etf_flow_score` (0–100) and a label from `STRONG_INFLOW_PRESSURE` down
to `DATA_INSUFFICIENT`.

**Interpreting the score:**
- 75–100 `STRONG_INFLOW_PRESSURE` — BTC move looks well-supported by ETF demand.
- 60–74 `MODERATE_INFLOW_PRESSURE` — some institutional tailwind.
- 41–59 `NEUTRAL_FLOW` — no clear signal either way.
- ≤40 `OUTFLOW_PRESSURE` — BTC move looks unsupported / at risk of fading.
- `DATA_INSUFFICIENT` — not enough history to trust the read; don't act on it.

**Limitations:** this is price/volume behavior, not confirmed
creations/redemptions. Treat as directional context, one input among
several — never the sole reason to trade.

**Core function:** `compute_etf_flow_score(etf_price_data, btc_price_data)`
is pure and network-free (tested with synthetic DataFrames).
`fetch_etf_flow_score()` wraps it with a `yfinance` pull and never
raises — any network/data failure degrades to `DATA_INSUFFICIENT`.

## 2. Gold Real Yield Engine

**What it does:** Estimates macro pressure on Gold via
`real_yield ≈ nominal_10Y_yield − breakeven_inflation`. Falling/negative
real yields are historically gold-supportive; rising real yields plus
dollar strength are historically a headwind.

**Two modes:**
- **Official mode** — pass real nominal-yield and breakeven-inflation
  series (e.g. from FRED: `DGS10`, `T10YIE`) → precise real yield.
- **Proxy mode** — pass `^TNX` (yield proxy, auto-divided by 10) and
  `TIP` (inflation-protected bond ETF, used via price momentum as a
  coarse inflation-expectations proxy anchored near 2%). Always labeled
  `is_proxy=True` so it's never confused with the official number.

Optional `DX-Y.NYB`/`UUP` dollar series and `GLD`/gold series add a
`dxy_pressure` adjustment and a `gold_real_yield_correlation` reading.

**Interpreting the score:**
- ≥65 `GOLD_MACRO_SUPPORTIVE` — low/negative real yields, favorable macro backdrop.
- 40–64 `GOLD_MACRO_NEUTRAL` — no strong push either way.
- <40 `GOLD_MACRO_PRESSURE` — elevated real yields (and/or a strong dollar) as a headwind.
- `DATA_INSUFFICIENT` — nominal yield or inflation input missing/too short.

**Limitations:** the proxy inflation estimate is a rough anchor, not a
true breakeven rate — it's directional, not precise. Always check
`is_proxy` before treating the number as macro-grade data.

## 3. Trade Opportunity Scanner

**What it does:** Explains, in plain language, why NERO is or isn't
paper-trading an asset right now. It is **not** an order engine — it
never touches an exchange.

**Inputs (all optional — missing ones are skipped, not faked):**
quant consensus score, sentiment score, ETF flow score (BTC) or real
yield score (GOLD), 90D Sharpe, a `TechnicalSnapshot` (trend, RSI,
z-score, volatility regime, MA20/MA200 position), and current
`PaperTradeState` (open/pending positions).

**Hard gates (always checked first):**
- An existing open/pending paper trade on the asset → `BLOCKED_BY_RISK`
  (duplicate-trade protection).
- `volatility_regime == "EXTREME"` → `BLOCKED_BY_RISK`.

**Otherwise**, sub-scores (quant consensus 30%, sentiment 15%,
ETF-flow/real-yield 20%, technical 20%, risk/Sharpe 15%) blend into a
0–100 `opportunity_score`, and every contributing condition is bucketed
into `passed_conditions` / `failed_conditions` / `near_miss_conditions`
so the reasoning is fully visible — not a black box.

**Decision thresholds:**
- ≥70 with **no** failed conditions → `TRADE_ALLOWED`.
- <35, or failed conditions with score <50 and a hard failure → `BLOCKED_BY_RISK`.
- Everything in between → `WAIT_FOR_CONFIRMATION`.
- No usable inputs at all → `DATA_INSUFFICIENT`.

`direction_bias` (`LONG_BIAS` / `SHORT_BIAS` / `NEUTRAL`) is derived
from trend + MA200 position + z-score sign — a hint for the paper-trade
workflow, not a signal to auto-execute.

## Why this improves NERO's intelligence

- **ETF flow proxy** adds an institutional-demand lens BTC signals
  didn't previously have — distinguishing a retail-driven pump from one
  with real capital behind it.
- **Gold real yield engine** gives Gold a macro-driver read that
  complements the existing technical/quant stack, instead of treating
  Gold purely as a price series.
- **Opportunity Scanner** turns NERO's many existing signals (quant
  consensus, GARCH, sentiment, driver matrix, etc.) into one auditable
  decision with a plain-English "why," which is what actually builds
  trust in a paper-trading accountability log — you can see exactly
  which condition was the near-miss instead of guessing.

## Testing

All three modules expose pure, network-free scoring functions
(`compute_etf_flow_score`, `compute_gold_real_yield_score`,
`scan_trade_opportunity`) that unit tests exercise with synthetic
DataFrames/Series only. The `fetch_*` wrappers do the (optional)
`yfinance` I/O and are intentionally excluded from unit tests — they
never raise, they degrade to `DATA_INSUFFICIENT` on any failure.

Run tests with your existing pytest setup:

```bash
pytest tests/test_etf_flow_intelligence.py tests/test_gold_real_yield.py tests/test_trade_opportunity_scanner.py -v
```

## Dashboard integration

`nero_app/ui/opportunity_scanner_tab.py` renders a new **"Opportunity
Scanner"** Streamlit tab: metric cards, an ETF evidence table,
passed/failed/near-miss condition columns, the final decision, and an
educational disclaimer. See `app_integration_snippet.py` for the exact
`app.py` wiring — copy the import and the tab block, and connect the
marked `TODO`s to your existing quant consensus / sentiment / technical
/ paper-trade-state functions.

## Disclaimer

All outputs are research estimates for paper trading only. Several
figures are explicitly-labeled proxies built from public price/volume
data, not official institutional flow or FRED data. Nothing here is
financial advice, and no score or label implies a guaranteed outcome.
