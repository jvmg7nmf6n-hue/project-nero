# NERO GitHub Repo Intelligence Plan

## Purpose

NERO will use public GitHub repositories as a research library, not as copy-paste source code. The goal is to learn mature architecture patterns for strategy testing, paper trading, data validation, dashboards, alerts, and risk control, then adapt only clean and relevant ideas into NERO.

NERO must remain paper-first until a strategy has enough clean evidence. No repository, model, or backtest result should be treated as proof of live profitability.

## Development Principle

1. Scout useful repositories.
2. Score them for quality, safety, license, and relevance.
3. Extract ideas, not code.
4. Implement NERO-native modules with tests.
5. Keep every new feature audit-friendly, paper-only, and evidence-gated.

## Initial Repository Watchlist

| Repository | Category | Why It Matters For NERO | Useful Ideas | Caution |
|---|---|---|---|---|
| `freqtrade/freqtrade` | Crypto trading bot | Mature dry-run, backtesting, persistence, WebUI, and risk-management discipline | Dry-run architecture, backtest reports, strategy comparison, pair whitelists/blacklists, FreqAI adaptive modeling concepts | Do not copy full bot architecture; NERO should stay lean |
| `freqtrade` docs | Backtesting discipline | Strong warnings on dry/live separation, historic data, fees, and reproducibility | Lookahead checks, recursive analysis, exported trades, fee-aware testing | Backtests can still mislead if data quality is weak |
| `polakowo/vectorbt` / VectorBT | Fast research engine | Rapid multi-asset, multi-parameter strategy testing | Vectorized backtests, parameter sweeps, random baselines, analytics | License and dependency weight must be checked before direct use |
| `drakkar-software/octobot` | Bot architecture and monitoring | Paper/live separation, Web UI, mobile monitoring, AI/social/TradingView connectors | Always-on service design, portfolio monitoring, strategy configuration UI | Too broad for direct integration |
| `backtrader` ecosystem | Backtest/live framework | Long-standing event-driven backtesting model | Broker simulation, analyzers, order lifecycle, trade logging | Some forks vary in quality and maintenance |
| CCXT examples | Exchange data layer | Multi-exchange OHLCV access and unified market metadata | Data-source fallback, symbol normalization, exchange adapters | Exchange API behavior differs; stale-feed checks are mandatory |
| Streamlit/Plotly trading dashboards | UI | Better front-page market display and audit panels | Candles, floating price cards, performance tables, status badges | UI must not hide weak evidence |
| Financial news sentiment repos | AI/sentiment | NERO already has news/sentiment; can improve extraction and scoring | Entity tagging, source confidence, headline impact memory | Social/news sentiment can be noisy and manipulable |

## Scoring Rubric For Each Repo

Each candidate repository will receive a 0-100 integration score:

| Area | Weight | Checks |
|---|---:|---|
| Relevance to NERO | 20 | Does it help strategy testing, data quality, alerts, dashboards, or risk control? |
| Code quality | 15 | Clear structure, tests, typed or readable interfaces, low hidden complexity |
| Maintenance | 15 | Recent commits, active issues, supported Python version |
| License safety | 15 | Allows learning/adaptation; no copy-restricted usage for NERO |
| Backtesting discipline | 15 | Handles fees, slippage, train/test split, lookahead risk, exported trades |
| Paper/live separation | 10 | Clear dry-run/test/live boundaries |
| Observability | 10 | Logs, reports, alerts, health checks, error handling |

Recommended actions:

- `80-100`: study deeply and adapt patterns.
- `60-79`: use for ideas only.
- `40-59`: keep as reference; no immediate build.
- `<40`: ignore.

## First NERO Module To Build

### Strategy Verification Engine

This should be the first GitHub-inspired upgrade because NERO currently has many strategies but still needs a stricter judge.

The engine should evaluate every strategy using:

- Total trades.
- Win rate.
- Net P/L.
- Expectancy in R.
- Profit factor.
- Maximum drawdown.
- Average win/loss ratio.
- Fees and slippage.
- Sample-size status.
- Random-entry baseline.
- Train/test split.
- Recent forward-test performance.
- Data freshness.
- Lookahead/future-data risk flags.

### Verdict Labels

| Verdict | Meaning |
|---|---|
| `PROMOTE_PAPER` | Strong enough for larger paper allocation, not real money |
| `WATCHLIST` | Interesting but sample or edge is not enough |
| `QUARANTINE` | Losses or risk are too high; stop new entries temporarily |
| `REJECT` | Evidence is clearly poor |
| `INSUFFICIENT_SAMPLE` | Not enough trades for judgment |
| `DATA_UNTRUSTED` | Data quality is not clean enough |

### Initial Promotion Gate

A strategy cannot be promoted unless:

- At least 30 closed paper trades.
- Positive expectancy in R.
- Profit factor above 1.10.
- Maximum drawdown inside configured limit.
- Net P/L positive after fees/slippage.
- It beats the random-entry baseline.
- No stale-feed or lookahead-risk flags.

### Quarantine Gate

A strategy should be quarantined if:

- At least 20 closed trades and expectancy is negative.
- Net P/L is materially negative.
- It repeatedly loses in the same market regime.
- Profit factor is below 0.90.
- Latest 10 trades show accelerating drawdown.

## NERO Integration Plan

### Files

Planned files:

- `nero_app/core/strategy_verification.py`
- `tools/nero_strategy_verification.py`
- `reports/strategy_verification_report.csv`
- `reports/strategy_verification_report.json`
- `tests/test_strategy_verification.py`

Dashboard placement:

- Add a `Verification` section inside the existing `TEST Lab`.
- Show one row per strategy.
- Show verdict, sample count, net P/L, expectancy, profit factor, drawdown, and reason.

### GitHub Actions

Add or extend workflow so the verification report runs after the strategy lab:

1. Run strategy lab.
2. Save paper-trade records.
3. Run strategy verification.
4. Commit reports.
5. Send ntfy summary.
6. Include Friday email report.

## Guardrails

- No real exchange trading API keys.
- No live-money auto-trading.
- No strategy promotion from backtest alone.
- No result based on sample data.
- No report without data-source freshness.
- No copying licensed code into NERO without explicit review.
- No guaranteed-profit language.

## Priority Roadmap

### Phase 1: Repo Scout Report

Create a scored shortlist of 20 repositories grouped by:

- Trading bots.
- Backtesting engines.
- Data pipelines.
- Dashboards.
- AI/sentiment.
- Automation/alerts.

### Phase 2: Strategy Verification Engine

Build the NERO-native strategy judge described above.

### Phase 3: Bias-Proof Backtesting Upgrade

Add:

- Lookahead-risk checks.
- Random-entry baseline.
- Train/test split summary.
- Data audit before each test.

### Phase 4: Strategy Quarantine Automation

Automatically block new paper entries for strategies with poor evidence while keeping their historical records.

### Phase 5: Dashboard Evidence Upgrade

Make every dashboard result explain:

- Data source.
- Last refresh time.
- Paper/live/test mode.
- Sample size.
- Confidence quality.
- Why a strategy is allowed or blocked.

## Current Recommendation

Start with the Strategy Verification Engine. NERO already has many paper strategies and enough early loss evidence to justify a stricter judge. This will reduce blind testing and make future strategy research more disciplined.
