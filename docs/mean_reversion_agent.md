# Nero Mean-Reversion Agent

This agent is a standalone paper-trading forward test. It never places real orders and uses no exchange trading API keys.

## Strategy version

`mean-reversion-v1.0.0`

Rules are intentionally fixed and versioned so results are auditable. Parameter changes must be manual and should create a new strategy version.

## Assets

Default GitHub Actions assets:

- `BTC:BTCUSDT`
- `SOL:SOLUSDT`
- `PAXG:PAXGUSDT` as a 24/7 gold proxy

## Entry rules

Long-only, 1-hour closed candles only:

- RSI(14) < 35
- Close below lower Bollinger Band
- Bollinger Band = MA20 +/- 2 standard deviations
- Close above MA200
- Frozen target = MA20 recorded at entry

## Risk and exits

- Separate virtual equity per asset starts at `$10,000`
- Risk = 1% of current equity per trade
- Stop-loss = entry - 1.5 * ATR(14)
- Target = frozen entry-time MA20
- Maximum holding period = 24 hours
- Daily loss guard = -3R
- Fees and slippage are configurable in the workflow
- If SL and target are both touched in the same candle, the agent assumes SL first for conservative testing

## Records

Generated records are stored under:

- `nero_app/data/mean_reversion/state/`
- `nero_app/data/mean_reversion/trades/evaluations.csv`
- `nero_app/data/mean_reversion/trades/trade_events.csv`
- `nero_app/data/mean_reversion/trades/closed_trades.csv`
- `nero_app/data/mean_reversion/trades/runtime_errors.csv`
- `nero_app/data/mean_reversion/heartbeats/heartbeats.csv`
- `reports/mean_reversion_report.csv`
- `reports/mean_reversion_report.json`

## GitHub Actions

Workflow: `.github/workflows/nero-mean-reversion.yml`

Runs every 5 minutes and can also be triggered manually. GitHub scheduled jobs are suitable for periodic forward testing, but they are not true always-on hosting.
