# Bellwether Stage 2 Report

## Cycle Intelligence Gate

Implemented a Tier A Cycle Intelligence layer for Project NERO.

- Scope: BTC, ETH, SOL, and PAXG as tokenized-gold proxy.
- Data rule: daily closes only; no synthetic fallback, no guessed values.
- Core fields: latest close, SMA200, Mayer Multiple, distance from SMA200, MM percentile, SMA200 30-day slope, drawdown from available high.
- Data sufficiency: at least 200 clean daily closes required before Mayer Multiple is reported.
- PAXG caveat: PAXG is tokenized gold, not spot XAU/USD.
- Consumer: dashboard Cycle Intel tab and `reports/cycle_intelligence_report.*`.
- Trade-rule boundary: this layer does not change live entry, quarantine, promotion, confidence, or strategy scoring.

## Layer Decisions

- MM Percentile Engine: built as real price arithmetic.
- 200D Slope Engine: built as real price arithmetic; requires enough SMA200 history for a 30-day slope.
- Drawdown From High Engine: built against the highest close in available clean history.
- ETF Flow Confirmation: not built as a proxy; marked unavailable unless a real configured feed exists.
- Liquidity Pressure Layer: not built as a composite in this tier.
- Derivatives Heat Layer: not built without real funding, open-interest, and liquidation feeds.
- Cycle Similarity Memory: deferred; no composite cycle score was added.

## Redundancy Check

The report includes a correlation matrix for new price-derived fields. Mayer Multiple and distance from SMA200 are algebraically related, so they are shown for readability but must not be double-counted in future scoring.
