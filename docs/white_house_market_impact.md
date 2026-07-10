# NERO White House Market Impact Memory

This module estimates how White House communications may affect BTC and Gold by comparing a new headline, press release, executive order, or press-conference summary with historical White House events.

## Why Event Study, Not Simple Correlation

A simple correlation between White House speech count and BTC/Gold price is too noisy. Markets also react to Fed policy, CPI, DXY, yields, wars, ETF flows and equity sentiment. NERO therefore treats White House communications as event shocks and asks:

- What type of event is this?
- Which historical White House events are similar?
- What happened to BTC and Gold after those events?
- Is the impact BTC-specific, Gold/safe-haven specific, or mixed?

## First Data Store

Seed file:

```text
nero_app/data/white_house_market_events.csv
```

Each row contains:

- event date
- source and URL
- administration and speaker
- event type
- headline and summary
- tags
- BTC/Gold prices and forward returns where available
- BTC/Gold impact score
- confidence
- notes

## Current Tags

NERO detects tags such as:

- crypto_regulation
- crypto_friendly_policy
- policy_hostile
- policy_clarity
- strategic_bitcoin_reserve
- stablecoin_legislation
- sanctions
- war
- geopolitical_risk
- risk_off
- safe_haven
- oil_supply_risk
- inflation_risk

## Example

```bash
python tools/nero_white_house_impact.py --text "White House announces strategic bitcoin reserve and digital asset stockpile"
```

Example output:

```text
BTC impact: bullish/high positive impact (79/100)
Gold impact: low impact (11/100)
```

## Interpretation

- Crypto-friendly policy tends to affect BTC more than Gold.
- Sanctions/geopolitical shocks tend to affect Gold more directly, while BTC impact can be mixed.
- Policy-hostile crypto communications can pressure BTC through regulatory and custody friction.

## Next Steps

1. Expand event memory to five years of official White House / NARA / Presidency Project records.
2. Attach real BTC and Gold forward returns for 1h, 4h, 1d, 7d and 30d.
3. Add DXY, Nasdaq, VIX and yields as control variables.
4. Add dashboard tab: White House Impact.
5. Add high-impact mobile alerts for new official White House events.
