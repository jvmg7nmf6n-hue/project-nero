# NERO Bellwether Calibration v1 Pre-Registration

Date locked: 2026-08-30
Status: Active for new schema-versioned calibration rows only.

This document fixes the first calibration rules before any v1 resolved outcome
is scored. Rows predating the v1 schema are operational history only and must be
reported as `LEGACY_UNSCORED`.

## Scope

Assets:

- BTC
- GOLD

Primary question:

- Does NERO's published forecast value behave like an honest probability of the
  asset closing higher at the intended resolution horizon?

Current Part 1 investigation found that the existing `confidence` field is a
mapped research confidence score, not a proven probability. Therefore v1 stores
new rows but marks NERO confidence-derived rows as `NOT_A_PROBABILITY` until a
probability mapping is explicitly approved and versioned.

## Outcome Definition

BTC:

- Resolution horizon: 24 wall-clock hours after `issued_at_utc`.
- Resolution price: first canonical BTC daily/intraday close available at or
  before the intended horizon without reading beyond the intended timestamp.

GOLD:

- Resolution horizon class: next London-New York overlap approximation.
- Current v1 practical rule: next weekday at 17:00 UTC, skipping Saturday and
  Sunday. This is an approximation until a full exchange-holiday calendar is
  added.
- Resolution price: canonical GOLD close available at or before the intended
  resolution timestamp without forward-fill.

Outcome:

- `up`: resolution price is strictly greater than issue price.
- `down`: resolution price is strictly lower than issue price.
- `flat`: absolute move is inside the deadband.

Deadband:

- v1 deadband is 0.05% of issue price.
- Reports must keep no-deadband and deadband-aware outcomes separate when
  enough records exist.

## Status Taxonomy

Use only these statuses without a schema-version bump:

- `PENDING`
- `RESOLVED`
- `MISSED`
- `UNRESOLVABLE`
- `SOURCE_MISMATCH`
- `LOW_COVERAGE`
- `LEGACY_UNSCORED`
- `SCHEMA_MISMATCH`
- `NOT_A_PROBABILITY`
- `DATA_QUALITY_FAIL`

Every non-`RESOLVED` status must carry a machine-readable `reason_code`.

## Source Rules

Canonical source id is recorded at generation time and again at resolution time.

- BTC primary source id: `binance:BTCUSDT:daily`
- GOLD primary source id: `twelvedata:XAU/USD:daily`

If the resolution source id differs from the issue source id, the record is not
scored and must be marked `SOURCE_MISMATCH`.

No synthetic, fallback, stale, duplicate, non-monotonic, or future-dated prices
may be scored.

## Probability Validity

Current NERO verdict confidence is stored for audit, but v1 marks it
`NOT_A_PROBABILITY` for Brier/Murphy scoring unless a future schema introduces a
pre-registered probability transform.

No Brier score, Brier Skill Score, Murphy decomposition, or reliability plot may
be shown for `NOT_A_PROBABILITY`, `LEGACY_UNSCORED`, or insufficient rows.

## Reliability Bins And Minimums

Bin edges if a future probability-valid value exists:

- 0.00-0.20
- 0.20-0.40
- 0.40-0.60
- 0.60-0.80
- 0.80-1.00

Minimum count per bin:

- 20 records for a bin point.
- 50 effective records per asset before any headline calibration score.

Below floor, display: `insufficient data`.

## Effective-N

Primary estimator:

- Non-overlapping subsample.

Secondary estimator:

- Block bootstrap with block length at least `ceil(horizon_hours / cadence_hours)`.

N_eff / N_raw must be shown with every calibration score once scoring becomes
valid.

## Attribution Review

The first monthly attribution dictionary is fixed as:

- monetary_policy_tone
- geopolitical_escalation
- scheduled_data_surprise
- etf_flow_surprise
- positioning_funding_extreme

Any monthly candidate remains a candidate until it survives train/test,
random-baseline comparison, and FDR control. Calibration quality does not equal
tradable profitability.

