# Bellwether PRISM Self-Calibration Part 1 Investigation

Date: 2026-08-30
Status: STOP AFTER PART 1, as required by the v4 directive.

This report investigates whether Project NERO currently has a probability-valid
Bellwether/PRISM forecast stream that can be resolved and scored. It does not
add calibration code.

## Executive Finding

FINDING: The current repository does not contain a runtime Bellwether/PRISM
forecast producer named "Chance Price Rises" or "Robot Consensus". The active
forecast-like path is NERO's core verdict/prediction log path.

CONFIDENCE: High. The app imports and runs `NeroOrchestrator` from
`nero_app.core.orchestrator` (`app.py:37`, `app.py:1509-1510`). Searches for
`Chance Price Rises`, `Robot Consensus`, `published_value`, `forecast_id`,
`Brier`, `Murphy`, and `N_eff` in runtime code returned no Bellwether
calibration implementation. The only Bellwether hit was documentation:
`docs/bellwether_stage2_report.md:1`.

RECOMMENDATION: Do not build Brier/Murphy calibration on the existing
`confidence` field. Treat current prediction rows as operational history and
mark them `LEGACY_UNSCORED` for the new calibration schema. Build a new
generation-time forecast record first, with explicit probability semantics.

## 1.1 Forecast Producer Location

FINDING: The current app's forecast-like producer is the NERO core orchestrator,
not a Bellwether/PRISM probability publisher.

Evidence:

- `app.py:37` imports `NeroOrchestrator`.
- `app.py:1509-1510` creates `AnalysisRequest` and runs
  `orchestrator.run(request, price_history)`.
- `nero_app/core/orchestrator.py:10-15` wires `BrainAgent`,
  `MarketAssessmentAgent`, and `VerdictAgent`.
- `nero_app/core/orchestrator.py:17-26` runs those agents and returns
  `NeroResult`.
- `nero_app/core/agents.py:32-75` creates the final `VerdictOutput`.

CONFIDENCE: High.

RECOMMENDATION: Rename the Part 1 implementation target from Bellwether/PRISM
to NERO verdict calibration unless a separate Bellwether forecast producer is
added later.

## 1.2 What The Forecast Actually Is

FINDING: The stored `confidence` field is a mapped display/confidence score,
not a proven binary probability of price rising.

Evidence:

- `nero_app/core/agents.py:37-44` builds a weighted composite from thematic,
  momentum, technical bias, fair value gap, liquidity sweep, and regime.
- `nero_app/core/agents.py:46-51` maps the composite into bullish, bearish, or
  neutral.
- `nero_app/core/agents.py:53` maps absolute composite and match quality into
  `confidence`.
- `nero_app/core/agents.py:64-67` displays the result as a research verdict
  with confidence.
- `nero_app/core/schema.py:66-71` stores `direction`, `confidence`,
  `risk_score`, `summary`, and `drivers`, but no calibrated probability field.

CONFIDENCE: High.

RECOMMENDATION: Mark existing rows as `NOT_A_PROBABILITY` for probability
scoring. A Brier score must wait until the pre-registration file defines
whether the generated value is binary `P(up)`, three-way probability, or a
non-probability confidence display.

## 1.2a Empirical Distribution Of Existing Published Values

FINDING: Existing live prediction rows occupy a narrow confidence range.

Measured on `nero_app/data/prediction_log.csv`:

- Total rows: 101
- Live-source rows: 99
- Evaluated live rows: 97
- Pending live rows: 2
- Live assets: GOLD 50, BTC 49
- Live directions: bullish 52, neutral 39, bearish 8
- Confidence range: min 0.350, mean 0.386, max 0.619
- Timestamp range: 2026-07-10T12:32:22 to 2026-08-30T05:28:12

CONFIDENCE: High for the current local checkout.

RECOMMENDATION: Publish this distribution in the calibration report as legacy
context only. Do not infer that 0.386 means "38.6% chance of rise".

## 1.2b Neutral Handling

FINDING: The existing truth report counts neutral verdicts separately and does
not treat them as binary wins/losses.

Evidence:

- `nero_app/core/prediction_log.py:143-148` assigns bullish, bearish, or
  neutral outcomes.
- `nero_app/core/prediction_log.py:203-209` calculates win rate from wins plus
  misses, excluding neutral from the denominator.
- The current live ledger contains 39 neutral rows out of 99 live rows.

CONFIDENCE: High.

RECOMMENDATION: The calibration schema must explicitly choose binary exclusion
or a three-way outcome. If neutral remains available, the v4 directive requires
three-way handling with a volatility-normalized deadband.

## 1.3 Cycle Count And Real Denominator

FINDING: The existing ledger cannot prove scheduled cycles or fire rate. It can
prove prediction rows and successful record commits.

Evidence:

- `.github/workflows/nero-prediction-lab.yml:4-5` schedules the Prediction Lab
  at `17 */6 * * *`.
- `.github/workflows/nero-prediction-lab.yml:21-25` runs
  `tools/nero_prediction_lab.py` for BTC and GOLD with a one-day horizon.
- `nero_app/core/prediction_lab.py:46` skips recording when the market feed is
  non-live, empty, or already recorded for the current asset/date/horizon.
- `nero_app/core/prediction_lab.py:119-130` enforces one Prediction Lab record
  per asset, entry date, and horizon.
- `git log --all --grep="Update Nero prediction lab records"` returned 88
  record-update commits in the current checkout.

CONFIDENCE: Medium. The number of record commits is reliable for the checkout,
but not identical to fired workflow cycles.

RECOMMENDATION: Add a first-class scheduler heartbeat table before reporting
fire rate. The resolver should mark expected-but-missing cycles as `MISSED`
only after a heartbeat schedule exists.

## 1.4 Cadence vs Horizon Overlap

FINDING: Prediction Lab cadence is every 6 hours and horizon is 1 day, so up to
4 scheduled forecasts per asset can overlap in outcome windows. The actual
stored rows are further reduced by the once-per-asset-date gate.

Evidence:

- `.github/workflows/nero-prediction-lab.yml:5` uses a six-hour cron cadence.
- `.github/workflows/nero-prediction-lab.yml:25` sets
  `PREDICTION_LAB_HORIZON_DAYS: "1"`.
- `nero_app/core/prediction_lab.py:119-130` stores at most one record per
  asset/date/horizon for Prediction Lab.

CONFIDENCE: High.

RECOMMENDATION: For calibration, define cadence explicitly. If multiple daily
forecasts are later enabled, use non-overlapping subsamples or serial
block-bootstrap/Newey-West treatment. Do not use iid bootstrap.

## 1.5 Existing Calendar Logic

FINDING: No reusable London-New York session/holiday calendar was found in the
current runtime code search.

Evidence:

- Search terms used: `holiday`, `calendar`, `London`, `New York`, `session`,
  `NYSE`, `LSE`, `market hours`.
- Existing price handling is date/candle based in `prediction_log.py` and
  `market_data.py`; no session-aware resolver is present in the cited files.

CONFIDENCE: Medium. This is based on text search and inspection of the active
prediction path.

RECOMMENDATION: Build a small calendar abstraction for Gold resolution before
Part 3. BTC can use a 24h wall-clock horizon.

## 1.6 Canonical Price Source And Same-Source Rule

FINDING: The current market source is stored as a human-readable string only.
There is no stable `source_id` or `resolution_source_id` for same-source
enforcement.

Evidence:

- `nero_app/core/prediction_log.py:13-35` defines ledger columns; it has
  `data_source` but no `source_id`, `resolution_source_id`, or schema version.
- `nero_app/core/prediction_log.py:68` stores `data_source`.
- `nero_app/core/prediction_log.py:109-156` evaluates rows using whichever
  price frame was passed into the function; it does not compare issue source
  with resolution source.
- `nero_app/core/market_data.py:125-164` uses Binance, then Coinbase, then
  Kraken for crypto daily live candles.
- `nero_app/core/market_data.py:166-180` uses Twelve Data for GOLD/SILVER/OIL
  when a key is configured; otherwise it falls back.

CONFIDENCE: High.

RECOMMENDATION: Add stable source IDs in the new schema. Resolution must mark
`SOURCE_MISMATCH` if the issue and resolution source IDs differ.

## 1.7 Effective-N Reusability Check

FINDING: No reusable serial dependence-adjusted effective-N mechanism was found
for overlapping forecast outcomes. Existing bootstrap/random-baseline utilities
belong to strategy backtesting/repair, not Bellwether forecast calibration.

Evidence:

- Runtime search for `N_eff`, `effective-N`, `Brier`, and `Murphy` returned no
  calibration implementation.
- Existing random baseline and bootstrap logic appears in strategy modules such
  as `nero_app/core/range_mean_reversion.py` and
  `nero_app/core/strategy_repair_lab.py`, which are trade-strategy harnesses,
  not forecast-calibration resolvers.

CONFIDENCE: Medium-high.

RECOMMENDATION: Implement the minimal correct serial method in the calibration
module: primary non-overlapping subsample, secondary block bootstrap or
Newey-West. Do not reuse cross-sectional/FDR logic if later found because this
problem is serial overlap.

## 1.8 Haircut Interaction

FINDING: No `trade_recommendation.py` or permanent 0.7x calibration haircut was
found in the current checkout's active NERO forecast path.

Evidence:

- Search for `trade_recommendation` returned no runtime file.
- Search for `0.7` found unrelated threshold/tests but no calibration haircut
  in the forecast/prediction path.
- `app.py:1509-1512` runs `NeroOrchestrator` and then applies only the
  White House modifier before saving/verdict display.

CONFIDENCE: High for current repo state.

RECOMMENDATION: Treat the haircut premise as stale for this repository. If a
haircut is later introduced, the forecast record must store whether the
published value is pre-haircut or post-haircut.

## Blockers Before Part 2/3

FINDING: The v4 directive's calibration build cannot safely proceed as a direct
implementation against the current `prediction_log.csv`.

CONFIDENCE: High.

RECOMMENDATION:

1. Add `docs/preregistration/calibration_v1.md` before examining any new
   resolved outcomes.
2. Define the generated forecast value as either real binary `P(up)`, a
   three-way probability, or a non-probability confidence score.
3. Introduce a new append-only calibration ledger with `forecast_id`,
   `source_id`, `schema_version`, intended resolution timestamp, and status
   taxonomy.
4. Mark current rows `LEGACY_UNSCORED` and keep them out of Brier/Murphy
   scoring.
5. Add Gold session resolution and BTC 24h wall-clock resolution.
6. Add source-mismatch, stale-feed, and no-forward-fill gates.
7. Add scheduler heartbeat records before claiming fire rate or missed cycles.

## Part 1 Decision

FINDING: Part 1 did not clear the probability-validity gate.

CONFIDENCE: High.

RECOMMENDATION: Stop here and report. The next safe engineering step is a
separate Part 0.3/Part 3-style schema and resolver foundation, not statistical
scoring.

