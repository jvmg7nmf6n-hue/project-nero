# Bellwether PRISM Self-Calibration Part 0 Stop Report

Date: 2026-08-30

## Finding

The v4 directive cannot be implemented as written because its architecture guard
states that `nero_core/` is the confirmed canonical package. In this checkout,
`nero_core/` does not exist. The application and current engines are wired
through `nero_app/core`.

## Evidence

- `app.py` imports core systems from `nero_app.core.*`.
- `app.py` creates and runs `NeroOrchestrator` from `nero_app.core.orchestrator`.
- `app.py` appends verdict records through `nero_app.core.prediction_log`.
- `git ls-files` shows many files under `nero_app/core/`.
- `git ls-files` shows no tracked files under `nero_core/`.

## Confidence

High. This is a direct repository structure mismatch, not a modeling judgment.

## Recommendation

Stop the build at Part 0 and revise the directive before coding:

1. Replace `nero_core/` references with the actual package path `nero_app/core/`,
   or explicitly instruct a migration from `nero_app/core` to `nero_core`.
2. Confirm whether Bellwether/PRISM should integrate into the existing NERO app
   namespace or become a new parallel package.
3. Re-run Part 1 only after the canonical package path is corrected.

## Status

No Bellwether calibration code was added. This report exists to prevent a silent
implementation in the wrong package.
