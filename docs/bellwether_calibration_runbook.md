# Bellwether Calibration Runbook

Project Nero's Bellwether Calibration layer records forecast-like outputs, waits for their pre-registered resolution horizon, and checks the actual market outcome later.

## What This System Does

- Records each new Prediction Lab verdict into `nero_app/data/bellwether_calibration/forecast_ledger.csv`.
- Refuses to score legacy confidence values as real probabilities.
- Resolves BTC against a 24 hour wall-clock horizon.
- Resolves GOLD against the next approximate London-New York weekday session.
- Blocks scoring when the issue feed and resolution feed do not match.
- Writes report files into `reports/bellwether_calibration_report.csv` and `reports/bellwether_calibration_report.json`.
- Records workflow health in `nero_app/data/bellwether_calibration/heartbeats.csv`.

## How To Run Locally

```cmd
cd /d path\to\project-nero
python tools\nero_prediction_lab.py
python tools\nero_bellwether_calibration.py
```

## How To Know It Is Working

- GitHub Actions workflow `Nero Prediction Lab` should be green.
- The workflow log should show `Bellwether calibration complete`.
- The report JSON should contain `schema_version: bellwether_calibration_v1`.
- The dashboard `Calibration` tab should show ledger rows and heartbeat rows.
- Early status can honestly be `INSUFFICIENT_PROBABILITY_DATA`; that is expected until enough probability-valid forecasts exist.

## Expected Warning

Nero's current confidence field is a mapped research confidence, not a mathematically calibrated probability. Therefore the system can resolve outcomes, but it should mark them `NOT_A_PROBABILITY` and block Brier scoring until a future probability model is explicitly built.

## Rollback

If the calibration layer causes workflow trouble, temporarily disable only the `Resolve Bellwether calibration` step in `.github/workflows/nero-prediction-lab.yml`. Do not delete the ledger; it is audit history.

## Integrity Rule

Calibration accuracy does not equal tradable profitability. This layer measures whether forecasts were honest and measurable, not whether a strategy should trade real money.
