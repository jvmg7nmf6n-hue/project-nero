# Project Nero MVP

Project Nero is a local research dashboard for macro-aware market scenario analysis. This MVP turns the product document into an offline-first Python application with a testable multi-agent backend and a Streamlit-ready interface.

## Included

- Local macro knowledge store with historical event matching
- Pydantic data models for events, candles, analysis requests, agent outputs, and verdicts
- Three-agent orchestration: Brain, Market Assessment, and Verdict
- Simple event backtester over generated sample price data
- Optional Binance daily candle connector for BTC and ETH, with sample-data fallback
- Optional Twelve Data connector for GOLD, OIL, and FedEx stock (FDX)
- Streamlit dashboard shell
- Educational research disclaimer

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

In the dashboard sidebar, turn on `Live market data` to try Binance candles for BTC or ETH. If the API or network is unavailable, Nero falls back to generated sample candles and shows the fallback status in the interface.

Run tests:

```powershell
python -m unittest discover -s tests
```

Nero is for educational research and historical probability modeling only. It does not provide financial, investment, legal, tax, or execution advice.

## Twelve Data

Use the `Twelve Data API key` field in the sidebar for GOLD, OIL, and FDX live candles, or set `TWELVE_DATA_API_KEY` before running Streamlit. Current symbol mapping: GOLD=`XAU/USD`, OIL=`WTI/USD`, FDX=`FDX`.

## Prediction Log

Press `Run Nero Verdict` to save the current analysis. Saved rows appear in the `Prediction Log` tab and are written to `nero_app/data/prediction_log.csv`.

## Local Settings

The Twelve Data API key and default sidebar toggles are saved locally to `nero_app/data/local_settings.json`. This file is ignored by git.

## Cloud Deployment

Nero is deployment-ready for Render, Streamlit Cloud, Railway, VPS, or AWS. Keep secrets out of git and set them as environment variables in the hosting dashboard.

### Required environment variables

```text
PREFER_LIVE=true
USE_LATEST_NEWS=true
MOBILE_ALERTS_ENABLED=true
TWELVE_DATA_API_KEY=your_twelve_data_key
GEMINI_API_KEY=your_gemini_key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SENDER_EMAIL=your_sender_gmail
EMAIL_APP_PASSWORD=your_gmail_app_password
RECEIVER_EMAIL=tareekh39@gmail.com
```

### Render

Use `render.yaml` blueprint or create a Python web service manually.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Streamlit Cloud

Set the app entry point to:

```text
app.py
```

Then add the environment variables/secrets from the list above in Streamlit Cloud settings.

### Important

`nero_app/data/local_settings.json` and `nero_app/data/prediction_log.csv` are local-only and ignored by git. On cloud, settings should come from environment variables. Prediction logs on free hosts may be ephemeral unless persistent storage or a database is added.

