# Squawk

Real-time arrival disruption detection, explanation and forecasting for London Heathrow, built on Databricks
(DataExpert.io capstone).

Live ADS-B positions from adsb.lol (OpenSky is supported too, but it blocks cloud-provider IPs) and Heathrow
weather from aviationweather.gov flow through a Lakeflow pipeline
(Bronze → Silver). A detector finds holding episodes and go-arounds (Gold) and puts them in a Lakebase queue.
A LangGraph agent explains each event from the evidence and forecasts next-hour holding. Analysts review the
agent's explanations in a Databricks App, and Lakebase Change Data Feed streams every change into Delta analytics
that measure how good the agent is.

## Layout

```
app/
  app.py, app.yaml, requirements.txt   Databricks App (Streamlit)
  squawk_lib/                          shared code (the app folder is deployed as-is, so it lives here)
    config.py      names, bounding box, stack/runway coordinates, thresholds  <- edit this first
    sources.py     adsb.lol + OpenSky + aviationweather.gov clients (retries, backoff, call log)
    detect.py      holding + go-around rules (pure pandas, unit-tested)
    db.py          Lakebase connections (OAuth token) and Delta readers
    store.py       every Lakebase read/write
    agent.py       LangGraph agent: 5 read tools, 2 write tools, prompts, guardrails
notebooks/
  00_setup.py            catalog, schemas, volume, secrets, API test, reference + Gold tables
  01_poller.py           job task: polls the APIs, lands JSON in the volume
  02_lakebase_setup.py   Lakebase tables, agent role, app grants, Change Data Feed
  03_detector.py         job task: Silver -> Gold episodes -> Lakebase queue
  04_agent_worker.py     job task: agent assessments, hourly forecasts, forecast scoring
pipeline/
  etl.py         Lakeflow: Auto Loader Bronze, Silver positions + weather
  analytics.py   Lakeflow: analytics from Lakebase CDF (its own scheduled pipeline, phase 6)
tests/
  test_detect.py    python tests/test_detect.py
  test_sources.py   python tests/test_sources.py
```

Follow the step-by-step build guide (`docs/build-guide.html`) phase by phase.
