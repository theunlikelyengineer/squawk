# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Poller
# MAGIC
# MAGIC Polls OpenSky every 30 s and the weather API every 10 min, and lands the raw responses as JSON-lines
# MAGIC files in the landing volume. Auto Loader (in the Lakeflow pipeline) picks them up from there.
# MAGIC
# MAGIC Runs for `duration_minutes`, then exits. As a job task with a **continuous** trigger, the job restarts
# MAGIC it straight away, so it runs 24/7 until you pause the job.

# COMMAND ----------

# MAGIC %pip install -q requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("duration_minutes", "5")   # the job sets 60
dbutils.widgets.text("poll_seconds", "30")

# COMMAND ----------

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import config
from squawk_lib.sources import OpenSkyClient, fetch_weather, state_rows

DURATION_S = int(dbutils.widgets.get("duration_minutes")) * 60
POLL_S = int(dbutils.widgets.get("poll_seconds"))
TMP = f"{config.VOLUME}/_tmp"
os.makedirs(TMP, exist_ok=True)

call_log = []
client = OpenSkyClient(
    dbutils.secrets.get(config.SECRET_SCOPE, "opensky_client_id"),
    dbutils.secrets.get(config.SECRET_SCOPE, "opensky_client_secret"),
    log=call_log,
)


def land(rows, folder, prefix):
    """Write rows as one JSON-lines file. Written to _tmp first, then moved in one step,
    so Auto Loader never sees a half-written file."""
    if not rows:
        return None
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    final_dir = f"{folder}/{day}"
    os.makedirs(final_dir, exist_ok=True)
    name = f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.json"
    tmp_path, final_path = f"{TMP}/{name}", f"{final_dir}/{name}"
    with open(tmp_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        # Some volume mounts don't support rename: fall back to copying.
        with open(tmp_path) as src, open(final_path, "w") as dst:
            dst.write(src.read())
        os.remove(tmp_path)
    return final_path


def flush_log():
    if call_log:
        as_int = lambda v: None if v is None else int(v)  # noqa: E731
        rows = [(r["source"], r["called_at"], as_int(r["http_status"]), as_int(r["latency_ms"]),
                 as_int(r["credits_remaining"]), r["error"]) for r in call_log]
        (spark.createDataFrame(rows, schema="source STRING, called_at TIMESTAMP, http_status INT, latency_ms BIGINT, "
                                          "credits_remaining INT, error STRING")
         .write.mode("append").saveAsTable(config.TABLES["api_call_log"]))
        call_log.clear()

# COMMAND ----------

started = time.time()
last_weather, cycle, aircraft_total = 0.0, 0, 0

while time.time() - started < DURATION_S:
    tick = time.time()
    cycle += 1

    try:
        result = client.fetch_states()
        if result is not None:
            api_time, states = result
            rows = state_rows(api_time, states, fetched_at=tick)
            land(rows, config.LANDING_OPENSKY, "states")
            aircraft_total += len(rows)
    except Exception as e:                      # e.g. token endpoint down: skip this cycle, keep running
        print(f"cycle {cycle}: OpenSky failed, skipping ({type(e).__name__}: {e})")

    if tick - last_weather >= config.WEATHER_POLL_SECONDS:
        try:
            land(fetch_weather(log=call_log), config.LANDING_WEATHER, "weather")
        except Exception as e:
            print(f"cycle {cycle}: weather failed, will retry in 10 min ({type(e).__name__}: {e})")
        last_weather = tick

    if cycle % 10 == 0:
        try:
            flush_log()
        except Exception as e:
            print(f"cycle {cycle}: couldn't write the API call log ({type(e).__name__}: {e})")
        print(f"{datetime.now(timezone.utc):%H:%M:%S}Z cycle {cycle}: {aircraft_total} positions landed so far, "
              f"credits remaining {client.credits_remaining}")

    time.sleep(max(0.0, POLL_S - (time.time() - tick)))

flush_log()
print(f"Done: {cycle} cycles, {aircraft_total} positions landed.")
