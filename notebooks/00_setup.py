# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC
# MAGIC Run this once, top to bottom. It:
# MAGIC 1. creates the catalog, schemas and landing volume
# MAGIC 2. stores your API credentials in a Databricks secret scope
# MAGIC 3. tests both APIs
# MAGIC 4. loads the aircraft reference table
# MAGIC 5. creates the Gold tables the detector writes to
# MAGIC
# MAGIC **Before you start:** edit `app/squawk_lib/config.py` if you need a different catalog name.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.81" requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import config

print("Catalog:  ", config.CATALOG)
print("Schemas:  ", config.SCHEMA_RAW, config.SCHEMA_BRONZE, config.SCHEMA_SILVER, config.SCHEMA_GOLD, config.SCHEMA_ANALYTICS)
print("Volume:   ", config.VOLUME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Catalog, schemas and volume
# MAGIC If `CREATE CATALOG` fails with a permissions error, ask your bootcamp admin for a catalog, or set
# MAGIC `CATALOG` in `config.py` to one you already own (e.g. the one you used for the Zachy homework) and rerun.

# COMMAND ----------

def run(statement):
    try:
        spark.sql(statement)
        print("OK  ", statement)
    except Exception as e:
        print("FAIL", statement, "\n     ", str(e).splitlines()[0])

run(f"CREATE CATALOG IF NOT EXISTS {config.CATALOG}")
for schema in {config.SCHEMA_RAW, config.SCHEMA_BRONZE, config.SCHEMA_SILVER, config.SCHEMA_GOLD, config.SCHEMA_ANALYTICS}:
    run(f"CREATE SCHEMA IF NOT EXISTS {config.CATALOG}.{schema}")
run(f"CREATE VOLUME IF NOT EXISTS {config.CATALOG}.{config.SCHEMA_RAW}.landing")

for path in (config.LANDING_OPENSKY, config.LANDING_WEATHER, config.LANDING_REFERENCE, f"{config.VOLUME}/_tmp"):
    os.makedirs(path, exist_ok=True)
print(os.listdir(config.VOLUME))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Store your API credentials as secrets
# MAGIC 1. Run the next cell once: it adds three text boxes at the top of the notebook.
# MAGIC 2. Paste your bootcamp LLM API key into `llm_api_key`. The two OpenSky boxes are optional -
# MAGIC    leave them empty unless you set `DATA_SOURCE = "opensky"` in config.py.
# MAGIC 3. Run the cell after it to save them, then run the clean-up cell so the values don't stay on screen.

# COMMAND ----------

dbutils.widgets.text("opensky_client_id", "")
dbutils.widgets.text("opensky_client_secret", "")
dbutils.widgets.text("llm_api_key", "")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
if config.SECRET_SCOPE not in [s.name for s in w.secrets.list_scopes()]:
    w.secrets.create_scope(scope=config.SECRET_SCOPE)
    print("Created secret scope", config.SECRET_SCOPE)

for key in ("opensky_client_id", "opensky_client_secret", "llm_api_key"):
    value = dbutils.widgets.get(key).strip()
    if value:
        w.secrets.put_secret(scope=config.SECRET_SCOPE, key=key, string_value=value)
        print("Saved secret", key)
    else:
        print("Skipped (empty)", key)

print("Secrets now in scope:", [s.key for s in w.secrets.list_secrets(scope=config.SECRET_SCOPE)])

# COMMAND ----------

dbutils.widgets.removeAll()   # clean-up: removes the text boxes and their values

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test both APIs
# MAGIC You should see a list of aircraft over London and the latest Heathrow METAR.
# MAGIC (`credits remaining` is only meaningful for OpenSky; adsb.lol has no credit budget.)

# COMMAND ----------

import json

import time

import pandas as pd

from squawk_lib.sources import fetch_weather, make_client


def opensky_secret(key):
    try:
        return dbutils.secrets.get(config.SECRET_SCOPE, key)
    except Exception:
        return None            # only needed if DATA_SOURCE is "opensky"


client = make_client(config.DATA_SOURCE,
                     client_id=opensky_secret("opensky_client_id"),
                     client_secret=opensky_secret("opensky_client_secret"))
rows = client.fetch_records(fetched_at=time.time())
print("Source:", config.DATA_SOURCE, "| call log:", client.log)
assert rows, "No positions returned - see the call log above"
print(f"{len(rows)} aircraft in range right now; credits remaining today: {client.credits_remaining}")
display(pd.DataFrame(rows)[["icao24", "callsign", "latitude", "longitude", "baro_altitude",
                            "geo_altitude", "velocity", "on_ground"]].head(10))

weather = fetch_weather()
print(f"{len(weather)} weather reports")
for w_row in weather[:2]:
    print(w_row["kind"], json.loads(w_row["raw_json"]).get("rawOb") or json.loads(w_row["raw_json"]).get("rawTAF"))
print(client.log)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aircraft reference table
# MAGIC Downloads OpenSky's aircraft database (type and operator for each transponder address). It's used to
# MAGIC enrich Silver positions. If the download fails, an empty table is created and enrichment is simply skipped.

# COMMAND ----------

import requests

AIRCRAFT_DB_URL = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"
csv_path = f"{config.LANDING_REFERENCE}/aircraftDatabase.csv"
ref_table = config.TABLES["aircraft_ref"]
wanted = ["icao24", "registration", "typecode", "model", "manufacturername", "operator", "operatoricao"]

try:
    with requests.get(AIRCRAFT_DB_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(csv_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        quote = "'" if f.readline().startswith("'") else '"'
    raw = (spark.read.option("header", True).option("quote", quote).option("escape", quote)
           .option("multiLine", True).csv(csv_path))
    from pyspark.sql import functions as F
    cols = [F.col(c) if c in raw.columns else F.lit(None).cast("string").alias(c) for c in wanted]
    ref = (raw.select(*cols)
           .withColumn("icao24", F.lower(F.trim("icao24")))
           .where(F.col("icao24").rlike("^[0-9a-f]{6}$"))
           .dropDuplicates(["icao24"]))
    ref.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(ref_table)
    print("Loaded", spark.table(ref_table).count(), "aircraft")
except Exception as e:
    print("Aircraft download failed, creating an empty table instead:", e)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {ref_table} ({', '.join(c + ' STRING' for c in wanted)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tables written outside the pipeline
# MAGIC The poller writes an API call log; the detector writes Gold episodes and stack occupancy.

# COMMAND ----------

T = config.TABLES
spark.sql(f"""CREATE TABLE IF NOT EXISTS {T['api_call_log']} (
    source STRING, called_at TIMESTAMP, http_status INT, latency_ms BIGINT, credits_remaining INT, error STRING)""")

spark.sql(f"""CREATE TABLE IF NOT EXISTS {T['disruption_episodes']} (
    event_id STRING, event_type STRING, icao24 STRING, callsign STRING, location STRING,
    started_at TIMESTAMP, ended_at TIMESTAMP, duration_s BIGINT, trigger_ts TIMESTAMP,
    min_alt_ft DOUBLE, max_alt_ft DOUBLE, max_turn_deg DOUBLE, n_points BIGINT,
    first_detected_at TIMESTAMP, updated_at TIMESTAMP)
  CLUSTER BY (started_at)
  COMMENT 'Holding episodes and go-arounds detected by 03_detector'""")

spark.sql(f"""CREATE TABLE IF NOT EXISTS {T['stack_occupancy_1min']} (
    minute TIMESTAMP, stack STRING, holding_count INT, updated_at TIMESTAMP)
  CLUSTER BY (minute)""")

spark.sql(f"""CREATE OR REPLACE VIEW {T['holding_hourly']} AS
  SELECT date_trunc('HOUR', started_at) AS hour,
         count_if(event_type = 'holding') AS holding_flights,
         round(avg(CASE WHEN event_type = 'holding' THEN duration_s END) / 60, 1) AS mean_hold_min,
         round(max(CASE WHEN event_type = 'holding' THEN duration_s END) / 60, 1) AS max_hold_min,
         count_if(event_type = 'go_around') AS go_arounds
  FROM {T['disruption_episodes']}
  GROUP BY 1""")

for name in ("api_call_log", "aircraft_ref", "disruption_episodes", "stack_occupancy_1min", "holding_hourly"):
    print("OK ", T[name])
print("\nSetup complete. Next: 01_poller.")
