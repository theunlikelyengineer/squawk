# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Detector
# MAGIC
# MAGIC Every 15 s: reads the last 3 hours of Silver positions, runs the holding and go-around rules
# MAGIC (`squawk_lib/detect.py`), then
# MAGIC * merges new or extended episodes into `gold.disruption_episodes`
# MAGIC * upserts them into Lakebase `disruption_events` (which puts them in the analyst's queue)
# MAGIC * refreshes `gold.stack_occupancy_1min`
# MAGIC
# MAGIC Episode IDs are deterministic (a hash of aircraft + type + start time), so re-detecting the same
# MAGIC episode every cycle never creates duplicates.

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]" "databricks-sdk>=0.81"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("duration_minutes", "5")   # the job sets 60
dbutils.widgets.text("cycle_seconds", "15")

# COMMAND ----------

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import config, store
from squawk_lib.db import PgSession
from squawk_lib.detect import EPISODE_SCHEMA, detect_window, episodes_for_spark, stack_occupancy

spark.conf.set("spark.sql.session.timeZone", "UTC")
T = config.TABLES
LOOKBACK = config.DETECT["lookback_minutes"]
DURATION_S = int(dbutils.widgets.get("duration_minutes")) * 60
CYCLE_S = int(dbutils.widgets.get("cycle_seconds"))

pg = PgSession()
sent = {}   # event_id -> ended_at already pushed, so we only write what changed


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_positions(now):
    window_start = pd.Timestamp(now) - pd.Timedelta(minutes=LOOKBACK)
    df = spark.sql(f"""
        SELECT icao24, callsign, event_ts, lat, lon, alt_ft, vr_fpm, track_deg, on_ground
        FROM {T['positions']}
        WHERE event_ts > TIMESTAMP'{window_start:%Y-%m-%d %H:%M:%S}'""").toPandas()
    return df, window_start


def detect_all(positions, window_start):
    # Episodes starting in the first 30 min of the window are skipped (their start may be cut off).
    return detect_window(positions, window_start, margin_minutes=30)


def merge_gold(changed, now):
    sdf = spark.createDataFrame(episodes_for_spark(changed), schema=EPISODE_SCHEMA)
    sdf.createOrReplaceTempView("new_episodes")
    spark.sql(f"""
        MERGE INTO {T['disruption_episodes']} t
        USING (SELECT *, TIMESTAMP'{now:%Y-%m-%d %H:%M:%S}' AS detected_now FROM new_episodes) s
        ON t.event_id = s.event_id
        WHEN MATCHED AND s.ended_at > t.ended_at THEN UPDATE SET
            t.ended_at = s.ended_at, t.duration_s = s.duration_s, t.callsign = coalesce(s.callsign, t.callsign),
            t.min_alt_ft = least(t.min_alt_ft, s.min_alt_ft), t.max_alt_ft = greatest(t.max_alt_ft, s.max_alt_ft),
            t.max_turn_deg = greatest(t.max_turn_deg, s.max_turn_deg), t.n_points = s.n_points,
            t.updated_at = s.detected_now
        WHEN NOT MATCHED THEN INSERT
            (event_id, event_type, icao24, callsign, location, started_at, ended_at, duration_s, trigger_ts,
             min_alt_ft, max_alt_ft, max_turn_deg, n_points, first_detected_at, updated_at)
        VALUES
            (s.event_id, s.event_type, s.icao24, s.callsign, s.location, s.started_at, s.ended_at, s.duration_s,
             s.trigger_ts, s.min_alt_ft, s.max_alt_ft, s.max_turn_deg, s.n_points, s.detected_now, s.detected_now)""")


def merge_occupancy(episodes, now):
    occ = stack_occupancy(episodes, pd.Timestamp(now) - pd.Timedelta(minutes=30), now)
    occ["updated_at"] = now
    spark.createDataFrame(occ, schema="minute TIMESTAMP, stack STRING, holding_count INT, updated_at TIMESTAMP") \
        .createOrReplaceTempView("new_occupancy")
    spark.sql(f"""
        MERGE INTO {T['stack_occupancy_1min']} t USING new_occupancy s
        ON t.minute = s.minute AND t.stack = s.stack
        WHEN MATCHED THEN UPDATE SET t.holding_count = s.holding_count, t.updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT *""")

# COMMAND ----------

# Run ONE cycle by hand first, to check everything works before starting the loop.
positions, window_start = load_positions(utcnow())
episodes = detect_all(positions, window_start)
print(f"{len(positions)} positions from {positions['icao24'].nunique() if len(positions) else 0} aircraft; "
      f"{len(episodes)} episodes")
if len(episodes):
    display(episodes)
else:
    print("No episodes in this window - normal outside busy periods.")

# COMMAND ----------

started, cycle = time.time(), 0
while time.time() - started < DURATION_S:
    tick, cycle = time.time(), cycle + 1
    try:
        now = utcnow()
        positions, window_start = load_positions(now)
        episodes = detect_all(positions, window_start)

        changed = episodes[[sent.get(r.event_id) != r.ended_at for r in episodes.itertuples()]]
        if len(changed):
            merge_gold(changed, now)
            with pg() as conn:
                store.upsert_events(conn, changed)
            sent.update(dict(zip(changed["event_id"], changed["ended_at"])))

        merge_occupancy(episodes, now)

        if cycle % 20 == 0 or len(changed):
            holding = episodes[(episodes["event_type"] == "holding")
                               & (episodes["ended_at"] > pd.Timestamp(now) - pd.Timedelta(minutes=2))]
            print(f"{now:%H:%M:%S}Z cycle {cycle}: {len(positions)} positions, {len(episodes)} episodes in window, "
                  f"{len(changed)} new/extended, {len(holding)} aircraft holding now, "
                  f"cycle took {time.time() - tick:.1f}s")
    except Exception as e:
        print(f"cycle {cycle} failed: {type(e).__name__}: {e}")
        pg.reset()
    time.sleep(max(0.0, CYCLE_S - (time.time() - tick)))

pg.close()
