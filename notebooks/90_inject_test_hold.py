# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 90 · Inject a synthetic holding aircraft (integration fixture)
# MAGIC
# MAGIC Heathrow's arrival management means classic stack holding is now a bad-weather and peak
# MAGIC phenomenon rather than a daily one, so the live detector can go a day or more without an
# MAGIC event. This notebook exercises the whole path end to end in the meantime.
# MAGIC
# MAGIC It writes landing files in exactly the format `01_poller` writes, for one aircraft flying a
# MAGIC textbook holding pattern over a stack fix. Everything downstream - Auto Loader, Bronze,
# MAGIC Silver, the detector, Lakebase, the agent, the app - then runs completely unmodified.
# MAGIC **Only the sensor input is synthetic.**
# MAGIC
# MAGIC The aircraft uses an ICAO24 address in the reserved `ffff00`-`ffffff` range, which is never
# MAGIC issued to a real airframe, so every synthetic row is identifiable and separable forever:
# MAGIC
# MAGIC ```sql
# MAGIC WHERE icao24 NOT LIKE 'ffff%'     -- real traffic only
# MAGIC ```
# MAGIC
# MAGIC **Run this by hand. Never add it to a job.** It writes in real time for `duration_minutes`,
# MAGIC because Silver deduplicates under a 2-minute watermark and would drop back-dated rows.
# MAGIC
# MAGIC Disclose it in your write-up. Suggested wording is in the last cell.

# COMMAND ----------

dbutils.widgets.text("duration_minutes", "16")      # 16 min at 220 kt = about 3.7 orbits
dbutils.widgets.text("interval_seconds", "20")      # match the poller's measured ~21.7 s cadence
dbutils.widgets.text("stack", "BNN")                # BNN, BIG, LAM or OCK
dbutils.widgets.text("altitude_ft", "9000")         # stack holding is 8,000-16,000 ft
dbutils.widgets.text("speed_kt", "220")             # typical holding speed
dbutils.widgets.text("radius_nm", "2.5")            # gives a ~4.3 min orbit, like a real hold
dbutils.widgets.text("icao24", "ffff01")            # reserved range: never a real aircraft
dbutils.widgets.text("callsign", "SQWK01")

# COMMAND ----------

import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import config

DURATION_S = int(dbutils.widgets.get("duration_minutes")) * 60
INTERVAL_S = float(dbutils.widgets.get("interval_seconds"))
STACK = dbutils.widgets.get("stack").strip().upper()
ALT_FT = float(dbutils.widgets.get("altitude_ft"))
SPEED_KT = float(dbutils.widgets.get("speed_kt"))
RADIUS_NM = float(dbutils.widgets.get("radius_nm"))
ICAO24 = dbutils.widgets.get("icao24").strip().lower()
CALLSIGN = dbutils.widgets.get("callsign").strip().upper()

assert STACK in config.STACKS, f"stack must be one of {list(config.STACKS)}"
assert ICAO24.startswith("ffff"), "use a reserved ffff.. address so synthetic rows stay identifiable"
assert ALT_FT >= config.DETECT["stack_min_alt_ft"], "below stack_min_alt_ft: the detector would ignore it"
assert RADIUS_NM < config.DETECT["stack_radius_nm"], "outside stack_radius_nm: the detector would ignore it"

FIX = config.STACKS[STACK]
FT_TO_M, KT_TO_MS = 0.3048, 0.514444

# Time for one full orbit: circumference / speed.
ORBIT_S = 3600.0 * (2 * math.pi * RADIUS_NM) / SPEED_KT

print(f"Synthetic hold over {STACK} ({FIX['name']}) at {ALT_FT:,.0f} ft, {SPEED_KT:.0f} kt, "
      f"{RADIUS_NM} NM radius")
print(f"  one orbit = {ORBIT_S:.0f} s ({ORBIT_S / 60:.1f} min), "
      f"{DURATION_S / ORBIT_S:.1f} orbits over {DURATION_S / 60:.0f} minutes")
print(f"  turn accumulated in the detector's {config.DETECT['hold_turn_window']} window = "
      f"{360 * (8 * 60) / ORBIT_S:.0f} deg (threshold is {config.DETECT['hold_turn_deg']:.0f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Position model
# MAGIC A right-hand circular orbit around the fix. Heading is the tangent, so it rotates a full 360
# MAGIC degrees per orbit in one consistent direction - which is exactly what `cum_turn_deg` measures.

# COMMAND ----------

def position_at(elapsed_s):
    """(lat, lon, track_deg) for a right-hand orbit, `elapsed_s` into the hold."""
    theta = 2 * math.pi * (elapsed_s / ORBIT_S)          # bearing from the fix, clockwise
    lat = FIX["lat"] + (RADIUS_NM / 60.0) * math.cos(theta)
    lon = FIX["lon"] + (RADIUS_NM / 60.0) * math.sin(theta) / math.cos(math.radians(FIX["lat"]))
    track = (math.degrees(theta) + 90.0) % 360.0         # tangent to the circle, turning right
    return lat, lon, track


def state_row(now_s, elapsed_s, poll_id):
    """One record in exactly the shape 01_poller lands (OpenSky field names, metres and m/s)."""
    lat, lon, track = position_at(elapsed_s)
    return {
        "poll_id": poll_id,
        "fetched_at": now_s,
        "api_time": int(now_s),
        "icao24": ICAO24,
        "callsign": CALLSIGN,
        "origin_country": None,
        "time_position": int(now_s),      # fetched_at - time_position must be <= 60 in Silver
        "last_contact": int(now_s),
        "longitude": lon,
        "latitude": lat,
        "baro_altitude": ALT_FT * FT_TO_M,
        "on_ground": False,
        "velocity": SPEED_KT * KT_TO_MS,
        "true_track": track,
        "vertical_rate": 0.0,
        "geo_altitude": ALT_FT * FT_TO_M,
        "squawk": None,
        "spi": False,
        "position_source": None,
        "category": None,
    }


TMP = f"{config.VOLUME}/_tmp"
os.makedirs(TMP, exist_ok=True)


def land(rows):
    """Write one JSON-lines file, via _tmp, so Auto Loader never sees a partial file."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    final_dir = f"{config.LANDING_OPENSKY}/{day}"
    os.makedirs(final_dir, exist_ok=True)
    name = f"states_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.json"
    tmp_path, final_path = f"{TMP}/{name}", f"{final_dir}/{name}"
    with open(tmp_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        with open(tmp_path) as src, open(final_path, "w") as dst:
            dst.write(src.read())
        os.remove(tmp_path)
    return final_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fly it
# MAGIC Writes one file per interval, in real time, until the duration is up.

# COMMAND ----------

started = time.time()
written = 0

while time.time() - started < DURATION_S:
    tick = time.time()
    elapsed = tick - started
    land([state_row(tick, elapsed, uuid.uuid4().hex)])
    written += 1
    if written % 10 == 0 or written == 1:
        lat, lon, track = position_at(elapsed)
        print(f"{datetime.now(timezone.utc):%H:%M:%S}Z  point {written:>3}  "
              f"{elapsed / ORBIT_S:.2f} orbits  track {track:5.1f} deg  "
              f"lat {lat:.4f} lon {lon:.4f}")
    time.sleep(max(0.0, INTERVAL_S - (time.time() - tick)))

print(f"\nDone: {written} points over {(time.time() - started) / 60:.1f} minutes "
      f"({(time.time() - started) / ORBIT_S:.1f} orbits).")
print("Silver should have them within a minute or two; the detector picks them up on its next cycle.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Did it land?
# MAGIC Run a minute or two after the loop finishes.

# COMMAND ----------

spark.sql(f"""
    SELECT count(*) AS points, min(event_ts) AS first_seen, max(event_ts) AS last_seen,
           round(min(alt_ft)) AS min_alt_ft, round(max(alt_ft)) AS max_alt_ft,
           round(avg(speed_kt)) AS avg_speed_kt
    FROM {config.TABLES['positions']}
    WHERE icao24 = '{ICAO24}'
""").display()

spark.sql(f"""
    SELECT event_id, event_type, location, callsign, started_at, ended_at,
           round(duration_s / 60.0, 1) AS minutes, round(max_turn_deg) AS turn_deg, n_points
    FROM {config.TABLES['disruption_episodes']}
    WHERE icao24 = '{ICAO24}' ORDER BY started_at DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Removing the fixture later
# MAGIC `silver.positions` is a streaming table owned by the pipeline, so its rows cannot be deleted
# MAGIC from outside it - exclude them with `icao24 NOT LIKE 'ffff%'` instead. Gold and Lakebase are
# MAGIC ordinary tables and can be cleaned if you want to:
# MAGIC
# MAGIC ```sql
# MAGIC DELETE FROM <catalog>.<schema>.disruption_episodes WHERE icao24 LIKE 'ffff%';
# MAGIC ```
# MAGIC ```sql
# MAGIC -- in Lakebase, assessments and reviews reference the event, so they go first
# MAGIC DELETE FROM squawk.analyst_reviews    WHERE event_id IN (SELECT event_id FROM squawk.disruption_events WHERE icao24 LIKE 'ffff%');
# MAGIC DELETE FROM squawk.agent_assessments  WHERE event_id IN (SELECT event_id FROM squawk.disruption_events WHERE icao24 LIKE 'ffff%');
# MAGIC DELETE FROM squawk.disruption_events  WHERE icao24 LIKE 'ffff%';
# MAGIC ```
# MAGIC
# MAGIC ### For the write-up
# MAGIC > Classic stack holding at Heathrow has become infrequent under time-based arrival management:
# MAGIC > a probe across 1,210 aircraft that entered a holding fix over a 12-hour period found a maximum
# MAGIC > accumulated turn of 235 degrees against a 360-degree detection threshold, confirming both that no
# MAGIC > holding occurred and that the threshold carries a clear margin over normal arrival geometry.
# MAGIC > To validate the detection, assessment and review path without waiting on weather, synthetic
# MAGIC > holding traffic was injected at the sensor boundary - landing files in the poller's own format,
# MAGIC > under a reserved ICAO24 address - so that every component downstream ran unmodified on it.
# MAGIC > Synthetic rows remain separable from real traffic by that address prefix.