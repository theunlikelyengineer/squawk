# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Threshold probe (diagnostic, not part of the job)
# MAGIC
# MAGIC Answers one question: of the aircraft that entered a holding-stack zone, how far did each one
# MAGIC actually turn? That separates "no holding is happening" from "the rules are mis-tuned".
# MAGIC
# MAGIC Run this by hand when you want to tune `DETECT["hold_turn_deg"]` or `stack_radius_nm`.
# MAGIC Never add it to a job.

# COMMAND ----------

dbutils.widgets.text("lookback_hours", "3")

# COMMAND ----------

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import config
from squawk_lib.config import DETECT
from squawk_lib.detect import add_features, detect_window, haversine_nm

spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load positions
# MAGIC Same query the detector uses, but with the window under your control.

# COMMAND ----------

LOOKBACK_HOURS = float(dbutils.widgets.get("lookback_hours"))
now = datetime.now(timezone.utc).replace(tzinfo=None)
window_start = pd.Timestamp(now) - pd.Timedelta(hours=LOOKBACK_HOURS)

positions = spark.sql(f"""
    SELECT icao24, callsign, event_ts, lat, lon, alt_ft, vr_fpm, track_deg, on_ground
    FROM {config.TABLES['positions']}
    WHERE event_ts > TIMESTAMP'{window_start:%Y-%m-%d %H:%M:%S}'
""").toPandas()

print(f"{len(positions):,} positions from {positions['icao24'].nunique():,} aircraft, "
      f"{window_start:%Y-%m-%d %H:%M}Z to {now:%H:%M}Z")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Which aircraft even entered a stack zone?
# MAGIC A vectorised pre-filter first, so we only run the per-aircraft feature builder on candidates.
# MAGIC Without this the loop takes minutes.

# COMMAND ----------

stack_dist = np.column_stack([
    haversine_nm(positions["lat"].values, positions["lon"].values, s["lat"], s["lon"])
    for s in config.STACKS.values()
])
positions["_dist_stack_nm"] = stack_dist.min(axis=1)

candidates = positions.loc[
    (positions["_dist_stack_nm"] <= DETECT["stack_radius_nm"])
    & (positions["alt_ft"] >= DETECT["stack_min_alt_ft"])
    & (~positions["on_ground"].fillna(False).astype(bool)),
    "icao24",
].unique()

print(f"{len(candidates):,} aircraft had at least one position inside a stack zone")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. How far did each one turn?
# MAGIC `cum_turn_deg` is the signed heading change accumulated over an 8-minute rolling window, so a
# MAGIC holding circuit (two 180 degree turns the same way, about 4 minutes) reaches 360 and keeps going.

# COMMAND ----------

probe = []
for icao, g in positions[positions["icao24"].isin(candidates)].groupby("icao24"):
    if len(g) < 5:
        continue
    f = add_features(g.drop(columns=["_dist_stack_nm"]))
    in_zone = (
        (f["dist_stack_nm"] <= DETECT["stack_radius_nm"])
        & (f["alt_ft"] >= DETECT["stack_min_alt_ft"])
        & (~f["on_ground"])
    )
    if not in_zone.any():
        continue
    z = f.loc[in_zone]
    probe.append({
        "icao24": icao,
        "callsign": f["callsign"].dropna().iloc[0] if f["callsign"].notna().any() else None,
        "stack": z["nearest_stack"].mode().iloc[0],
        "points_in_zone": int(len(z)),
        "minutes_in_zone": round((z["event_ts"].max() - z["event_ts"].min()).total_seconds() / 60, 1),
        "min_dist_nm": round(float(f["dist_stack_nm"].min()), 1),
        "min_alt_ft": int(z["alt_ft"].min()),
        "max_alt_ft": int(z["alt_ft"].max()),
        "max_turn_deg": round(float(z["cum_turn_deg"].abs().max()), 0),
        "max_gap_s": int(z["gap_s"].max()),
    })

probe = pd.DataFrame(probe).sort_values("max_turn_deg", ascending=False).reset_index(drop=True)
display(probe.head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The tuning answer
# MAGIC How many aircraft would be called "holding" at each candidate threshold. Look for a gap: if nothing
# MAGIC sits between 200 and 360 the current threshold is safe, and if there is a cluster just below 360
# MAGIC the sampling rate is clipping real holds and the threshold should come down.

# COMMAND ----------

buckets = [0, 90, 180, 270, 300, 360, 540, 720, 10_000]
labels = ["<90", "90-180", "180-270", "270-300", "300-360", "360-540", "540-720", "720+"]
counts = pd.cut(probe["max_turn_deg"], bins=buckets, labels=labels, right=False).value_counts().sort_index()
print(counts.to_string(), "\n")

for thr in (270, 300, 360, 540):
    hit = probe[probe["max_turn_deg"] >= thr]
    print(f"hold_turn_deg = {thr:>3}  ->  {len(hit):>3} aircraft, "
          f"median {hit['minutes_in_zone'].median() if len(hit) else float('nan'):.1f} min in zone")

print(f"\nconfig currently uses hold_turn_deg = {DETECT['hold_turn_deg']:.0f}, "
      f"stack_radius_nm = {DETECT['stack_radius_nm']:.0f}, "
      f"stack_min_alt_ft = {DETECT['stack_min_alt_ft']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. What the real rules found over the same window
# MAGIC For comparison. `detect_window` skips episodes starting in the first 30 minutes, because their
# MAGIC start may be cut off by the edge of the window.

# COMMAND ----------

episodes = detect_window(positions.drop(columns=["_dist_stack_nm"]), window_start, margin_minutes=30)
print(f"{len(episodes)} episodes "
      f"({(episodes['event_type'] == 'holding').sum() if len(episodes) else 0} holding, "
      f"{(episodes['event_type'] == 'go_around').sum() if len(episodes) else 0} go-around)")
display(episodes)
