"""Rule-based detection of holding episodes and go-arounds.

Everything here is plain pandas + numpy so it can be unit-tested on a laptop
(see tests/test_detect.py). The detector notebook (03_detector) reads recent
Silver positions with Spark and calls `detect_episodes` once per aircraft.

Input columns (one row per position report, one aircraft per call):
    icao24, callsign, event_ts (UTC, naive or tz-aware), lat, lon,
    alt_ft, vr_fpm, track_deg, on_ground
"""
import hashlib
import uuid

import numpy as np
import pandas as pd

from .config import DETECT, RUNWAYS, STACKS

EARTH_RADIUS_NM = 3440.065

# Spark schema for the episodes DataFrame (column order = EPISODE_COLUMNS).
EPISODE_SCHEMA = (
    "event_id string, event_type string, icao24 string, callsign string, location string, "
    "started_at timestamp, ended_at timestamp, duration_s long, trigger_ts timestamp, "
    "min_alt_ft double, max_alt_ft double, max_turn_deg double, n_points long"
)
EPISODE_COLUMNS = [c.split()[0] for c in EPISODE_SCHEMA.split(", ")]


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def event_id_for(icao24, event_type, started_at):
    """Deterministic UUID, so re-detecting the same episode never creates a duplicate."""
    key = f"{icao24}|{event_type}|{pd.Timestamp(started_at):%Y-%m-%dT%H:%M:%S}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def _empty_episodes():
    return pd.DataFrame({c: pd.Series(dtype="object") for c in EPISODE_COLUMNS})


def add_features(df, cfg=DETECT):
    """Sort one aircraft's positions and add distance and turn features."""
    df = df.sort_values("event_ts").drop_duplicates("event_ts").reset_index(drop=True)
    ts = pd.to_datetime(df["event_ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["event_ts"] = ts
    df["on_ground"] = df["on_ground"].fillna(False).astype(bool)

    # Distance to each stack fix and the nearest one.
    stack_d = np.column_stack([
        haversine_nm(df["lat"].values, df["lon"].values, s["lat"], s["lon"]) for s in STACKS.values()
    ])
    stack_ids = list(STACKS.keys())
    df["nearest_stack"] = [stack_ids[i] for i in stack_d.argmin(axis=1)]
    df["dist_stack_nm"] = stack_d.min(axis=1)

    rwy_d = np.column_stack([
        haversine_nm(df["lat"].values, df["lon"].values, r["lat"], r["lon"]) for r in RUNWAYS.values()
    ])
    rwy_ids = list(RUNWAYS.keys())
    df["nearest_rwy"] = [rwy_ids[i] for i in rwy_d.argmin(axis=1)]
    df["dist_rwy_nm"] = rwy_d.min(axis=1)

    # Heading change between consecutive reports, wrapped to [-180, 180).
    gap_s = df["event_ts"].diff().dt.total_seconds()
    dtrack = (df["track_deg"].diff() + 540.0) % 360.0 - 180.0
    # Ignore turns across data gaps (we can't know what happened in between).
    dtrack = dtrack.where(gap_s <= 120, 0.0).fillna(0.0)
    df["dtrack"] = dtrack
    df["gap_s"] = gap_s.fillna(0.0)

    # Signed turn accumulated over the rolling window (a holding pattern turns
    # consistently one way, so its signed total keeps growing).
    rolled = pd.Series(dtrack.values, index=df["event_ts"]).rolling(cfg["hold_turn_window"]).sum()
    df["cum_turn_deg"] = rolled.values

    # Vertical rate: use the reported value, else derive it from altitude change.
    derived_vr = df["alt_ft"].diff() / (gap_s / 60.0)
    df["vr_fpm"] = df["vr_fpm"].fillna(derived_vr)
    return df


def _holding_episodes(df, cfg):
    in_zone = (
        (df["dist_stack_nm"] <= cfg["stack_radius_nm"])
        & (df["alt_ft"] >= cfg["stack_min_alt_ft"])
        & (~df["on_ground"])
    )
    prev_in = in_zone.shift(fill_value=False)
    new_visit = in_zone & ((~prev_in) | (df["gap_s"] > cfg["visit_gap_s"]))
    visit_id = new_visit.cumsum().where(in_zone)

    rows = []
    for _, v in df[in_zone].groupby(visit_id[in_zone]):
        turned = v[v["cum_turn_deg"].abs() >= cfg["hold_turn_deg"]]
        if turned.empty:
            continue  # passed through the stack area without holding
        started, ended = v["event_ts"].iloc[0], v["event_ts"].iloc[-1]
        rows.append({
            "event_type": "holding",
            "location": v["nearest_stack"].mode().iloc[0],
            "started_at": started,
            "ended_at": ended,
            "trigger_ts": turned["event_ts"].iloc[0],
            "min_alt_ft": float(v["alt_ft"].min()),
            "max_alt_ft": float(v["alt_ft"].max()),
            "max_turn_deg": float(v["cum_turn_deg"].abs().max()),
            "n_points": int(len(v)),
        })
    return rows


def _go_arounds(df, cfg):
    low = (
        (df["dist_rwy_nm"] <= cfg["ga_runway_radius_nm"])
        & (df["alt_ft"] < cfg["ga_low_alt_ft"])
        & (~df["on_ground"])
    )
    # First point of each run of consecutive low points.
    first_low = low & ~low.shift(fill_value=False)
    rows = []
    ts = df["event_ts"]
    for i in df.index[first_low]:
        t0, a0 = ts[i], df.at[i, "alt_ft"]

        # Must be an arrival: higher in the previous 3 minutes, and not just departed.
        before = df[(ts >= t0 - pd.Timedelta(seconds=180)) & (ts < t0)]
        if before.empty or before["on_ground"].any() or before["alt_ft"].max() < a0 + 300:
            continue

        after = df[(ts > t0) & (ts <= t0 + pd.Timedelta(seconds=cfg["ga_window_s"]))]
        if after.empty:
            continue
        # A landing shows up as on_ground; stop looking at the first ground report.
        if after["on_ground"].any():
            after = after.loc[: after.index[after["on_ground"].values][0]]
            after = after[~after["on_ground"]]
        climb = after[(after["alt_ft"] > cfg["ga_climb_alt_ft"]) & (after["vr_fpm"] > cfg["ga_climb_rate_fpm"])]
        if climb.empty:
            continue
        j = climb.index[0]
        seg = df.loc[i:j]
        rows.append({
            "event_type": "go_around",
            "location": df.at[i, "nearest_rwy"],
            "started_at": t0,
            "ended_at": ts[j],
            "trigger_ts": ts[j],
            "min_alt_ft": float(seg["alt_ft"].min()),
            "max_alt_ft": float(seg["alt_ft"].max()),
            "max_turn_deg": float(seg["dtrack"].abs().sum()),
            "n_points": int(len(seg)),
        })
    return rows


def detect_episodes(pdf, cfg=DETECT):
    """Detect holding episodes and go-arounds for ONE aircraft's positions."""
    pdf = pdf.dropna(subset=["event_ts", "lat", "lon", "alt_ft"])
    if len(pdf) < 3:
        return _empty_episodes()
    df = add_features(pdf.copy(), cfg)
    rows = _holding_episodes(df, cfg) + _go_arounds(df, cfg)
    if not rows:
        return _empty_episodes()

    out = pd.DataFrame(rows)
    icao24 = str(df["icao24"].iloc[0])
    callsigns = df["callsign"].dropna().astype(str).str.strip()
    out["icao24"] = icao24
    out["callsign"] = callsigns.iloc[-1] if len(callsigns) else None
    out["duration_s"] = (out["ended_at"] - out["started_at"]).dt.total_seconds().astype("int64")
    out["event_id"] = [event_id_for(icao24, r.event_type, r.started_at) for r in out.itertuples()]
    return out[EPISODE_COLUMNS]


def detect_window(positions, window_start, margin_minutes=30, cfg=DETECT):
    """Detect episodes for many aircraft read from a sliding window starting at window_start.

    Episodes that start within `margin_minutes` of the window's start are dropped: their real
    start may be before the window, so their start time (and therefore their ID) isn't stable
    yet. They were already emitted, with their true start, by earlier cycles.
    """
    if len(positions) == 0:
        return _empty_episodes()
    parts = [detect_episodes(g, cfg) for _, g in positions.groupby("icao24", sort=False)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty_episodes()
    eps = pd.concat(parts, ignore_index=True)
    cutoff = pd.Timestamp(window_start) + pd.Timedelta(minutes=margin_minutes)
    return eps[pd.to_datetime(eps["started_at"]) >= cutoff].reset_index(drop=True)


def episodes_for_spark(episodes):
    """Episodes with clean dtypes, ready for spark.createDataFrame(..., schema=EPISODE_SCHEMA)."""
    out = episodes[EPISODE_COLUMNS].copy()
    for c in ("started_at", "ended_at", "trigger_ts"):
        out[c] = pd.to_datetime(out[c])
    for c in ("duration_s", "n_points"):
        out[c] = out[c].astype("int64")
    for c in ("min_alt_ft", "max_alt_ft", "max_turn_deg"):
        out[c] = out[c].astype("float64")
    for c in ("event_id", "event_type", "icao24", "callsign", "location"):
        out[c] = out[c].astype(object).where(pd.notna(out[c]), None)
    return out.reset_index(drop=True)


def stack_occupancy(episodes, start, end):
    """Holding aircraft per stack per minute between start and end (inclusive)."""
    minutes = pd.date_range(pd.Timestamp(start).floor("min"), pd.Timestamp(end).floor("min"), freq="min")
    holds = episodes[episodes["event_type"] == "holding"]
    rows = []
    for m in minutes:
        active = holds[(holds["started_at"] <= m + pd.Timedelta(minutes=1)) & (holds["ended_at"] >= m)]
        counts = active.groupby("location")["icao24"].nunique()
        for stack in STACKS:
            rows.append({"minute": m, "stack": stack, "holding_count": int(counts.get(stack, 0))})
    return pd.DataFrame(rows, columns=["minute", "stack", "holding_count"])
