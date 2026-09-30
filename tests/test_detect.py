"""Unit tests for the detection rules, using synthetic flight tracks.

Run locally from the repo root:   python -m pytest tests -q
(or just: python tests/test_detect.py)
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from squawk_lib.config import RUNWAYS, STACKS  # noqa: E402
from squawk_lib.detect import detect_episodes, detect_window, event_id_for, stack_occupancy  # noqa: E402

T0 = pd.Timestamp("2026-09-21 07:00:00")
NM_PER_DEG_LAT = 60.0


def _move(lat, lon, track_deg, dist_nm):
    dlat = dist_nm * math.cos(math.radians(track_deg)) / NM_PER_DEG_LAT
    dlon = dist_nm * math.sin(math.radians(track_deg)) / (NM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def _track(icao24, legs, start_lat, start_lon, start_track, start_alt, t0=T0, step_s=30, gs_kt=220):
    """legs: list of (duration_s, turn_rate_deg_s, vr_fpm, on_ground)."""
    rows, lat, lon, trk, alt, t = [], start_lat, start_lon, start_track, start_alt, t0
    for dur, turn, vr, ground in legs:
        for _ in range(int(dur // step_s)):
            rows.append(dict(icao24=icao24, callsign="TEST1", event_ts=t, lat=lat, lon=lon,
                             alt_ft=alt, vr_fpm=vr, track_deg=trk % 360, on_ground=ground))
            trk += turn * step_s
            lat, lon = _move(lat, lon, trk, gs_kt * step_s / 3600)
            alt = max(0.0, alt + vr * step_s / 60)
            t += pd.Timedelta(seconds=step_s)
    return pd.DataFrame(rows)


def test_holding_detected_at_bovingdon():
    s = STACKS["BNN"]
    lap = [(60, 0, 0, False), (60, 3, 0, False), (60, 0, 0, False), (60, 3, 0, False)]
    df = _track("aaa001", [(120, 0, 0, False)] + lap * 3 + [(240, 0, -1500, False)],
                s["lat"], s["lon"] - 0.05, 90, 9000)
    ep = detect_episodes(df)
    holds = ep[ep.event_type == "holding"]
    assert len(holds) == 1, ep
    h = holds.iloc[0]
    assert h.location == "BNN"
    assert h.max_turn_deg >= 360
    assert h.duration_s >= 600
    assert h.trigger_ts > h.started_at


def test_straight_pass_through_stack_is_not_holding():
    s = STACKS["LAM"]
    df = _track("aaa002", [(600, 0, -800, False)], s["lat"], s["lon"] - 0.2, 90, 11000)
    assert detect_episodes(df).empty


def _approach(icao24, rwy, legs, dist_out_nm=10, alt=3300):
    r = RUNWAYS[rwy]
    # Start dist_out_nm east of a 27 threshold (or west of 09), flying towards it.
    track = 270 if rwy.startswith("27") else 90
    lat, lon = _move(r["lat"], r["lon"], (track + 180) % 360, dist_out_nm)
    return _track(icao24, legs, lat, lon, track, alt, gs_kt=150)


def test_go_around_detected():
    # Descend at 700 fpm for ~3.5 min to ~850 ft, then climb away at 1,800 fpm.
    df = _approach("aaa003", "27L", [(210, 0, -700, False), (150, 0, 1800, False), (60, 0, 0, False)])
    ep = detect_episodes(df)
    ga = ep[ep.event_type == "go_around"]
    assert len(ga) == 1, ep
    assert ga.iloc[0].location == "27L"
    assert ga.iloc[0].min_alt_ft < 1500


def test_normal_landing_is_not_go_around():
    df = _approach("aaa004", "27R", [(270, 0, -700, False), (180, 0, 0, True)])
    assert detect_episodes(df).empty


def test_departure_is_not_go_around():
    r = RUNWAYS["27L"]
    df = _track("aaa005", [(90, 0, 0, True), (240, 0, 2000, False)], r["lat"], r["lon"], 270, 0, gs_kt=160)
    assert detect_episodes(df).empty


def test_event_id_is_deterministic():
    a = event_id_for("abc123", "holding", pd.Timestamp("2026-09-21 07:00:00"))
    b = event_id_for("abc123", "holding", pd.Timestamp("2026-09-21 07:00:00"))
    c = event_id_for("abc123", "holding", pd.Timestamp("2026-09-21 07:00:30"))
    assert a == b and a != c and len(a) == 36


def test_redetection_gives_same_event():
    s = STACKS["OCK"]
    lap = [(60, 0, 0, False), (60, -3, 0, False), (60, 0, 0, False), (60, -3, 0, False)]
    full = _track("aaa006", [(90, 0, 0, False)] + lap * 3, s["lat"], s["lon"], 0, 8000)
    early = detect_episodes(full.iloc[: len(full) * 2 // 3])
    late = detect_episodes(full)
    assert early.event_id.tolist() == late.event_id.tolist()
    assert late.iloc[0].ended_at > early.iloc[0].ended_at


def test_sliding_window_never_duplicates_an_event():
    """Simulate the detector: every 15 s, read the last 180 min and detect. A 20-minute hold
    must produce exactly one event ID, even as its start slides out of the window."""
    s = STACKS["LAM"]
    lap = [(60, 0, 0, False), (60, 3, 0, False), (60, 0, 0, False), (60, 3, 0, False)]
    hold = _track("ccc001", [(120, 0, 0, False)] + lap * 5 + [(1200, 0, -1500, False)],
                  s["lat"], s["lon"] - 0.05, 90, 9000, t0=T0 + pd.Timedelta(minutes=60))
    ids = set()
    now = T0 + pd.Timedelta(minutes=60)
    while now < T0 + pd.Timedelta(hours=5):
        start = now - pd.Timedelta(minutes=180)
        window = hold[(hold.event_ts > start) & (hold.event_ts <= now)]
        ids |= set(detect_window(window, start)["event_id"])
        now += pd.Timedelta(minutes=2)
    assert len(ids) == 1, ids


def test_stack_occupancy_counts():
    s = STACKS["BIG"]
    lap = [(60, 0, 0, False), (60, 3, 0, False), (60, 0, 0, False), (60, 3, 0, False)]
    eps = pd.concat([
        detect_episodes(_track(f"bbb00{i}", [(60, 0, 0, False)] + lap * 3, s["lat"], s["lon"], 90, 9000 + 1000 * i))
        for i in range(3)
    ])
    occ = stack_occupancy(eps, T0 + pd.Timedelta(minutes=5), T0 + pd.Timedelta(minutes=6))
    assert occ[occ["stack"] == "BIG"].holding_count.max() == 3
    assert occ[occ["stack"] == "LAM"].holding_count.max() == 0


def test_timezone_aware_input_and_nulls():
    s = STACKS["BNN"]
    lap = [(60, 0, 0, False), (60, 3, 0, False), (60, 0, 0, False), (60, 3, 0, False)]
    df = _track("aaa007", lap * 3, s["lat"], s["lon"], 90, 9000)
    df["event_ts"] = df["event_ts"].dt.tz_localize("UTC")
    df["vr_fpm"] = df["vr_fpm"].astype(float)
    df["on_ground"] = df["on_ground"].astype(object)   # Spark booleans with nulls arrive like this
    df.loc[3, "vr_fpm"] = np.nan
    df.loc[5, "on_ground"] = None
    assert len(detect_episodes(df)) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
