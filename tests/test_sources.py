"""Tests for the adsb.lol -> shared record shape mapping, using a real API response."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from squawk_lib.sources import adsblol_rows, state_rows  # noqa: E402

# Trimmed from a live api.adsb.lol/v2/point/51.4775/-0.4614/30 response.
SAMPLE = {
    "ac": [
        {"hex": "400ed2", "flight": "NCKL13  ", "alt_baro": 3575, "alt_geom": 3775, "gs": 38.3,
         "track": 15.12, "geom_rate": 6656, "squawk": "3646", "lat": 51.590149, "lon": -1.223268,
         "seen_pos": 0.232, "seen": 0.1, "spi": 0, "category": "A1"},
        {"hex": "44B4A4", "flight": "BNJ1364 ", "alt_baro": 22000, "alt_geom": 22700, "gs": 470.4,
         "track": 337.63, "baro_rate": -64, "squawk": "7475", "lat": 51.57724, "lon": -1.1091,
         "seen_pos": 0.265, "seen": 0.0, "spi": 0},
        {"hex": "abc123", "flight": "GND1    ", "alt_baro": "ground", "gs": 12.0, "track": 90.0,
         "lat": 51.47, "lon": -0.45, "seen_pos": 1.0, "seen": 1.0},
        {"hex": "nopos1", "flight": "NOPOS   ", "alt_baro": 10000},          # no lat/lon: dropped
    ],
    "now": 1790778428500,
    "total": 4,
}


def test_maps_to_the_opensky_record_shape():
    rows = adsblol_rows(SAMPLE, fetched_at=1790778429.0, poll_id="p1")
    assert len(rows) == 3, "rows without a position must be dropped"
    assert set(rows[0]) == set(state_rows(1, [["a"] * 18], 1.0, "p1")[0]), "field names must match OpenSky's"

    a = rows[0]
    assert a["icao24"] == "400ed2"
    assert a["callsign"] == "NCKL13"
    assert round(a["baro_altitude"]) == 1090          # 3575 ft -> m
    assert round(a["geo_altitude"]) == 1151           # 3775 ft -> m
    assert round(a["velocity"], 1) == 19.7            # 38.3 kt -> m/s
    assert round(a["vertical_rate"], 1) == 33.8       # 6656 fpm -> m/s (geom_rate used when baro_rate is absent)
    assert a["true_track"] == 15.12
    assert a["on_ground"] is False
    assert a["time_position"] == 1790778428           # now - seen_pos
    assert a["poll_id"] == "p1" and a["fetched_at"] == 1790778429.0

    b = rows[1]
    assert b["icao24"] == "44b4a4", "icao24 must be lower case"
    assert round(b["vertical_rate"], 2) == -0.33      # baro_rate preferred

    g = rows[2]
    assert g["on_ground"] is True and g["baro_altitude"] is None


def test_survives_a_sparse_response():
    assert adsblol_rows({}, fetched_at=1.0) == []
    assert adsblol_rows({"ac": [], "now": 1000}, fetched_at=1.0) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
