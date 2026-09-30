"""Clients for the third-party APIs, with retry, backoff and call logging.

    adsb.lol / adsb.fi /       aircraft positions, no auth, with failover between them
    airplanes.live                                                   (default)
    OpenSky Network REST API   aircraft positions, OAuth2 client credentials
    Aviation Weather Center    METAR + TAF for Heathrow, no auth

Both position sources emit the SAME record shape (OpenSky's field names, metres and
m/s), so everything downstream - Bronze, Silver, detection - is identical either way.

Note: OpenSky deliberately blocks hosting and cloud-provider IP ranges, so its calls
time out from Databricks compute. adsb.lol is the working source there.
"""
import json
import random
import time
import uuid
from datetime import datetime, timezone

import requests

from . import config

OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
ADSBLOL_POINT_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius}"
AWC_BASE = "https://aviationweather.gov/api/data"

FT_TO_M, KT_TO_MS, FPM_TO_MS = 0.3048, 0.514444, 1 / 196.8504
USER_AGENT = "squawk-capstone/1.0 (DataExpert.io bootcamp project)"

# Field order of an OpenSky state vector (index -> name).
STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact", "longitude", "latitude",
    "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate", "sensors", "geo_altitude",
    "squawk", "spi", "position_source", "category",
]


class ApiError(Exception):
    pass


def _backoff_sleep(attempt):
    time.sleep(2 ** attempt + random.uniform(0, 1))   # 2, 4, 8 s (+ jitter)


class OpenSkyClient:
    """Fetches state vectors for the Squawk bounding box.

    Caches the OAuth token and refreshes it 5 minutes before its 30-minute expiry,
    or immediately if the API says 401.
    """

    def __init__(self, client_id, client_secret, session=None, log=None):
        self.client_id, self.client_secret = client_id, client_secret
        self.http = session or requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        self._token, self._token_expiry = None, 0.0
        self.log = log if log is not None else []     # list of dicts -> bronze.api_call_log
        self.credits_remaining = None

    def _get_token(self, force=False):
        if not force and self._token and time.time() < self._token_expiry - 300:
            return self._token
        r = self.http.post(OPENSKY_TOKEN_URL, timeout=10, data={
            "grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret,
        })
        if r.status_code != 200:
            raise ApiError(f"OpenSky token request failed: HTTP {r.status_code} {r.text[:200]}")
        body = r.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + int(body.get("expires_in", 1800))
        return self._token

    def fetch_states(self, bbox=None, max_attempts=3):
        """Return (api_time, list_of_state_arrays), or None if this cycle should be skipped."""
        bbox = bbox or config.BBOX
        refreshed = False
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            status, err = None, None
            try:
                r = self.http.get(OPENSKY_STATES_URL, params=bbox, timeout=10,
                                  headers={"Authorization": f"Bearer {self._get_token()}"})
                status = r.status_code
                remaining = r.headers.get("X-Rate-Limit-Remaining")
                if remaining is not None:
                    self.credits_remaining = int(remaining)
            except (requests.Timeout, requests.ConnectionError, ApiError) as e:
                err = f"{type(e).__name__}: {e}"[:200]
            self._log("opensky", status, started, err)

            if status == 200:
                body = r.json()
                return body.get("time"), body.get("states") or []
            if status == 401 and not refreshed:          # token expired early: refresh once and retry
                self._get_token(force=True)
                refreshed = True
                continue
            if status == 429:                             # out of credits: wait as instructed, skip this cycle
                wait = int(r.headers.get("X-Rate-Limit-Retry-After-Seconds", "60"))
                print(f"OpenSky rate limit hit; waiting {min(wait, 900)} s")
                time.sleep(min(wait, 900))
                return None
            if status is not None and status < 500 and status != 401:
                raise ApiError(f"OpenSky HTTP {status}: {r.text[:200]}")
            if attempt < max_attempts:                    # 5xx or timeout: back off and retry
                _backoff_sleep(attempt)
        return None

    def _log(self, source, status, started, err):
        self.log.append({
            "source": source, "called_at": datetime.fromtimestamp(started, timezone.utc).replace(tzinfo=None),
            "http_status": status, "latency_ms": int((time.time() - started) * 1000),
            "credits_remaining": self.credits_remaining, "error": err,
        })


    def fetch_records(self, fetched_at, poll_id=None):
        """Positions in the shared record shape, or [] if this cycle should be skipped."""
        result = self.fetch_states()
        if result is None:
            return []
        api_time, states = result
        return state_rows(api_time, states, fetched_at, poll_id)


class AdsbClient:
    """Aircraft within a radius of a point, from the community ADS-B networks.

    All of them serve the same readsb JSON, so we can fail over between them: when one
    rate-limits us (429) it goes on a short cooldown and the next provider is tried.
    No key, no credit budget. Data is contributed by volunteer receiver operators and
    published under the Open Database Licence - credit the network wherever you show it.
    """

    def __init__(self, session=None, log=None, center=None, radius_nm=None, providers=None):
        self.http = session or requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        self.center = center or config.ADSB_CENTER
        self.radius_nm = radius_nm or config.ADSB_RADIUS_NM
        self.providers = providers or config.ADSB_PROVIDERS
        self.cooldown = {}                     # provider name -> time it can be used again
        self.log = log if log is not None else []
        self.credits_remaining = None          # these networks have no credit budget

    def _available(self):
        now = time.time()
        ready = [p for p in self.providers if self.cooldown.get(p["name"], 0) <= now]
        return ready or self.providers        # all cooling down: try anyway rather than skip

    def fetch_records(self, fetched_at, poll_id=None):
        for provider in self._available():
            url = provider["url"].format(lat=self.center["lat"], lon=self.center["lon"],
                                         radius=self.radius_nm)
            started, status, err, body = time.time(), None, None, None
            try:
                r = self.http.get(url, timeout=20)
                status = r.status_code
                if status == 200:
                    body = r.json()
            except (requests.Timeout, requests.ConnectionError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"[:200]
            self.log.append({
                "source": provider["name"],
                "called_at": datetime.fromtimestamp(started, timezone.utc).replace(tzinfo=None),
                "http_status": status, "latency_ms": int((time.time() - started) * 1000),
                "credits_remaining": None, "error": err,
            })
            if body is not None:
                return readsb_rows(body, fetched_at, poll_id)
            # Rate-limited or broken: rest this provider and try the next one.
            self.cooldown[provider["name"]] = time.time() + config.ADSB_COOLDOWN_S
        return []


AdsbLolClient = AdsbClient          # old name, kept so existing notebooks keep working


def _num(value):
    """adsb.lol uses the string "ground" where an altitude would be."""
    return value if isinstance(value, (int, float)) else None


def readsb_rows(body, fetched_at, poll_id=None):
    """Map a readsb-format response (adsb.lol, adsb.fi, airplanes.live) onto the
    OpenSky-style record shape (metres, m/s). The aircraft array is "ac" or "aircraft"."""
    poll_id = poll_id or uuid.uuid4().hex
    now_s = (body.get("now") or int(time.time() * 1000)) / 1000.0
    rows = []
    for a in body.get("ac") or body.get("aircraft") or []:
        if a.get("lat") is None or a.get("lon") is None:
            continue
        alt_baro, alt_geom = _num(a.get("alt_baro")), _num(a.get("alt_geom"))
        vr_fpm = a.get("baro_rate") if a.get("baro_rate") is not None else a.get("geom_rate")
        callsign = (a.get("flight") or "").strip() or None
        rows.append({
            "icao24": (a.get("hex") or "").strip().lower(),
            "callsign": callsign,
            "origin_country": None,                     # not provided by adsb.lol
            "time_position": int(now_s - (a.get("seen_pos") or 0)),
            "last_contact": int(now_s - (a.get("seen") or 0)),
            "longitude": a.get("lon"),
            "latitude": a.get("lat"),
            "baro_altitude": None if alt_baro is None else alt_baro * FT_TO_M,
            "on_ground": a.get("alt_baro") == "ground",
            "velocity": None if a.get("gs") is None else a["gs"] * KT_TO_MS,
            "true_track": a.get("track"),
            "vertical_rate": None if vr_fpm is None else vr_fpm * FPM_TO_MS,
            "geo_altitude": None if alt_geom is None else alt_geom * FT_TO_M,
            "squawk": a.get("squawk"),
            "spi": bool(a.get("spi")),
            "position_source": None,
            "category": None,
            "poll_id": poll_id, "fetched_at": fetched_at, "api_time": int(now_s),
        })
    return rows


adsblol_rows = readsb_rows          # old name, kept for compatibility


def make_client(source=None, client_id=None, client_secret=None, log=None):
    """Build the position client named by config.DATA_SOURCE (or `source`)."""
    source = (source or config.DATA_SOURCE).lower()
    if source in ("adsb.lol", "adsblol", "adsb"):
        return AdsbClient(log=log)
    if source == "opensky":
        if not (client_id and client_secret):
            raise ValueError("OpenSky needs client_id and client_secret")
        return OpenSkyClient(client_id, client_secret, log=log)
    raise ValueError(f"Unknown DATA_SOURCE: {source}")


def state_rows(api_time, states, fetched_at, poll_id=None):
    """Turn OpenSky state arrays into flat dicts (one per aircraft) for the landing JSON file."""
    poll_id = poll_id or uuid.uuid4().hex
    rows = []
    for s in states:
        rec = {name: (s[i] if i < len(s) else None) for i, name in enumerate(STATE_FIELDS)}
        rec.pop("sensors", None)
        if isinstance(rec.get("callsign"), str):
            rec["callsign"] = rec["callsign"].strip()
        rec.update({"poll_id": poll_id, "fetched_at": fetched_at, "api_time": api_time})
        rows.append(rec)
    return rows


def fetch_weather(session=None, log=None, station=config.HEATHROW_ICAO, max_attempts=3):
    """Return a list of {kind, fetched_at, obs_time, raw_json} rows for recent METARs and the latest TAF."""
    http = session or requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    log = log if log is not None else []
    rows, fetched_at = [], time.time()
    for kind, url, params in (
        ("metar", f"{AWC_BASE}/metar", {"ids": station, "format": "json", "hours": 2}),
        ("taf", f"{AWC_BASE}/taf", {"ids": station, "format": "json"}),
    ):
        data = None
        for attempt in range(1, max_attempts + 1):
            started, status, err = time.time(), None, None
            try:
                r = http.get(url, params=params, timeout=10)
                status = r.status_code
            except (requests.Timeout, requests.ConnectionError) as e:
                err = type(e).__name__
            log.append({"source": f"awc_{kind}", "called_at": datetime.fromtimestamp(started, timezone.utc).replace(tzinfo=None),
                        "http_status": status, "latency_ms": int((time.time() - started) * 1000),
                        "credits_remaining": None, "error": err})
            if status == 200:
                data = r.json()
                break
            if status == 204:        # valid request, no data
                data = []
                break
            if attempt < max_attempts:
                _backoff_sleep(attempt)
        if not data:
            continue
        for item in (data if isinstance(data, list) else [data]):
            if kind == "metar":
                obs = item.get("obsTime")
            else:
                issued = item.get("issueTime")
                obs = int(datetime.fromisoformat(issued.replace("Z", "+00:00")).timestamp()) if issued else None
            rows.append({"kind": kind, "fetched_at": fetched_at, "obs_time": obs, "raw_json": json.dumps(item)})
    return rows
