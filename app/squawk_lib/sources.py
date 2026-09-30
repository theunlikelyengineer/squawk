"""Clients for the two third-party APIs, with retry, backoff and call logging.

    OpenSky Network REST API   aircraft positions (OAuth2 client credentials)
    Aviation Weather Center    METAR + TAF for Heathrow (no auth)
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
AWC_BASE = "https://aviationweather.gov/api/data"
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
