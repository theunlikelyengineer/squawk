"""Squawk configuration shared by the notebooks, the detector, the agent and the app.

Edit the values in the "YOU EDIT THESE" block once, then every notebook and the
app pick them up. Values can also be overridden with environment variables of
the same name (the Databricks App sets some of them in app.yaml).
"""
import os


def _env(name, default):
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# YOU EDIT THESE
# ---------------------------------------------------------------------------

# Unity Catalog catalog that holds everything. If you are not allowed to create
# a catalog in the bootcamp workspace, set this to a catalog you already own.
CATALOG = _env("SQUAWK_CATALOG", "squawk")

# One schema per layer. If you can only use ONE schema, set all five to the same
# schema name: every table name below is unique across layers, so it still works.
SCHEMA_RAW = _env("SQUAWK_SCHEMA_RAW", "raw")              # holds the landing volume
SCHEMA_BRONZE = _env("SQUAWK_SCHEMA_BRONZE", "bronze")
SCHEMA_SILVER = _env("SQUAWK_SCHEMA_SILVER", "silver")
SCHEMA_GOLD = _env("SQUAWK_SCHEMA_GOLD", "gold")
SCHEMA_ANALYTICS = _env("SQUAWK_SCHEMA_ANALYTICS", "analytics")

# Databricks secret scope holding the API credentials (created in 00_setup).
SECRET_SCOPE = _env("SQUAWK_SECRET_SCOPE", "squawk")

# Lakebase endpoint, e.g. "projects/squawk/branches/production/endpoints/primary".
# 02_lakebase_setup prints the exact value for you to paste here.
LAKEBASE_ENDPOINT = _env("SQUAWK_LAKEBASE_ENDPOINT", "projects/squawk/branches/production/endpoints/primary")
LAKEBASE_DATABASE = _env("PGDATABASE", "databricks_postgres")
LAKEBASE_SCHEMA = _env("SQUAWK_LAKEBASE_SCHEMA", "squawk")   # Postgres schema for the 4 app tables

# LLM provider for the agent: "anthropic" or "openai" (whichever key the
# bootcamp onboarding page gave you). Model names are only defaults: change
# them to models your key can use.
LLM_PROVIDER = _env("SQUAWK_LLM_PROVIDER", "anthropic")
LLM_MODELS = {
    "anthropic": {"fast": "claude-haiku-4-5", "smart": "claude-sonnet-4-5"},
    "openai": {"fast": "gpt-4.1-mini", "smart": "gpt-4.1"},
}
# Which model assesses events: "fast" (cheaper) or "smart" (more reliable tool use).
ASSESS_MODEL = _env("SQUAWK_ASSESS_MODEL", "fast")
PROMPT_VERSION = "prompt-v1"

# ---------------------------------------------------------------------------
# Derived names (you shouldn't need to edit below this line)
# ---------------------------------------------------------------------------

VOLUME = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/landing"
LANDING_OPENSKY = f"{VOLUME}/opensky"
LANDING_WEATHER = f"{VOLUME}/weather"
LANDING_REFERENCE = f"{VOLUME}/reference"


def t(schema, name):
    """Fully qualified Unity Catalog table name."""
    return f"{CATALOG}.{schema}.{name}"


TABLES = {
    "api_call_log": t(SCHEMA_BRONZE, "api_call_log"),
    "opensky_states": t(SCHEMA_BRONZE, "opensky_states"),
    "weather_raw": t(SCHEMA_BRONZE, "weather_raw"),
    "aircraft_ref": t(SCHEMA_SILVER, "aircraft_ref"),
    "positions": t(SCHEMA_SILVER, "positions"),
    "weather": t(SCHEMA_SILVER, "weather"),
    "disruption_episodes": t(SCHEMA_GOLD, "disruption_episodes"),
    "stack_occupancy_1min": t(SCHEMA_GOLD, "stack_occupancy_1min"),
    "holding_hourly": t(SCHEMA_GOLD, "holding_hourly"),
}

# ---------------------------------------------------------------------------
# Airspace reference data
# ---------------------------------------------------------------------------

# OpenSky bounding box: about 1.8 square degrees, so 1 API credit per call.
BBOX = {"lamin": 51.0, "lomin": -1.2, "lamax": 52.0, "lomax": 0.6}

POLL_SECONDS = int(_env("SQUAWK_POLL_SECONDS", "30"))       # 30 s = 2,880 credits/day if run 24/7
WEATHER_POLL_SECONDS = 600

# Heathrow holding-stack fixes (VOR positions, from OurAirports / UK AIP).
STACKS = {
    "BNN": {"name": "Bovingdon", "lat": 51.726101, "lon": -0.549722},
    "BIG": {"name": "Biggin Hill", "lat": 51.330898, "lon": 0.034811},
    "LAM": {"name": "Lambourne", "lat": 51.646099, "lon": 0.151667},
    "OCK": {"name": "Ockham", "lat": 51.305000, "lon": -0.447222},
}

# Heathrow runway thresholds (approximate; good to ~100 m, which is plenty for a 6 NM rule).
RUNWAYS = {
    "09L": {"lat": 51.4775, "lon": -0.4850},
    "27R": {"lat": 51.4776, "lon": -0.4332},
    "09R": {"lat": 51.4647, "lon": -0.4823},
    "27L": {"lat": 51.4649, "lon": -0.4341},
}

HEATHROW_ICAO = "EGLL"

# ---------------------------------------------------------------------------
# Detection thresholds (starting values: tune them on your first day of data)
# ---------------------------------------------------------------------------
DETECT = {
    "stack_radius_nm": 12.0,
    "stack_min_alt_ft": 7000.0,
    "hold_turn_deg": 360.0,
    "hold_turn_window": "8min",
    "visit_gap_s": 180,          # a gap longer than this ends a stack visit
    "ga_runway_radius_nm": 6.0,
    "ga_low_alt_ft": 1500.0,
    "ga_climb_alt_ft": 2500.0,
    "ga_climb_rate_fpm": 1000.0,
    "ga_window_s": 300,
    "lookback_minutes": 180,     # how much Silver history the detector reads each cycle
}

CAUSES = ["wind", "low_visibility_cloud", "runway_change", "traffic_volume", "other"]
