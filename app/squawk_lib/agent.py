"""The Squawk agent: LangGraph tool-calling agent with read tools and two write tools.

Used in three places:
  * 04_agent_worker (job): assesses new events and issues hourly forecasts (read + write tools)
  * the Databricks App chat: read-only tools, so chat can never change data
  * notebooks, for testing a single event by hand

The data access is passed in, so the same tools work in a notebook (Spark) and
in the app (SQL warehouse):
    sql_fn(query) -> pandas.DataFrame     reads Delta tables
    pg_read       -> db.PgSession        reads Lakebase (`with pg_read() as conn:`)
    pg_agent      -> db.PgSession(role="squawk_agent")  writes Lakebase as the restricted role
"""
import json
import re
from datetime import datetime, timezone

import pandas as pd
from langchain_core.tools import tool

import config
from .db import pg_df
from .detect import haversine_nm

T = config.TABLES
ICAO24_RE = re.compile(r"^[0-9a-f]{6}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ASSESS_PROMPT = """You are Squawk, an assistant to an airline operations analyst at London Heathrow (EGLL).
You explain arrival disruption events: holding episodes (aircraft circling in the Bovingdon BNN,
Biggin Hill BIG, Lambourne LAM or Ockham OCK stacks) and go-arounds (aborted landings).

For the event you are given:
1. Use get_flight_track to look at what the aircraft did.
2. Use get_weather for the METAR reports around the event and the TAF.
3. Use get_stack_occupancy to see how busy all four stacks were.
4. Optionally use get_forecast_accuracy to see recent analyst corrections and learn from them.
5. Call save_assessment EXACTLY ONCE with your conclusion. Then reply with one short sentence.

Cause taxonomy (pick the single best one):
- wind: strong or gusty wind (gusts >= 25 kt, or mean wind >= 20 kt), or a strong crosswind.
- low_visibility_cloud: visibility below about 5 km (3 statute miles), cloud ceiling below about 1,000 ft,
  or flight category IFR/LIFR. Low-visibility procedures increase spacing between arrivals.
- runway_change: arrivals switching direction (e.g. from 27L/R to 09L/R) around the event time,
  often when the wind direction swings through north or south.
- traffic_volume: benign weather but several stacks busy at once (arrival demand above capacity).
- other: anything else, or not enough evidence.

Severity: 1 = minor (hold under 10 min, or a single go-around in good weather);
2 = moderate (hold 10-20 min, or several stacks busy); 3 = severe (hold over 20 min, or repeated go-arounds).
Confidence (0-1) must reflect the evidence: use <= 0.5 when weather data is missing or ambiguous.
Reasoning: 2-4 sentences an analyst can check, quoting the specific numbers you relied on.
Never invent data. All times are UTC."""

FORECAST_PROMPT = """You are Squawk. Forecast arrival holding at London Heathrow for ONE target hour (UTC).
1. Use get_stack_occupancy to see current holding and the last few hours of actual holding.
2. Use get_weather for the latest METAR and the TAF covering the target hour.
3. Use get_forecast_accuracy to see how accurate your recent forecasts were, and correct for any bias.
4. Call save_forecast EXACTLY ONCE with the predicted mean holding time per holding flight (minutes)
   and the predicted number of flights that will hold. Then reply with one short sentence.
Base the prediction on persistence (recent hours) adjusted for the forecast weather. Never invent data."""

CHAT_PROMPT = """You are Squawk, an assistant to an airline operations analyst at London Heathrow (EGLL).
Answer questions about arrival holding, go-arounds, stack occupancy, weather and forecasts using your
read-only tools. You cannot change any data. Be concise, quote the numbers you used, and say when
the data doesn't answer the question. All times are UTC."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_ts(value, name):
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise ValueError(f"{name} must be an ISO timestamp like 2026-09-21T07:30:00Z") from e
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _sql_ts(ts):
    return f"TIMESTAMP'{ts:%Y-%m-%d %H:%M:%S}'"


def _json(obj):
    return json.dumps(obj, default=str)


def _records(df, n=None):
    """DataFrame -> list of plain dicts (UUIDs, timestamps and decimals become strings, NaN becomes None)."""
    df = df if n is None else df.head(n)
    df = df.astype(object).where(pd.notna(df), None)
    return json.loads(json.dumps(df.to_dict(orient="records"), default=str))


def _trace_id():
    try:
        import mlflow
        return mlflow.get_active_trace_id()
    except Exception:
        return None


def model_version(kind):
    return f"{config.LLM_PROVIDER}:{config.LLM_MODELS[config.LLM_PROVIDER][kind]}|{config.PROMPT_VERSION}"


def build_llm(kind="fast"):
    """kind: "fast" for routine assessments, "smart" for forecasts and chat."""
    name = config.LLM_MODELS[config.LLM_PROVIDER][kind]
    kwargs = {}
    if getattr(config, "LLM_BASE_URL", None):
        kwargs["base_url"] = config.LLM_BASE_URL
    if getattr(config, "LLM_HEADERS", None):
        kwargs["default_headers"] = dict(config.LLM_HEADERS)
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=0, max_tokens=1500,
                             timeout=60, max_retries=2, **kwargs)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=name, temperature=0, timeout=60, max_retries=2, **kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def build_tools(sql_fn, pg_read, pg_agent=None, kind="fast"):
    """Return (read_tools, write_tools). write_tools is empty when pg_agent is None."""

    @tool(parse_docstring=True)
    def list_open_events(status: str = "open", hours: int = 6, limit: int = 20) -> str:
        """List recent disruption events with the latest agent assessment and analyst review.

        Args:
            status: "open" (detected or assessed), "detected", "assessed", "confirmed", "rejected" or "all".
            hours: How many hours back to look (1-48).
            limit: Maximum number of events (1-50).
        """
        hours, limit = max(1, min(int(hours), 48)), max(1, min(int(limit), 50))
        allowed = {"open": ("detected", "assessed"), "all": ("detected", "assessed", "confirmed", "rejected")}
        statuses = allowed.get(status, (status,))
        with pg_read() as conn:
            df = pg_df(conn, """
                SELECT e.event_id, e.event_type, e.icao24, e.callsign, e.location, e.started_at, e.ended_at,
                       e.duration_s, e.status, a.cause AS agent_cause, a.confidence, r.verdict, r.corrected_cause
                FROM disruption_events e
                LEFT JOIN LATERAL (SELECT cause, confidence FROM agent_assessments x
                                   WHERE x.event_id = e.event_id ORDER BY created_at DESC LIMIT 1) a ON true
                LEFT JOIN LATERAL (SELECT verdict, corrected_cause FROM analyst_reviews y
                                   WHERE y.event_id = e.event_id ORDER BY reviewed_at DESC LIMIT 1) r ON true
                WHERE e.status = ANY(%s) AND e.started_at > now() - make_interval(hours => %s)
                ORDER BY e.started_at DESC LIMIT %s""", (list(statuses), hours, limit))
        return _json(_records(df))

    @tool(parse_docstring=True)
    def get_flight_track(icao24: str, start_utc: str, end_utc: str) -> str:
        """Get an aircraft's track, one row per minute, with distance to the nearest stack and runway.

        Args:
            icao24: The 6-character hex transponder address, e.g. "4ca7b3".
            start_utc: Start time, ISO format (UTC). Use a few minutes before the event.
            end_utc: End time, ISO format (UTC). Maximum window is 90 minutes.
        """
        icao24 = icao24.strip().lower()
        if not ICAO24_RE.match(icao24):
            return "Error: icao24 must be 6 hex characters."
        start, end = _utc_ts(start_utc, "start_utc"), _utc_ts(end_utc, "end_utc")
        if end - start > pd.Timedelta(minutes=90):
            start = end - pd.Timedelta(minutes=90)
        df = sql_fn(f"""
            SELECT date_trunc('minute', event_ts) AS minute,
                   round(avg(lat), 4) AS lat, round(avg(lon), 4) AS lon,
                   round(avg(alt_ft)) AS alt_ft, round(avg(speed_kt)) AS speed_kt,
                   round(first(track_deg)) AS track_deg, round(avg(vr_fpm)) AS vr_fpm,
                   max(on_ground) AS on_ground
            FROM {T['positions']}
            WHERE icao24 = '{icao24}' AND event_ts BETWEEN {_sql_ts(start)} AND {_sql_ts(end)}
            GROUP BY 1 ORDER BY 1""")
        if df.empty:
            return "No positions found for that aircraft and time window."
        from .config import RUNWAYS, STACKS
        for key, pts in (("stack", STACKS), ("rwy", RUNWAYS)):
            d = pd.DataFrame({k: haversine_nm(df.lat, df.lon, v["lat"], v["lon"]) for k, v in pts.items()})
            df[f"nearest_{key}"] = d.idxmin(axis=1)
            df[f"dist_{key}_nm"] = d.min(axis=1).round(1)
        return _json(_records(df.drop(columns=["lat", "lon"]), 90))

    @tool(parse_docstring=True)
    def get_weather(start_utc: str, end_utc: str) -> str:
        """Get Heathrow METAR observations for a time window (plus the hour before) and the latest TAF.

        Args:
            start_utc: Start time, ISO format (UTC).
            end_utc: End time, ISO format (UTC).
        """
        start, end = _utc_ts(start_utc, "start_utc"), _utc_ts(end_utc, "end_utc")
        metars = sql_fn(f"""
            SELECT obs_time, wind_dir_deg, wind_kt, gust_kt, visibility_sm, ceiling_ft, flight_category, raw_text
            FROM {T['weather']}
            WHERE kind = 'metar' AND obs_time BETWEEN {_sql_ts(start - pd.Timedelta(hours=1))} AND {_sql_ts(end)}
            ORDER BY obs_time""")
        taf = sql_fn(f"""
            SELECT obs_time AS issued, raw_text FROM {T['weather']}
            WHERE kind = 'taf' AND obs_time <= {_sql_ts(end)}
            ORDER BY obs_time DESC LIMIT 1""")
        if metars.empty and taf.empty:
            return "No weather data for that window."
        return _json({"metars": _records(metars, 12), "latest_taf": _records(taf)})

    @tool(parse_docstring=True)
    def get_stack_occupancy(minutes: int = 60) -> str:
        """Current and recent holding: aircraft per stack per minute, plus actual holding for recent hours.

        Args:
            minutes: How many minutes of per-minute occupancy to summarise (5-240).
        """
        minutes = max(5, min(int(minutes), 240))
        occ = sql_fn(f"""
            SELECT stack, max_by(holding_count, minute) AS now, max(holding_count) AS peak,
                   round(avg(holding_count), 2) AS mean
            FROM {T['stack_occupancy_1min']}
            WHERE minute > current_timestamp() - INTERVAL {minutes} MINUTES
            GROUP BY stack ORDER BY stack""")
        hourly = sql_fn(f"""
            SELECT hour, holding_flights, mean_hold_min, max_hold_min, go_arounds
            FROM {T['holding_hourly']}
            WHERE hour > current_timestamp() - INTERVAL 6 HOURS ORDER BY hour""")
        return _json({"per_stack_last_minutes": _records(occ), "hourly_actuals": _records(hourly)})

    @tool(parse_docstring=True)
    def get_forecast_accuracy(days: int = 3) -> str:
        """Recent forecast errors (agent vs naive persistence) and recent analyst corrections to learn from.

        Args:
            days: How many days back to look (1-14).
        """
        days = max(1, min(int(days), 14))
        with pg_read() as conn:
            fc = pg_df(conn, """
                SELECT target_hour, predicted_mean_hold_min, actual_mean_hold_min, abs_error
                FROM holding_forecasts
                WHERE status = 'scored' AND target_hour > now() - make_interval(days => %s)
                ORDER BY target_hour DESC LIMIT 24""", (days,))
            corrections = pg_df(conn, """
                SELECT e.event_type, e.location, a.cause AS agent_cause, r.corrected_cause, r.notes
                FROM analyst_reviews r
                JOIN disruption_events e USING (event_id)
                JOIN agent_assessments a USING (assessment_id)
                WHERE r.verdict = 'disagree' ORDER BY r.reviewed_at DESC LIMIT 5""")
        summary = {}
        if not fc.empty:
            summary["agent_mae_min"] = round(float(fc["abs_error"].mean()), 2)
            summary["bias_min"] = round(float((fc["predicted_mean_hold_min"] - fc["actual_mean_hold_min"]).mean()), 2)
        return _json({"summary": summary, "recent_forecasts": _records(fc, 8),
                      "recent_analyst_corrections": _records(corrections)})

    read_tools = [list_open_events, get_flight_track, get_weather, get_stack_occupancy, get_forecast_accuracy]
    if pg_agent is None:
        return read_tools, []

    version = model_version(kind)

    @tool(parse_docstring=True)
    def save_assessment(event_id: str, cause: str, severity: int, confidence: float,
                        reasoning: str, evidence_summary: str) -> str:
        """Save your assessment of one disruption event. Call exactly once per event.

        Args:
            event_id: The event's UUID.
            cause: One of wind, low_visibility_cloud, runway_change, traffic_volume, other.
            severity: 1 (minor), 2 (moderate) or 3 (severe).
            confidence: Between 0 and 1.
            reasoning: 2-4 sentences quoting the numbers you relied on.
            evidence_summary: Short list of the key evidence (METAR used, track facts, stack counts).
        """
        if not UUID_RE.match(event_id or ""):
            return "Error: event_id must be a UUID."
        if cause not in config.CAUSES:
            return f"Error: cause must be one of {config.CAUSES}."
        if int(severity) not in (1, 2, 3):
            return "Error: severity must be 1, 2 or 3."
        if not 0 <= float(confidence) <= 1:
            return "Error: confidence must be between 0 and 1."
        evidence = {"summary": evidence_summary[:2000]}
        with pg_agent() as conn, conn.cursor() as cur:
            # Idempotent: one assessment per event per model version.
            cur.execute("""
                INSERT INTO agent_assessments
                    (event_id, cause, severity, confidence, reasoning, evidence, model_version, mlflow_trace_id)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (event_id, model_version) DO NOTHING""",
                (event_id, cause, int(severity), float(confidence), reasoning[:4000],
                 json.dumps(evidence), version, _trace_id()))
            inserted = cur.rowcount
            # Only ever moves detected -> assessed, so it can't overwrite an analyst decision.
            cur.execute("""UPDATE disruption_events SET status = 'assessed', updated_at = now()
                           WHERE event_id = %s AND status = 'detected'""", (event_id,))
            conn.commit()
        return "Saved." if inserted else "Already assessed by this model version; nothing changed."

    @tool(parse_docstring=True)
    def save_forecast(target_hour_utc: str, predicted_mean_hold_min: float,
                      predicted_holding_flights: int, rationale: str) -> str:
        """Save a holding forecast for one target hour. Call exactly once.

        Args:
            target_hour_utc: The start of the target hour, ISO format (UTC), e.g. "2026-09-21T09:00:00Z".
            predicted_mean_hold_min: Predicted mean holding time per holding flight, in minutes (0-120).
            predicted_holding_flights: Predicted number of flights that will hold (0-200).
            rationale: 2-3 sentences explaining the prediction.
        """
        hour = _utc_ts(target_hour_utc, "target_hour_utc").floor("h")
        now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
        if not (now - pd.Timedelta(hours=1) <= hour <= now + pd.Timedelta(hours=3)):
            return "Error: target hour must be within the next few hours."
        if not 0 <= float(predicted_mean_hold_min) <= 120 or not 0 <= int(predicted_holding_flights) <= 200:
            return "Error: prediction out of range."
        with pg_agent() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO holding_forecasts
                    (target_hour, predicted_mean_hold_min, predicted_holding_flights, rationale, model_version, mlflow_trace_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (target_hour, model_version) DO NOTHING""",
                (hour.to_pydatetime(), float(predicted_mean_hold_min), int(predicted_holding_flights),
                 rationale[:2000], version, _trace_id()))
            inserted = cur.rowcount
            conn.commit()
        return "Saved." if inserted else "A forecast for that hour already exists; nothing changed."

    return read_tools, [save_assessment, save_forecast]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _make_agent(llm, tools, prompt):
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(llm, tools, prompt=SystemMessage(content=prompt))


def build_worker_agents(sql_fn, pg_read, pg_agent):
    """Agents for the background worker: one for assessments, one for forecasts."""
    kind = config.ASSESS_MODEL
    read_a, write_a = build_tools(sql_fn, pg_read, pg_agent, kind=kind)
    read_smart, write_smart = build_tools(sql_fn, pg_read, pg_agent, kind="smart")
    assess = _make_agent(build_llm(kind), read_a + [write_a[0]], ASSESS_PROMPT)
    forecast = _make_agent(build_llm("smart"), read_smart + [write_smart[1]], FORECAST_PROMPT)
    return assess, forecast


def build_chat_agent(sql_fn, pg_read):
    read_tools, _ = build_tools(sql_fn, pg_read, None)
    return _make_agent(build_llm("smart"), read_tools, CHAT_PROMPT)


RUN_CONFIG = {"recursion_limit": 20}   # at most ~8 tool calls per run


def _text(content):
    if isinstance(content, list):  # Anthropic can return a list of content blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def assess_event(agent, event):
    """event: dict/Series with event_id, event_type, location, icao24, callsign, started_at, ended_at."""
    msg = (f"Assess disruption event {event['event_id']}: a {event['event_type']} at {event['location']} "
           f"by aircraft {event['icao24']} (callsign {event.get('callsign') or 'unknown'}), "
           f"from {pd.Timestamp(event['started_at']):%Y-%m-%dT%H:%M:%SZ} to "
           f"{pd.Timestamp(event['ended_at']):%Y-%m-%dT%H:%M:%SZ}.")
    out = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config=RUN_CONFIG)
    return _text(out["messages"][-1].content)


def issue_forecast(agent, target_hour):
    msg = f"Issue the holding forecast for the target hour starting {pd.Timestamp(target_hour):%Y-%m-%dT%H:00:00Z}."
    out = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config=RUN_CONFIG)
    return _text(out["messages"][-1].content)


def chat(agent, history, max_messages=10):
    """history: list of (role, text) tuples ending with the user's message. Returns the reply text."""
    recent = list(history[-max_messages:])
    while recent and recent[0][0] != "user":     # the conversation sent to the model must start with the user
        recent.pop(0)
    out = agent.invoke({"messages": [{"role": r, "content": t} for r, t in recent]}, config=RUN_CONFIG)
    return _text(out["messages"][-1].content)
