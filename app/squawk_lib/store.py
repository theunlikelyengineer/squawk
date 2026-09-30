"""All Lakebase reads and writes used by the detector, the worker and the app, in one place.

Every function takes an open psycopg connection and commits its own work.
"""

import pandas as pd

from . import config
from .db import pg_df


def _utc(ts):
    """pandas/py timestamp (naive = UTC) -> timezone-aware datetime for timestamptz columns."""
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

UPSERT_EVENT = """
INSERT INTO disruption_events
    (event_id, event_type, icao24, callsign, location, started_at, ended_at, duration_s, trigger_ts)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
    ended_at   = EXCLUDED.ended_at,
    duration_s = EXCLUDED.duration_s,
    callsign   = COALESCE(EXCLUDED.callsign, disruption_events.callsign),
    updated_at = now()
WHERE disruption_events.ended_at < EXCLUDED.ended_at
"""


def upsert_events(conn, episodes):
    """Insert new events and extend ongoing ones. Never touches status (the agent and analyst own it)."""
    rows = [
        (r.event_id, r.event_type, r.icao24, r.callsign if isinstance(r.callsign, str) else None, r.location,
         _utc(r.started_at), _utc(r.ended_at), int(r.duration_s), _utc(r.trigger_ts))
        for r in episodes.itertuples()
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(UPSERT_EVENT, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def agent_role_available(conn, role="squawk_agent"):
    exists = pg_df(conn, "SELECT count(*) AS n FROM pg_roles WHERE rolname = %s", (role,))["n"][0] > 0
    if not exists:
        return False
    return bool(pg_df(conn, "SELECT pg_has_role(current_user, %s, 'MEMBER') AS ok", (role,))["ok"][0])


def events_to_assess(conn, limit):
    return pg_df(conn, """
        SELECT event_id::text AS event_id, event_type, icao24, callsign, location, started_at, ended_at
        FROM disruption_events WHERE status = 'detected'
        ORDER BY started_at LIMIT %s""", (limit,))


def event_status(conn, event_id):
    df = pg_df(conn, "SELECT status FROM disruption_events WHERE event_id = %s", (event_id,))
    return None if df.empty else df["status"][0]


def forecast_exists(conn, target_hour, model_version):
    df = pg_df(conn, "SELECT count(*) AS n FROM holding_forecasts WHERE target_hour = %s AND model_version = %s",
               (_utc(target_hour), model_version))
    return df["n"][0] > 0


def forecasts_to_score(conn, now):
    """Pending forecasts whose target hour finished at least 5 minutes ago."""
    return pg_df(conn, """
        SELECT forecast_id, target_hour, predicted_mean_hold_min, predicted_holding_flights
        FROM holding_forecasts
        WHERE status = 'pending' AND target_hour + interval '65 minutes' <= %s
        ORDER BY target_hour""", (_utc(now),))


def score_forecast(conn, forecast_id, actual_mean_hold_min, actual_holding_flights):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE holding_forecasts
            SET actual_mean_hold_min = %s, actual_holding_flights = %s,
                abs_error = abs(predicted_mean_hold_min - %s), status = 'scored'
            WHERE forecast_id = %s AND status = 'pending'""",
            (actual_mean_hold_min, actual_holding_flights, actual_mean_hold_min, int(forecast_id)))
    conn.commit()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def event_queue(conn, hours=24, limit=200):
    return pg_df(conn, """
        SELECT e.event_id::text AS event_id, e.event_type, e.location, e.icao24, e.callsign,
               e.started_at, e.ended_at, round(e.duration_s / 60.0, 1) AS duration_min, e.status,
               e.trigger_ts, e.first_detected_at,
               a.assessment_id, a.cause, a.severity, a.confidence, a.reasoning, a.evidence, a.model_version,
               a.mlflow_trace_id,
               r.verdict, r.corrected_cause, r.notes, r.reviewer, r.reviewed_at
        FROM disruption_events e
        LEFT JOIN LATERAL (SELECT * FROM agent_assessments x WHERE x.event_id = e.event_id
                           ORDER BY created_at DESC LIMIT 1) a ON true
        LEFT JOIN LATERAL (SELECT * FROM analyst_reviews y WHERE y.event_id = e.event_id
                           ORDER BY reviewed_at DESC LIMIT 1) r ON true
        WHERE e.started_at > now() - make_interval(hours => %s)
        ORDER BY e.started_at DESC LIMIT %s""", (hours, limit))


def active_holding_icao24(conn, minutes=3):
    df = pg_df(conn, """SELECT DISTINCT icao24 FROM disruption_events
                        WHERE event_type = 'holding' AND ended_at > now() - make_interval(mins => %s)""", (minutes,))
    return set(df["icao24"])


def record_review(conn, event_id, assessment_id, verdict, corrected_cause=None, notes=None, reviewer=None):
    """Save the analyst's decision and move the event to confirmed/rejected, in one transaction."""
    if verdict not in ("agree", "disagree", "not_an_event"):
        raise ValueError("bad verdict")
    if corrected_cause is not None and corrected_cause not in config.CAUSES:
        raise ValueError("bad cause")
    new_status = "rejected" if verdict == "not_an_event" else "confirmed"
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO analyst_reviews (event_id, assessment_id, verdict, corrected_cause, notes, reviewer)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (event_id, None if assessment_id is None or pd.isna(assessment_id) else int(assessment_id),
                     verdict, corrected_cause, notes or None, reviewer))
        cur.execute("UPDATE disruption_events SET status = %s, updated_at = now() WHERE event_id = %s",
                    (new_status, event_id))
    conn.commit()
    return new_status


def forecasts(conn, hours=48):
    return pg_df(conn, """
        SELECT target_hour, predicted_mean_hold_min, actual_mean_hold_min, predicted_holding_flights,
               actual_holding_flights, abs_error, status, rationale, model_version
        FROM holding_forecasts WHERE target_hour > now() - make_interval(hours => %s)
        ORDER BY target_hour""", (hours,))
