"""Squawk - Databricks App (Streamlit).

Tabs: Live map · Event queue · Forecast · Ask Squawk · Analytics
Reads Delta through the SQL warehouse resource and Lakebase through the postgres resource.
"""
import json
import os

import pandas as pd
import pydeck as pdk
import streamlit as st

from squawk_lib import config, store
from squawk_lib.db import PgSession, warehouse_sql_fn

st.set_page_config(page_title="Squawk", page_icon="✈️", layout="wide")
T = config.TABLES
REFRESH = "15s"

# The LLM key comes from the app's secret resource (see app.yaml).
if os.environ.get("SQUAWK_LLM_API_KEY"):
    key_env = "ANTHROPIC_API_KEY" if config.LLM_PROVIDER == "anthropic" else "OPENAI_API_KEY"
    os.environ.setdefault(key_env, os.environ["SQUAWK_LLM_API_KEY"])


# ---------------------------------------------------------------------------
# Connections (one per app process, shared by all users)
# ---------------------------------------------------------------------------

def pg():
    # One Lakebase connection per browser session, so two users' transactions never interleave.
    if "pg" not in st.session_state:
        st.session_state["pg"] = PgSession()
    return st.session_state["pg"]


@st.cache_resource
def sql_runner():
    return warehouse_sql_fn()


@st.cache_data(ttl=10, show_spinner=False)
def delta(query):
    return sql_runner()(query)


def lakebase(fn, *args, **kwargs):
    with pg()() as conn:
        return fn(conn, *args, **kwargs)


def reviewer():
    try:
        return st.context.headers.get("X-Forwarded-Email") or "unknown"
    except Exception:
        return "unknown"


st.title("✈️ Squawk")
st.caption("Arrival disruption at London Heathrow: detected live from ADS-B, explained by an agent, reviewed by you. "
           "All times UTC.")

tab_map, tab_queue, tab_forecast, tab_chat, tab_analytics = st.tabs(
    ["Live map", "Event queue", "Forecast", "Ask Squawk", "Analytics"])


# ---------------------------------------------------------------------------
# Live map
# ---------------------------------------------------------------------------

@st.fragment(run_every=REFRESH)
def live_map():
    aircraft = delta(f"""
        SELECT icao24, max_by(callsign, event_ts) AS callsign, max_by(lat, event_ts) AS lat,
               max_by(lon, event_ts) AS lon, round(max_by(alt_ft, event_ts)) AS alt_ft,
               max(event_ts) AS last_seen
        FROM {T['positions']}
        WHERE event_ts > current_timestamp() - INTERVAL 2 MINUTES AND NOT coalesce(on_ground, false)
        GROUP BY icao24""")
    holding = lakebase(store.active_holding_icao24)
    aircraft["holding"] = aircraft["icao24"].isin(holding)
    aircraft["color"] = aircraft["holding"].map(lambda h: [220, 50, 47, 230] if h else [42, 104, 207, 180])
    aircraft["label"] = aircraft["callsign"].fillna(aircraft["icao24"])

    stacks = pd.DataFrame([{"id": k, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
                           for k, v in config.STACKS.items()])
    c1, c2, c3 = st.columns(3)
    c1.metric("Aircraft in the London box", len(aircraft))
    c2.metric("Holding now", int(aircraft["holding"].sum()))
    c3.metric("Last position", str(aircraft["last_seen"].max())[:19] if len(aircraft) else "-")

    deck = pdk.Deck(
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=51.5, longitude=-0.3, zoom=8.3),
        layers=[
            pdk.Layer("ScatterplotLayer", stacks, get_position="[lon, lat]", get_radius=22224,   # 12 NM
                      stroked=True, filled=False, get_line_color=[120, 120, 120, 160], line_width_min_pixels=1),
            pdk.Layer("TextLayer", stacks, get_position="[lon, lat]", get_text="id", get_size=13,
                      get_color=[90, 90, 90]),
            pdk.Layer("ScatterplotLayer", aircraft, get_position="[lon, lat]", get_fill_color="color",
                      get_radius=900, pickable=True),
        ],
        tooltip={"text": "{label}\n{alt_ft} ft"},
    )
    st.pydeck_chart(deck, height=560)
    st.caption("Red = currently in a holding episode. Grey rings = the four holding stacks (12 NM).")


with tab_map:
    live_map()


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------

STATUS_ICON = {"detected": "🟡", "assessed": "🔵", "confirmed": "🟢", "rejected": "⚪"}


def _blank(v):
    return v is None or (not isinstance(v, (str, dict, list)) and pd.isna(v))


def event_label(r):
    return (f"{pd.Timestamp(r['started_at']):%H:%M} · {r['event_type'].replace('_', '-')} · {r['location']} · "
            f"{r['icao24'] if _blank(r['callsign']) else r['callsign']} · {STATUS_ICON.get(r['status'], '')} {r['status']}")


def event_detail(ev):
    left, right = st.columns([3, 2])
    with left:
        who = ev["icao24"] if _blank(ev.get("callsign")) else ev["callsign"]
        st.subheader(f"{ev['event_type'].replace('_', '-')} · {ev['location']} · {who}")
        st.write(f"{pd.Timestamp(ev['started_at']):%H:%M:%S} → {pd.Timestamp(ev['ended_at']):%H:%M:%S} "
                 f"({ev['duration_min']} min) · status **{ev['status']}**")
        start = pd.Timestamp(ev["started_at"]).tz_convert("UTC").tz_localize(None) - pd.Timedelta(minutes=10)
        end = pd.Timestamp(ev["ended_at"]).tz_convert("UTC").tz_localize(None) + pd.Timedelta(minutes=5)
        track = delta(f"""SELECT event_ts, alt_ft FROM {T['positions']}
                          WHERE icao24 = '{ev['icao24']}'
                            AND event_ts BETWEEN TIMESTAMP'{start:%Y-%m-%d %H:%M:%S}' AND TIMESTAMP'{end:%Y-%m-%d %H:%M:%S}'
                          ORDER BY event_ts""")
        if len(track):
            st.caption("Altitude (ft)")
            st.line_chart(track.set_index("event_ts")[["alt_ft"]], height=220)
        metar = delta(f"""SELECT obs_time, raw_text FROM {T['weather']} WHERE kind = 'metar'
                          AND obs_time BETWEEN TIMESTAMP'{start - pd.Timedelta(minutes=50):%Y-%m-%d %H:%M:%S}'
                                           AND TIMESTAMP'{end:%Y-%m-%d %H:%M:%S}' ORDER BY obs_time""")
        if len(metar):
            st.caption("METAR reports around the event")
            st.dataframe(metar, hide_index=True, width="stretch")
    with right:
        st.markdown("**Agent assessment**")
        if _blank(ev.get("cause")):
            st.write("Waiting for the agent…")
        else:
            st.write(f"Cause: **{ev['cause']}** · severity {int(ev['severity'])} · confidence {float(ev['confidence']):.2f}")
            st.write(ev["reasoning"])
            evidence = ev.get("evidence")
            if isinstance(evidence, str):
                evidence = json.loads(evidence)
            if isinstance(evidence, dict) and evidence.get("summary"):
                st.caption(evidence["summary"])
            trace = "-" if _blank(ev.get("mlflow_trace_id")) else ev["mlflow_trace_id"]
            st.caption(f"{ev['model_version']} · MLflow trace {trace}")

        st.markdown("**Your review**")
        if not _blank(ev.get("verdict")):
            extra = "" if _blank(ev.get("corrected_cause")) else f" → {ev['corrected_cause']}"
            st.success(f"Reviewed: {ev['verdict']}{extra}")
        with st.form(f"review_{ev['event_id']}"):
            verdict = st.radio("Verdict", ["agree", "disagree", "not_an_event"], horizontal=True,
                               format_func=lambda v: {"agree": "Confirm", "disagree": "Correct cause",
                                                      "not_an_event": "Reject (not an event)"}[v])
            corrected = st.selectbox("Correct cause (only if you pick Correct cause)", ["-"] + config.CAUSES)
            notes = st.text_input("Notes (optional)")
            if st.form_submit_button("Save review"):
                cause = corrected if verdict == "disagree" and corrected != "-" else None
                if verdict == "disagree" and cause is None:
                    st.error("Pick the correct cause.")
                else:
                    assessment_id = None if _blank(ev.get("assessment_id")) else int(ev["assessment_id"])
                    new_status = lakebase(store.record_review, ev["event_id"], assessment_id, verdict,
                                          cause, notes, reviewer())
                    st.success(f"Saved. Event is now {new_status}.")


@st.fragment(run_every=REFRESH)
def event_queue():
    q = lakebase(store.event_queue, 24)
    if q.empty:
        st.info("No events in the last 24 hours yet. Holding is most common during the morning arrival peak.")
        return
    counts = q["status"].value_counts()
    cols = st.columns(4)
    for col, status in zip(cols, ["detected", "assessed", "confirmed", "rejected"]):
        col.metric(f"{STATUS_ICON[status]} {status}", int(counts.get(status, 0)))

    show = q.assign(started=pd.to_datetime(q["started_at"]).dt.strftime("%H:%M"))[
        ["started", "event_type", "location", "callsign", "duration_min", "status", "cause", "confidence", "verdict"]]
    st.dataframe(show, hide_index=True, width="stretch", height=260)

    # Chosen by event_id (not row number) so the selection survives the auto-refresh.
    ids = q["event_id"].tolist()
    labels = {r["event_id"]: event_label(r) for r in q.to_dict(orient="records")}
    chosen = st.selectbox("Open an event", ids, format_func=labels.get, key="chosen_event")
    if chosen in ids:
        event_detail(q[q["event_id"] == chosen].iloc[0].to_dict())


with tab_queue:
    event_queue()


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

with tab_forecast:
    fc = lakebase(store.forecasts, 48)
    if fc.empty:
        st.info("No forecasts yet. The agent worker issues one for the next hour at the start of each hour.")
    else:
        fc["target_hour"] = pd.to_datetime(fc["target_hour"]).dt.tz_convert("UTC").dt.tz_localize(None)
        st.line_chart(fc.set_index("target_hour")[["predicted_mean_hold_min", "actual_mean_hold_min"]], height=300)
        scored = fc[fc["status"] == "scored"]
        if len(scored):
            st.metric("Agent mean absolute error (min)", f"{scored['abs_error'].mean():.1f}")
        st.dataframe(fc.sort_values("target_hour", ascending=False), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Ask Squawk (read-only agent)
# ---------------------------------------------------------------------------

def chat_agent():
    if "chat_agent" not in st.session_state:
        from squawk_lib import agent
        st.session_state["chat_agent"] = agent.build_chat_agent(lambda q: sql_runner()(q), pg())
    return st.session_state["chat_agent"]


with tab_chat:
    st.caption("The chat agent has read-only tools: it can look things up but can't change any data.")
    history = st.session_state.setdefault("chat", [])
    for role, text in history:
        st.chat_message(role).write(text)
    prompt = st.chat_input("e.g. Why is Lambourne busy right now?")
    if prompt:
        history.append(("user", prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"), st.spinner("Checking the data…"):
            from squawk_lib import agent
            try:
                reply = agent.chat(chat_agent(), history)
            except Exception as e:
                reply = f"Sorry, that failed: {type(e).__name__}: {e}"
            st.write(reply)
        history.append(("assistant", reply))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def try_delta(query):
    try:
        return delta(query)
    except Exception:
        return None


with tab_analytics:
    a = config.CATALOG
    rows = try_delta(f"SELECT count(*) AS n FROM {T['positions']}")
    credits = try_delta(f"SELECT credits_remaining FROM {T['api_call_log']} WHERE credits_remaining IS NOT NULL "
                        "ORDER BY called_at DESC LIMIT 1")
    c1, c2 = st.columns(2)
    c1.metric("Silver position rows (volume)", f"{int(rows['n'][0]):,}" if rows is not None else "-")
    c2.metric("OpenSky credits left today", int(credits["credits_remaining"][0]) if credits is not None and len(credits) else "-")

    daily = try_delta(f"SELECT * FROM {a}.{config.SCHEMA_ANALYTICS}.daily_summary ORDER BY day DESC")
    agree = try_delta(f"SELECT * FROM {a}.{config.SCHEMA_ANALYTICS}.agent_agreement ORDER BY reviewed DESC")
    if daily is None:
        st.info("Analytics tables appear once Change Data Feed is on and the squawk-analytics pipeline has run (Phase 6).")
    else:
        st.subheader("Daily summary")
        st.caption("Latency = seconds from the position that triggered detection to the event reaching the queue.")
        st.dataframe(daily, hide_index=True, width="stretch")
        st.subheader("Agent vs analyst")
        st.dataframe(agree, hide_index=True, width="stretch")
