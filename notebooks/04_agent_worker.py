# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Agent worker
# MAGIC
# MAGIC Every 30 s:
# MAGIC 1. **Assess**: takes up to 3 events with status `detected` and asks the agent to explain each one.
# MAGIC    The agent must call `save_assessment`, which writes to Lakebase and moves the event to `assessed`.
# MAGIC 2. **Forecast**: if there's no forecast yet for next hour, asks the agent for one (`save_forecast`).
# MAGIC 3. **Score**: fills in the actual holding for forecasts whose hour has finished.
# MAGIC
# MAGIC Every agent run is traced in MLflow (Experiments → `squawk-agent`).
# MAGIC
# MAGIC The model comes from `config.LLM_PROVIDER`. The default is `databricks`, which serves Claude
# MAGIC through the workspace's own model serving - no API key, no third-party gateway.

# COMMAND ----------

# MAGIC  %pip install -q --upgrade "langgraph>=1.2,<2" "langgraph-prebuilt>=1.1,<2" "langchain-core>=1.6,<2" databricks-langchain "psycopg[binary]" "databricks-sdk>=0.81" "mlflow>=3.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("duration_minutes", "5")   # the job sets 60
dbutils.widgets.text("cycle_seconds", "30")
dbutils.widgets.text("max_events_per_cycle", "3")

# COMMAND ----------

import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import mlflow
import pandas as pd

sys.path.insert(0, os.path.abspath("../app"))
from squawk_lib import agent, config, store
from squawk_lib.db import PgSession, spark_sql_fn

spark.conf.set("spark.sql.session.timeZone", "UTC")

# Only the external providers need a key; "databricks" authenticates as you.
if config.LLM_PROVIDER in ("anthropic", "openai"):
    key_env = "ANTHROPIC_API_KEY" if config.LLM_PROVIDER == "anthropic" else "OPENAI_API_KEY"
    os.environ[key_env] = dbutils.secrets.get(config.SECRET_SCOPE, "llm_api_key")

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/squawk-agent")
mlflow.langchain.autolog()

pg_read = PgSession()                     # your role: reads + scoring updates
with pg_read() as conn:
    use_role = store.agent_role_available(conn)
pg_agent = PgSession(role="squawk_agent" if use_role else None)

sql_fn = spark_sql_fn(spark)
assess_agent, forecast_agent = agent.build_worker_agents(sql_fn, pg_read, pg_agent)
FORECAST_VERSION = agent.model_version("smart")

print("Provider:    ", config.LLM_PROVIDER)
print("Assess model:", agent.model_version(config.ASSESS_MODEL))
print("Forecast:    ", FORECAST_VERSION)
print("Agent writes as:", "squawk_agent (restricted role)" if use_role else "your own role (role not available)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the model answers at all
# MAGIC One call, no tools, no agent. If this fails the problem is the model endpoint, not Squawk.

# COMMAND ----------

print(agent.build_llm(config.ASSESS_MODEL).invoke("Reply with the single word: ok").content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try it on one event first
# MAGIC Run this cell by hand once the detector has produced at least one event. Then open the MLflow
# MAGIC experiment to see the trace: every tool call, its arguments and what came back.

# COMMAND ----------

with pg_read() as conn:
    todo = store.events_to_assess(conn, 1)

if todo.empty:
    print("No events waiting - start the detector and wait for an event.")
else:
    ev = todo.iloc[0].to_dict()
    print("Assessing", ev["event_id"], ev["event_type"], ev["location"], ev["callsign"])
    print(agent.assess_event(assess_agent, ev))
    with pg_read() as conn:
        print("Status now:", store.event_status(conn, ev["event_id"]))

# COMMAND ----------

from squawk_lib.db import pg_df

with pg_read() as conn:
    row = pg_df(conn, """
        SELECT cause, severity, confidence, model_version, mlflow_trace_id, reasoning, evidence
        FROM agent_assessments ORDER BY created_at DESC LIMIT 1
    """).iloc[0]

for k in ("cause", "severity", "confidence", "model_version"):
    print(f"{k:14} {row[k]}")
print("\nreasoning:\n", row["reasoning"])
print("\nevidence:\n", row["evidence"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## The worker loop
# MAGIC Everything below is what the job runs. Nothing above this point writes anything except the
# MAGIC single assessment in the test cell.

# COMMAND ----------

import importlib.metadata as md
for p in ("langgraph", "langgraph-prebuilt", "langchain-core", "langchain-anthropic", "mlflow"):
    try:
        print(f"{p:22} {md.version(p)}")
    except Exception:
        print(f"{p:22} not installed")

# COMMAND ----------

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def assess_pending(limit, failures):
    with pg_read() as conn:
        todo = store.events_to_assess(conn, limit + len(failures))
    done = 0
    for ev in todo.to_dict(orient="records"):
        if failures[ev["event_id"]] >= 2 or done >= limit:
            continue
        try:
            reply = agent.assess_event(assess_agent, ev)
            with pg_read() as conn:
                status = store.event_status(conn, ev["event_id"])
            if status == "detected":            # the agent answered without saving
                failures[ev["event_id"]] += 1
                print(f"  {ev['event_id'][:8]} not saved (attempt {failures[ev['event_id']]}): {reply[:120]}")
            else:
                print(f"  {ev['event_id'][:8]} {ev['event_type']} {ev['location']} -> {status}: {reply[:120]}")
            done += 1
        except Exception as e:
            failures[ev["event_id"]] += 1
            print(f"  {ev['event_id'][:8]} failed: {type(e).__name__}: {e}")
            pg_agent.reset()


def forecast_next_hour():
    target = pd.Timestamp(utcnow()).floor("h") + pd.Timedelta(hours=1)
    with pg_read() as conn:
        if store.forecast_exists(conn, target, FORECAST_VERSION):
            return
    print(f"  forecasting {target:%H:%M}Z:", agent.issue_forecast(forecast_agent, target)[:160])


def score_finished():
    with pg_read() as conn:
        todo = store.forecasts_to_score(conn, pd.Timestamp(utcnow()))
    for f in todo.itertuples():
        hour = pd.Timestamp(f.target_hour).tz_convert("UTC").tz_localize(None)
        actual = sql_fn(f"""SELECT holding_flights, mean_hold_min FROM {config.TABLES['holding_hourly']}
                            WHERE hour = TIMESTAMP'{hour:%Y-%m-%d %H:%M:%S}'""")
        flights = int(actual["holding_flights"][0]) if len(actual) else 0
        mean = float(actual["mean_hold_min"][0]) if len(actual) and pd.notna(actual["mean_hold_min"][0]) else 0.0
        with pg_read() as conn:
            store.score_forecast(conn, f.forecast_id, mean, flights)
        print(f"  scored {hour:%H:%M}Z: predicted {f.predicted_mean_hold_min:.1f} min, actual {mean:.1f} min")

# COMMAND ----------

DURATION_S = int(dbutils.widgets.get("duration_minutes")) * 60
CYCLE_S = int(dbutils.widgets.get("cycle_seconds"))
MAX_EVENTS = int(dbutils.widgets.get("max_events_per_cycle"))
failures = defaultdict(int)

started, cycle = time.time(), 0
while time.time() - started < DURATION_S:
    tick, cycle = time.time(), cycle + 1
    print(f"{utcnow():%H:%M:%S}Z cycle {cycle}")
    for step in (lambda: assess_pending(MAX_EVENTS, failures), forecast_next_hour, score_finished):
        try:
            step()
        except Exception as e:
            print(f"  step failed: {type(e).__name__}: {e}")
            pg_read.reset()
            pg_agent.reset()
    time.sleep(max(0.0, CYCLE_S - (time.time() - tick)))

pg_read.close()
pg_agent.close()