# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Lakebase setup
# MAGIC
# MAGIC Safe to **Run all** at any point: every section is idempotent, section 4 skips itself until you paste
# MAGIC the app's ID, and section 5 skips itself until you set `ENABLE_CDF = True` (Phase 6).

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]" "databricks-sdk>=0.81"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../app"))
from databricks.sdk import WorkspaceClient
from squawk_lib import config
from squawk_lib.db import pg_connect, pg_df

w = WorkspaceClient()
S = config.LAKEBASE_SCHEMA

try:
    conn = pg_connect()
    print("Connected to Lakebase as", pg_df(conn, "SELECT current_user AS u")["u"][0])
except Exception as e:
    conn = None
    print("Not connected yet (expected until LAKEBASE_ENDPOINT in config.py is right):", str(e).splitlines()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find your Lakebase endpoint
# MAGIC Lists every Lakebase project you can see, with its branches and endpoints. Copy the **endpoint name** of
# MAGIC your Squawk project (it looks like `projects/.../branches/.../endpoints/...`) into
# MAGIC `LAKEBASE_ENDPOINT` in `app/squawk_lib/config.py`, then rerun this notebook from the top.

# COMMAND ----------

for project in w.postgres.list_projects():
    print("PROJECT ", project.name)
    for branch in w.postgres.list_branches(parent=project.name):
        print("  BRANCH  ", branch.name)
        for ep in w.postgres.list_endpoints(parent=branch.name):
            print("    ENDPOINT", ep.name, "| host:", ep.status.hosts.host if ep.status and ep.status.hosts else "?")
        for db in w.postgres.list_databases(parent=branch.name):
            print("    DATABASE", db.name)

print("\nconfig.LAKEBASE_ENDPOINT is currently:", config.LAKEBASE_ENDPOINT)
assert conn is not None, "Copy your endpoint name into LAKEBASE_ENDPOINT in config.py, then Run all again."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the schema and the four tables
# MAGIC Safe to rerun (everything is `IF NOT EXISTS`). `REPLICA IDENTITY FULL` is required by Change Data Feed.

# COMMAND ----------

CAUSES = ", ".join(f"'{c}'" for c in config.CAUSES)

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {S};
SET search_path TO {S};

CREATE TABLE IF NOT EXISTS disruption_events (
    event_id          uuid PRIMARY KEY,
    event_type        text NOT NULL CHECK (event_type IN ('holding', 'go_around')),
    icao24            text NOT NULL,
    callsign          text,
    location          text NOT NULL,
    started_at        timestamptz NOT NULL,
    ended_at          timestamptz NOT NULL,
    duration_s        integer NOT NULL,
    trigger_ts        timestamptz,
    first_detected_at timestamptz NOT NULL DEFAULT now(),
    status            text NOT NULL DEFAULT 'detected'
                      CHECK (status IN ('detected', 'assessed', 'confirmed', 'rejected')),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (icao24, event_type, started_at)
);
CREATE INDEX IF NOT EXISTS disruption_events_status_idx ON disruption_events (status, started_at);

CREATE TABLE IF NOT EXISTS agent_assessments (
    assessment_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id        uuid NOT NULL REFERENCES disruption_events (event_id),
    cause           text NOT NULL CHECK (cause IN ({CAUSES})),
    severity        smallint NOT NULL CHECK (severity BETWEEN 1 AND 3),
    confidence      real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reasoning       text NOT NULL,
    evidence        jsonb,
    model_version   text NOT NULL,
    mlflow_trace_id text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (event_id, model_version)
);

CREATE TABLE IF NOT EXISTS analyst_reviews (
    review_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id        uuid NOT NULL REFERENCES disruption_events (event_id),
    assessment_id   bigint REFERENCES agent_assessments (assessment_id),
    verdict         text NOT NULL CHECK (verdict IN ('agree', 'disagree', 'not_an_event')),
    corrected_cause text CHECK (corrected_cause IS NULL OR corrected_cause IN ({CAUSES})),
    notes           text,
    reviewer        text,
    reviewed_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS holding_forecasts (
    forecast_id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    issued_at                 timestamptz NOT NULL DEFAULT now(),
    target_hour               timestamptz NOT NULL,
    predicted_mean_hold_min   real NOT NULL,
    predicted_holding_flights integer NOT NULL,
    rationale                 text,
    model_version             text NOT NULL,
    mlflow_trace_id           text,
    actual_mean_hold_min      real,
    actual_holding_flights    integer,
    abs_error                 real,
    status                    text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'scored')),
    UNIQUE (target_hour, model_version)
);

ALTER TABLE disruption_events REPLICA IDENTITY FULL;
ALTER TABLE agent_assessments REPLICA IDENTITY FULL;
ALTER TABLE analyst_reviews   REPLICA IDENTITY FULL;
ALTER TABLE holding_forecasts REPLICA IDENTITY FULL;
"""

with conn.cursor() as cur:
    cur.execute(DDL)
conn.commit()
print(pg_df(conn, f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{S}' ORDER BY 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The agent's restricted role
# MAGIC The agent connects as you, then runs `SET ROLE squawk_agent`. From then on Postgres itself only lets it
# MAGIC read everything, insert into `agent_assessments` and `holding_forecasts`, and update the `status`
# MAGIC column of `disruption_events`. It cannot write reviews or delete anything.
# MAGIC
# MAGIC If `CREATE ROLE` fails with a permissions error, the worker falls back to your own role and the same
# MAGIC rules are enforced only in the tool code. Mention that in your write-up if it happens.

# COMMAND ----------

me = w.current_user.me().user_name
ROLE_SQL = f"""
DO $$ BEGIN
    CREATE ROLE squawk_agent NOLOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
GRANT USAGE ON SCHEMA {S} TO squawk_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA {S} TO squawk_agent;
GRANT INSERT ON {S}.agent_assessments, {S}.holding_forecasts TO squawk_agent;
GRANT UPDATE (status, updated_at) ON {S}.disruption_events TO squawk_agent;
GRANT squawk_agent TO "{me}";
"""
try:
    with conn.cursor() as cur:
        cur.execute(ROLE_SQL)
    conn.commit()
    print("Role squawk_agent ready and granted to", me)
except Exception as e:
    conn.rollback()
    print("Could not create the role (the worker will fall back to your own role):", e)

# COMMAND ----------

# Prove the guardrail works: as squawk_agent, writing a review must be refused.
has_role = pg_df(conn, "SELECT count(*) AS n FROM pg_roles WHERE rolname = 'squawk_agent'")["n"][0] > 0
if not has_role:
    print("No squawk_agent role - skipping this test.")
else:
    agent_conn = None
    try:
        agent_conn = pg_connect(role="squawk_agent")
        with agent_conn.cursor() as cur:
            cur.execute("INSERT INTO analyst_reviews (event_id, verdict) VALUES (gen_random_uuid(), 'agree')")
        print("UNEXPECTED: the agent role could write a review")
    except Exception as e:
        print("Good - refused as expected:", str(e).splitlines()[0])
    finally:
        if agent_conn is not None:
            agent_conn.rollback()
            agent_conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. (Phase 5) Give the app's service principal access
# MAGIC After you've created the app and added this Lakebase database as a resource, copy the app's
# MAGIC **service principal client ID** (Compute → Apps → your app → Authorization) into the box and run the cell.

# COMMAND ----------

dbutils.widgets.text("app_client_id", "")

# COMMAND ----------

app_sp = dbutils.widgets.get("app_client_id").strip()
APP_SQL = f"""
GRANT USAGE ON SCHEMA {S} TO "{app_sp}";
GRANT SELECT ON ALL TABLES IN SCHEMA {S} TO "{app_sp}";
GRANT INSERT ON {S}.analyst_reviews TO "{app_sp}";
GRANT UPDATE (status, updated_at) ON {S}.disruption_events TO "{app_sp}";
"""
if not app_sp:
    print("Skipped: paste the app's service principal client ID into the app_client_id box (Phase 5).")
else:
    try:
        with conn.cursor() as cur:
            cur.execute(APP_SQL)
        conn.commit()
        print("Granted app access to", app_sp)
    except Exception as e:
        conn.rollback()
        print("Grant failed. If the role doesn't exist yet, make sure the Lakebase resource is attached to the app "
              "and the app has been deployed once, or create it with: SELECT databricks_create_role("
              f"'{app_sp}', 'SERVICE_PRINCIPAL');\n", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. (Phase 6) Turn on Lakebase Change Data Feed
# MAGIC Streams every insert, update and delete on the four tables into `lb_<table>_history` Delta tables in
# MAGIC `<catalog>.lakebase_cdf`. You can do the same in the UI: Lakebase → your project → branch → **Lakebase CDF** tab.
# MAGIC
# MAGIC Requirement: the destination catalog must use external storage (most bootcamp catalogs do). If this fails with
# MAGIC a storage error, see the fallback in the build guide.

# COMMAND ----------

from databricks.sdk.service.postgres import CdfConfig

ENABLE_CDF = False             # set to True in Phase 6, then Run all
CDF_CATALOG = config.CATALOG   # change if your admin gives you a different catalog with external storage
CDF_SCHEMA = "student_jcdc9919_capstone"


def pg_db_name(d):
    return (d.status and d.status.postgres_database) or (d.spec and d.spec.postgres_database) or d.database_id


branch = config.LAKEBASE_ENDPOINT.split("/endpoints/")[0]
database = next(d.name for d in w.postgres.list_databases(parent=branch) if pg_db_name(d) == config.LAKEBASE_DATABASE)

if not ENABLE_CDF:
    print("Skipped: set ENABLE_CDF = True in Phase 6.")
else:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CDF_CATALOG}.{CDF_SCHEMA}")
    existing = list(w.postgres.list_cdf_configs(parent=database))
    if existing:
        print("CDF already configured:", [c.name for c in existing])
    else:
        op = w.postgres.create_cdf_config(
            parent=database,
            cdf_config=CdfConfig(catalog=CDF_CATALOG, schema=CDF_SCHEMA, postgres_schema=S),
        )
        print("CDF config created:", op.wait().name)

# COMMAND ----------

for cfg in w.postgres.list_cdf_configs(parent=database):
    for st in w.postgres.list_cdf_statuses(parent=cfg.name):
        print(st.postgres_table, "->", st.uc_table, "|", st.state, "|", st.status_detail or "")