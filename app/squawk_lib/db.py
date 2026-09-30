"""Connections: Lakebase (Postgres) and Delta (Spark in notebooks, SQL warehouse in the app)."""
import os
import time
from contextlib import contextmanager

import pandas as pd

from . import config

_ws = None
_host = None


def workspace():
    global _ws
    if _ws is None:
        from databricks.sdk import WorkspaceClient
        _ws = WorkspaceClient()
    return _ws


# ---------------------------------------------------------------------------
# Lakebase
# ---------------------------------------------------------------------------

def lakebase_host():
    """PGHOST if the app provides it, else looked up from the endpoint name."""
    global _host
    if os.environ.get("PGHOST"):
        return os.environ["PGHOST"]
    if _host is None:
        ep = workspace().postgres.get_endpoint(name=config.LAKEBASE_ENDPOINT)
        _host = ep.status.hosts.host
    return _host


def lakebase_user():
    """In the app this is the app's service principal (PGUSER); in notebooks it's you."""
    return os.environ.get("PGUSER") or workspace().current_user.me().user_name


def pg_connect(role=None):
    """Open a Postgres connection with a fresh OAuth token.

    role: optional Postgres role to switch to after connecting (the agent uses
    "squawk_agent", which can only write to its own tables).
    """
    import psycopg
    from psycopg import sql

    token = workspace().postgres.generate_database_credential(endpoint=config.LAKEBASE_ENDPOINT).token
    conn = psycopg.connect(
        host=lakebase_host(),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=config.LAKEBASE_DATABASE,
        user=lakebase_user(),
        password=token,
        sslmode="require",
        connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(config.LAKEBASE_SCHEMA)))
        if role:
            cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
    conn.commit()
    return conn


class PgSession:
    """Keeps one connection open for a long-running loop and replaces it before
    the OAuth token expires (tokens last about an hour) or after an error.

    Use it as `with session() as conn:` - the connection stays open afterwards
    (unlike `with psycopg.connect(...)`, which would close it).
    """

    def __init__(self, role=None, max_age_s=45 * 60):
        self.role, self.max_age_s = role, max_age_s
        self._conn, self._opened = None, 0.0

    def get(self):
        stale = time.time() - self._opened > self.max_age_s
        if self._conn is None or self._conn.closed or stale:
            self.close()
            self._conn, self._opened = pg_connect(self.role), time.time()
        return self._conn

    @contextmanager
    def __call__(self):
        conn = self.get()
        try:
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                self.reset()
            raise

    def reset(self):
        self.close()

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None


def pg_df(conn, query, params=None):
    """Run a SELECT and return a pandas DataFrame."""
    with conn.cursor() as cur:
        cur.execute(query, params)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    conn.commit()
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

def spark_sql_fn(spark):
    """Delta reader for notebooks and jobs."""
    return lambda query: spark.sql(query).toPandas()


def warehouse_sql_fn(warehouse_id=None):
    """Delta reader for the Databricks App, via the SQL warehouse resource."""
    from databricks import sql as dbsql
    from databricks.sdk.core import Config

    cfg = Config()
    warehouse_id = warehouse_id or os.environ["DATABRICKS_WAREHOUSE_ID"]

    def run(query):
        with dbsql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
        ) as conn, conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC'")      # timestamp literals in our queries are UTC
            cur.execute(query)
            return cur.fetchall_arrow().to_pandas()

    return run
