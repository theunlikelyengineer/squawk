# Squawk Lakeflow pipeline: analytics built from Lakebase Change Data Feed.
#
# Runs as its OWN triggered pipeline (squawk-analytics), scheduled every 15 minutes, so a problem here
# can never stop Silver ingestion. Create it only after Change Data Feed is switched on
# (02_lakebase_setup section 5), because it reads the lb_<table>_history Delta tables CDF creates.
#
# Pipeline configuration parameters:
#   squawk.catalog            e.g. squawk
#   squawk.schema.gold        e.g. gold
#   squawk.schema.analytics   e.g. analytics
#   squawk.schema.cdf         e.g. lakebase_cdf
#   squawk.cdf_catalog        only if CDF writes to a different catalog (defaults to squawk.catalog)
#
# CDF history columns used here: _pg_change_type (insert / update_preimage / update_postimage / delete),
# _timestamp (when the change was flushed to Delta, ~15 s granularity) and _sort_by (ordering key).

from pyspark import pipelines as dp
from pyspark.sql import functions as F

CATALOG = spark.conf.get("squawk.catalog", "squawk")
GOLD = spark.conf.get("squawk.schema.gold", "gold")
ANALYTICS = spark.conf.get("squawk.schema.analytics", "analytics")
CDF = spark.conf.get("squawk.schema.cdf", "lakebase_cdf")
CDF_CATALOG = spark.conf.get("squawk.cdf_catalog", CATALOG)


def history(table):
    return spark.read.table(f"{CDF_CATALOG}.{CDF}.lb_{table}_history")


def latest_rows(table, key):
    """Current state of each row, rebuilt from its change history (ignores deleted rows)."""
    h = history(table).where(F.col("_pg_change_type").isin("insert", "update_postimage", "delete"))
    latest = h.groupBy(key).agg(F.max("_sort_by").alias("_sort_by"))
    return h.join(latest, [key, "_sort_by"]).where(F.col("_pg_change_type") != "delete")


@dp.materialized_view(
    name=f"{CATALOG}.{ANALYTICS}.event_lifecycle",
    comment="One row per event: when it was detected, assessed by the agent and reviewed by an analyst.",
)
def event_lifecycle():
    h = history("disruption_events").where(F.col("_pg_change_type").isin("insert", "update_postimage"))
    first_ts = lambda cond: F.min(F.when(cond, F.col("_timestamp")))  # noqa: E731
    out = h.groupBy("event_id").agg(
        F.max_by("event_type", "_sort_by").alias("event_type"),
        F.max_by("location", "_sort_by").alias("location"),
        F.max_by("status", "_sort_by").alias("current_status"),
        F.max_by("duration_s", "_sort_by").alias("duration_s"),
        F.min("first_detected_at").alias("detected_at"),
        F.min("trigger_ts").alias("trigger_ts"),
        first_ts(F.col("status") == "assessed").alias("assessed_at"),
        first_ts(F.col("status").isin("confirmed", "rejected")).alias("reviewed_at"),
    )
    secs = lambda a, b: F.col(a).cast("long") - F.col(b).cast("long")  # noqa: E731
    return (
        out.withColumn("detection_latency_s", secs("detected_at", "trigger_ts"))
        .withColumn("time_to_assess_s", secs("assessed_at", "detected_at"))
        .withColumn("time_to_review_s", secs("reviewed_at", "detected_at"))
    )


@dp.materialized_view(
    name=f"{CATALOG}.{ANALYTICS}.agent_agreement",
    comment="How often analysts agree with the agent, by event type and agent cause.",
)
def agent_agreement():
    reviews = latest_rows("analyst_reviews", "review_id")
    latest_review = reviews.groupBy("event_id").agg(F.max("reviewed_at").alias("reviewed_at"))
    reviews = reviews.join(latest_review, ["event_id", "reviewed_at"])
    assessments = latest_rows("agent_assessments", "assessment_id").select(
        "assessment_id", "cause", "severity", "confidence", "model_version")
    events = latest_rows("disruption_events", "event_id").select("event_id", "event_type")
    joined = reviews.join(assessments, "assessment_id", "left").join(events, "event_id", "left")
    return joined.groupBy("event_type", "cause", "model_version").agg(
        F.count("*").alias("reviewed"),
        F.sum(F.when(F.col("verdict") == "agree", 1).otherwise(0)).alias("agree"),
        F.sum(F.when(F.col("verdict") == "disagree", 1).otherwise(0)).alias("disagree"),
        F.sum(F.when(F.col("verdict") == "not_an_event", 1).otherwise(0)).alias("not_an_event"),
        F.round(F.avg("confidence"), 2).alias("mean_confidence"),
    ).withColumn(
        "agreement_rate",
        F.round(F.col("agree") / F.when(F.col("agree") + F.col("disagree") > 0, F.col("agree") + F.col("disagree")), 3)
    ).withColumn(
        "detection_precision", F.round(1 - F.col("not_an_event") / F.col("reviewed"), 3)
    )


@dp.materialized_view(
    name=f"{CATALOG}.{ANALYTICS}.forecast_accuracy",
    comment="Each scored forecast next to a naive persistence baseline (next hour = previous hour).",
)
def forecast_accuracy():
    fc = latest_rows("holding_forecasts", "forecast_id").where("status = 'scored'")
    hourly = spark.read.table(f"{CATALOG}.{GOLD}.holding_hourly").select(
        (F.col("hour") + F.expr("INTERVAL 1 HOUR")).alias("target_hour"),
        F.coalesce("mean_hold_min", F.lit(0.0)).alias("naive_pred_min"),
    )
    return (
        fc.select("forecast_id", "target_hour", "model_version", "predicted_mean_hold_min",
                  "actual_mean_hold_min", "abs_error")
        .join(hourly, "target_hour", "left")
        .withColumn("naive_pred_min", F.coalesce("naive_pred_min", F.lit(0.0)))
        .withColumn("naive_abs_error", F.abs(F.col("naive_pred_min") - F.col("actual_mean_hold_min")))
        .withColumn("hour_of_day", F.hour("target_hour"))
    )


@dp.materialized_view(
    name=f"{CATALOG}.{ANALYTICS}.daily_summary",
    comment="Daily KPIs for the Analytics tab.",
)
def daily_summary():
    life = spark.read.table(f"{CATALOG}.{ANALYTICS}.event_lifecycle")
    fc = spark.read.table(f"{CATALOG}.{ANALYTICS}.forecast_accuracy")
    events = life.groupBy(F.to_date("detected_at").alias("day")).agg(
        F.count("*").alias("events"),
        F.sum(F.when(F.col("event_type") == "holding", 1).otherwise(0)).alias("holding_events"),
        F.sum(F.when(F.col("event_type") == "go_around", 1).otherwise(0)).alias("go_arounds"),
        F.percentile_approx("detection_latency_s", 0.5).alias("latency_p50_s"),
        F.percentile_approx("detection_latency_s", 0.95).alias("latency_p95_s"),
        F.percentile_approx("time_to_review_s", 0.5).alias("time_to_review_p50_s"),
    )
    forecasts = fc.groupBy(F.to_date("target_hour").alias("day")).agg(
        F.round(F.avg("abs_error"), 2).alias("agent_mae_min"),
        F.round(F.avg("naive_abs_error"), 2).alias("naive_mae_min"),
        F.count("*").alias("forecasts_scored"),
    )
    return events.join(forecasts, "day", "full_outer")
