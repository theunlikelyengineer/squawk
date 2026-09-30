# Squawk Lakeflow pipeline: Bronze and Silver.
#
# Bronze: Auto Loader streams the poller's JSON-lines files from the landing volume.
# Silver: cleans, deduplicates, converts units and enriches aircraft positions;
#         parses and deduplicates the weather reports.
#
# Pipeline configuration parameters (set them in the pipeline's Settings):
#   squawk.catalog         e.g. squawk
#   squawk.schema.raw      e.g. raw
#   squawk.schema.bronze   e.g. bronze
#   squawk.schema.silver   e.g. silver

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = spark.conf.get("squawk.catalog", "squawk")
RAW = spark.conf.get("squawk.schema.raw", "raw")
BRONZE = spark.conf.get("squawk.schema.bronze", "bronze")
SILVER = spark.conf.get("squawk.schema.silver", "silver")

LANDING = f"/Volumes/{CATALOG}/{RAW}/landing"
BBOX = {"lamin": 50.60, "lomin": -1.81, "lamax": 52.32, "lomax": 0.89}   # keep in sync with config.py

M_TO_FT, MS_TO_KT, MS_TO_FPM = 3.28084, 1.943844, 196.8504

# One JSON object per aircraft per poll, written by 01_poller.
STATE_SCHEMA = (
    "poll_id STRING, fetched_at DOUBLE, api_time BIGINT, icao24 STRING, callsign STRING, "
    "origin_country STRING, time_position BIGINT, last_contact BIGINT, longitude DOUBLE, latitude DOUBLE, "
    "baro_altitude DOUBLE, on_ground BOOLEAN, velocity DOUBLE, true_track DOUBLE, vertical_rate DOUBLE, "
    "geo_altitude DOUBLE, squawk STRING, spi BOOLEAN, position_source INT, category INT"
)
WEATHER_SCHEMA = "kind STRING, fetched_at DOUBLE, obs_time BIGINT, raw_json STRING"


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

@dp.table(
    name=f"{CATALOG}.{BRONZE}.opensky_states",
    comment="Raw OpenSky state vectors, one row per aircraft per poll (append-only).",
    partition_cols=["ingest_date"],
)
def opensky_states():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(STATE_SCHEMA)
        .load(f"{LANDING}/opensky/")
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("ingest_date", F.to_date("ingested_at"))
    )


@dp.table(
    name=f"{CATALOG}.{BRONZE}.weather_raw",
    comment="Raw METAR and TAF JSON from aviationweather.gov (append-only).",
    partition_cols=["ingest_date"],
)
def weather_raw():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(WEATHER_SCHEMA)
        .load(f"{LANDING}/weather/")
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("ingest_date", F.to_date("ingested_at"))
    )


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

@dp.table(
    name=f"{CATALOG}.{SILVER}.positions",
    comment="Clean, deduplicated aircraft positions in feet/knots, enriched with aircraft type and operator.",
    cluster_by=["event_date", "icao24"],
)
@dp.expect_or_drop("has_position", "lat IS NOT NULL AND lon IS NOT NULL AND event_ts IS NOT NULL")
@dp.expect_or_drop(
    "inside_box",
    f"lat BETWEEN {BBOX['lamin']} AND {BBOX['lamax']} AND lon BETWEEN {BBOX['lomin']} AND {BBOX['lomax']}",
)
@dp.expect_or_drop("plausible_altitude", "alt_ft IS NULL OR alt_ft BETWEEN -1500 AND 60000")
@dp.expect_or_drop("plausible_speed", "speed_kt IS NULL OR speed_kt < 700")
@dp.expect("has_callsign", "callsign IS NOT NULL AND callsign <> ''")
def positions():
    aircraft = spark.read.table(f"{CATALOG}.{SILVER}.aircraft_ref")
    states = (
        spark.readStream.table(f"{CATALOG}.{BRONZE}.opensky_states")
        .where(F.col("time_position").isNotNull())
        # Stale: OpenSky repeats an aircraft's last position for a while after it stops hearing it.
        .where(F.col("fetched_at") - F.col("time_position") <= 60)
        .select(
            F.lower(F.trim("icao24")).alias("icao24"),
            F.expr("nullif(trim(callsign), '')").alias("callsign"),
            F.timestamp_seconds("time_position").alias("event_ts"),
            F.timestamp_seconds(F.col("fetched_at").cast("long")).alias("fetched_ts"),
            F.col("latitude").alias("lat"),
            F.col("longitude").alias("lon"),
            # Geometric (GNSS) altitude when available: pressure altitude can be
            # several hundred feet off on high- or low-pressure days.
            (F.coalesce("geo_altitude", "baro_altitude") * M_TO_FT).alias("alt_ft"),
            (F.col("baro_altitude") * M_TO_FT).alias("baro_alt_ft"),
            (F.col("velocity") * MS_TO_KT).alias("speed_kt"),
            F.col("true_track").alias("track_deg"),
            (F.col("vertical_rate") * MS_TO_FPM).alias("vr_fpm"),
            "on_ground",
            "squawk",
            "source_file",
        )
        .withWatermark("event_ts", "2 minutes")
        .dropDuplicatesWithinWatermark(["icao24", "event_ts"])
        .withColumn("event_date", F.to_date("event_ts"))
    )
    return states.join(F.broadcast(aircraft), "icao24", "left")


@dp.materialized_view(
    name=f"{CATALOG}.{SILVER}.weather",
    comment="Parsed Heathrow METAR observations and TAF forecasts, one row per report.",
)
def weather():
    raw = spark.read.table(f"{CATALOG}.{BRONZE}.weather_raw")
    j = lambda path: F.get_json_object("raw_json", path)  # noqa: E731
    parsed = raw.select(
        "kind",
        F.timestamp_seconds("obs_time").alias("obs_time"),
        F.timestamp_seconds(F.col("fetched_at").cast("long")).alias("fetched_ts"),
        F.expr("try_cast(get_json_object(raw_json, '$.wdir') AS INT)").alias("wind_dir_deg"),  # null when VRB
        F.expr("try_cast(get_json_object(raw_json, '$.wspd') AS INT)").alias("wind_kt"),
        F.expr("try_cast(get_json_object(raw_json, '$.wgst') AS INT)").alias("gust_kt"),
        # "6+" means 6 statute miles or more
        F.expr("try_cast(replace(get_json_object(raw_json, '$.visib'), '+', '') AS DOUBLE)").alias("visibility_sm"),
        F.expr(
            "array_min(transform(filter(from_json(get_json_object(raw_json, '$.clouds'), "
            "'array<struct<cover:string,base:int>>'), c -> c.cover IN ('BKN', 'OVC', 'OVX')), c -> c.base))"
        ).alias("ceiling_ft"),
        j("$.fltCat").alias("flight_category"),
        F.coalesce(j("$.rawOb"), j("$.rawTAF")).alias("raw_text"),
    )
    latest = Window.partitionBy("kind", "obs_time").orderBy(F.col("fetched_ts").desc())
    return (
        parsed.where(F.col("obs_time").isNotNull())
        .withColumn("rn", F.row_number().over(latest))
        .where("rn = 1")
        .drop("rn")
    )
