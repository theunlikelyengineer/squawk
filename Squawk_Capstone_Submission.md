# Squawk — Capstone Submission

**Real-time arrival disruption detection, explanation and forecasting for London Heathrow**

Jamie Di Cataldo · DataExpert.io Databricks AI Capstone · 2 October 2026

| | |
|---|---|
| Repository | `https://github.com/theunlikelyengineer/squawk` |
| Deployed application | `https://squawkv2-1352785079224954.aws.databricksapps.com` |
| Unity Catalog | `bootcamp_students.student_jcdc9919_capstone` |
| Lakebase project | `projects/squawk/branches/production/endpoints/primary` |
| Operating period | 2026-09-30 16:24:36Z → 2026-10-02 17:19:00Z (**48.9 hours continuous**) |

---

## 1. What Squawk does

Heathrow runs at close to full runway capacity. When arrival demand exceeds what the
runways can absorb, aircraft are held in one of four stacks — Bovingdon (BNN), Biggin Hill
(BIG), Lambourne (LAM) and Ockham (OCK) — or sent around on final approach. Those events are
visible in public ADS-B telemetry minutes before they appear in any airline's own systems,
but nobody joins them to weather and explains *why* they happened.

Squawk ingests live aircraft positions and Heathrow weather, detects holding episodes and
go-arounds from the raw track geometry, asks an LLM agent to explain each event from the
evidence, queues the explanation for a human analyst to confirm or correct, and measures how
often the agent was right.

The pipeline runs continuously: a poller lands raw JSON, a Lakeflow pipeline builds Bronze
and Silver, a detector writes Gold episodes and a Lakebase work queue, an agent worker
assesses events and issues hourly forecasts, a Databricks App presents the queue to an
analyst, and Lakebase Change Data Feed streams every operational change back into Delta for
analytics.

---

## 2. Architecture

```
adsb.lol / adsb.fi / airplanes.live ─┐
                                     ├─► 01_poller ──► UC Volume (raw JSON)
aviationweather.gov (METAR / TAF) ───┘                      │
                                                            ▼
                                        squawk-etl (Lakeflow, continuous)
                                        Auto Loader ─► Bronze ─► Silver
                                                            │
                                     ┌──────────────────────┴─────────────┐
                                     ▼                                    ▼
                          03_detector (15 s loop)                04_agent_worker (30 s loop)
                          Gold episodes + occupancy              assess events, forecast, score
                                     │                                    │
                                     └────────► Lakebase (Postgres) ◄─────┘
                                                        │
                                   ┌────────────────────┼─────────────────┐
                                   ▼                    ▼                 ▼
                          Databricks App       Lakebase CDF ──► squawk-analytics
                          (Streamlit, 5 tabs)   lb_*_history      4 materialized views
```

Two pipelines rather than one, deliberately. `squawk-etl` streams continuously because ADS-B
data not captured is gone forever. `squawk-analytics` runs on a 15-minute schedule because no
decision depends on a forecast-accuracy figure being seconds fresh, and because a failure in
analytics must never be able to stop ingestion.

---

## 3. Evidence by rubric category

### 3.1 Spark data pipeline (`pipeline/etl.py`)

Bronze is two Auto Loader streams reading JSON-lines files from a Unity Catalog volume, with
an explicit schema (no inference), `_metadata.file_path` captured as lineage, and partitioning
by ingest date. Silver is where the work happens:

- **Null and position handling** — rows without a position or timestamp are dropped by the
  `has_position` expectation.
- **Staleness** — the community ADS-B networks repeat an aircraft's last known position for a
  while after they stop hearing it, so rows where `fetched_at - time_position > 60` are
  discarded before they can create phantom tracks.
- **Deduplication** — `dropDuplicatesWithinWatermark(["icao24", "event_ts"])` under a
  two-minute watermark. At a ~22 second poll interval the same position is reported several
  times; 1,103,270 Bronze rows reduce to 315,211 distinct Silver positions.
- **Unit conversion** — metres to feet, m/s to knots and ft/min. GPS (geometric) altitude is
  preferred over barometric, because pressure altitude drifts by hundreds of feet with the
  weather, which matters for a rule keyed on an 7,000 ft floor.
- **Enrichment** — stream-static join to `aircraft_ref` for type and operator.
- **Validation** — four Lakeflow expectations. Three are `expect_or_drop` (`inside_box`,
  `plausible_altitude`, `plausible_speed`); one is a warn-only `expect` (`has_callsign`).

**Measured result:** Written 100% (5,785 rows in the sampled run), **Dropped 0%**. The only
expectation firing is `has_callsign` at 4.6% (265 rows), which is ALLOW rather than DROP —
those are military and older transponders that broadcast position but no flight identifier.
They are deliberately kept, because every detection rule keys on `icao24`, and excluding them
would under-count stack occupancy.

Zero drops across the three hard rules is the useful signal: it confirms the bounding box
actually covers the polled radius and that the unit conversions agree with reality. A unit
error would have binned thousands of rows immediately.

The pipeline is re-runnable and idempotent: Auto Loader checkpoints file state, the Gold
detector uses `MERGE` on a deterministic event ID, and the Lakebase upsert is
`ON CONFLICT DO UPDATE ... WHERE ended_at < EXCLUDED.ended_at`.

### 3.2 Third-party API integration (`app/squawk_lib/sources.py`)

Two live APIs, no mocked or fabricated data at any point.

**Aircraft positions** come from the community ADS-B networks. The original design used the
OpenSky Network, which was abandoned after diagnosis: OpenSky deliberately blocks hosting and
cloud-provider IP ranges, so every call from Databricks compute timed out. This was confirmed
by testing reachability of several hosts from the same compute — `aviationweather.gov`,
`api.adsb.lol` and `pypi.org` all returned HTTP 200 while `auth.opensky-network.org` timed out,
ruling out a workspace egress restriction.

The replacement is a **multi-provider client with failover**. adsb.lol, adsb.fi and
airplanes.live all serve the same readsb JSON, so `AdsbClient.fetch_records` tries each in turn;
a provider that rate-limits (HTTP 429) or errors is placed on a 120-second cooldown and the next
is used. This was not speculative hardening — the first production run logged 8 of 20 calls as
429, because a shared workspace egress IP shares a rate-limit budget with every other student on
it. After the change, a 20-call sample logged 12×200 from adsb.fi and 6×200 plus 2×429 from
adsb.lol, with **zero lost polling cycles**.

**Weather** comes from the NOAA Aviation Weather Center (METAR and TAF for EGLL), with three
attempts, exponential backoff with jitter, and HTTP 204 handled as "valid request, no data"
rather than an error.

Every call is logged to `bronze.api_call_log` with source, timestamp, status, latency and error,
which is what makes the following measurable over a 30-minute window:

| Metric | Value |
|---|---|
| Position calls | 83 |
| Effective poll interval | 21.7 s |
| Mean latency | 397 ms |
| p95 latency | 926 ms |

Credentials were never placed in source. The LLM key was entered through a notebook widget,
written to a Databricks secret scope, and the widget cleared — the agent now uses workspace
model serving and needs no key at all.

### 3.3 Lakebase data model (`notebooks/02_lakebase_setup.py`)

Four tables in the `squawk` Postgres schema, modelling the operational workflow:

- `disruption_events` — the work queue. UUID primary key (deterministic, see below), a
  `status` enum constrained to `detected / assessed / confirmed / rejected`, `first_detected_at`
  and `updated_at` audit timestamps, a `UNIQUE (icao24, event_type, started_at)` natural key
  preventing duplicates from a second source, and an index on `(status, started_at)` because the
  queue is always read by status.
- `agent_assessments` — identity primary key, FK to the event, CHECK constraints on cause,
  severity (1–3) and confidence (0–1), the reasoning text, a JSONB evidence blob, the model
  version and the MLflow trace id. `UNIQUE (event_id, model_version)` means the same event can be
  assessed once per model-and-prompt version — which is what made the prompt comparison in §3.4
  possible without deleting anything.
- `analyst_reviews` — identity primary key, FKs to event and assessment, verdict enum,
  optional corrected cause (constrained to the same taxonomy), notes, reviewer and timestamp.
  Append-only: a re-review inserts a new row rather than overwriting, so the full decision
  history is preserved and the latest row per event is authoritative.
- `holding_forecasts` — prediction, actuals, absolute error and a `pending / scored` status,
  with `UNIQUE (target_hour, model_version)`.

All four carry `REPLICA IDENTITY FULL`, which is required for Change Data Feed to emit
before-images on update.

**Separation of duties is enforced by Postgres, not by application code.** A `squawk_agent`
role can read everything, insert into `agent_assessments` and `holding_forecasts`, and update
only the `status` and `updated_at` columns of `disruption_events` — a column-level grant. The
agent connects as the workspace identity and immediately runs `SET ROLE squawk_agent`. The
application's service principal has the mirror-image grant: it may insert reviews but not
assessments. Neither may delete anything.

This is demonstrated rather than asserted (`evidence/guardrail.png`):

```
REFUSED  insert an analyst review:   permission denied for table analyst_reviews
REFUSED  delete a disruption event:  permission denied for table disruption_events
REFUSED  change an event's callsign: permission denied for column callsign
ALLOWED  update an event's status
```

The agent cannot fabricate human sign-off even if the model decides to try.

### 3.4 Action-taking AI agent (`app/squawk_lib/agent.py`)

A LangGraph tool-calling agent on Databricks model serving —
`databricks-claude-haiku-4-5` for assessments, `databricks-claude-sonnet-4-5` for forecasts
and chat.

**Five read tools:** `list_open_events`, `get_flight_track`, `get_weather`,
`get_stack_occupancy`, `get_forecast_accuracy`. These span both stores — Delta tables for
tracks, weather and occupancy; Lakebase for the queue and past accuracy — through an injected
`sql_fn`, so the identical tool set works against Spark in a notebook and a SQL warehouse in
the deployed app.

**Two write tools:** `save_assessment` writes an assessment and moves the event to `assessed`
in one transaction; `save_forecast` writes an hourly prediction. Both validate their inputs
(cause against the taxonomy, severity and confidence against range, event id against a UUID
pattern) before touching the database, both are `ON CONFLICT DO NOTHING` so a retry cannot
duplicate, and both execute as the restricted role.

The worker also closes the loop: `score_finished` fills in the actual holding for forecasts
whose target hour has elapsed, computes absolute error and marks them `scored`.

Every run is traced to MLflow (`/Users/…/squawk-agent`), so each tool call, its arguments and
its return value are inspectable (`evidence/mlflow_trace.png`).

**Agent quality, measured.** An example of the agent working well — the synthetic fixture
(§3.8), assessed under `prompt-v2`:

> SQWK01 held at BNN for 15 min 40 sec in VFR conditions (visibility 6 SM, wind 240° at
> 12 kt). All four stacks showed zero occupancy during the event window. No weather, traffic
> volume, or runway change evidence supports this hold; **insufficient data to determine root
> cause.** — cause `other`, severity 1, **confidence 0.40**

Calibrated abstention: there was no real cause because there was no real aircraft, and the
agent declined to invent one and lowered its confidence accordingly.

An example of the agent working badly, and how it was fixed. Under `prompt-v1`, a go-around by
CBJ431 on runway 27R was explained as a *runway change to 09L*. That is wrong for three
reasons: 09L and 27R are **the same physical runway** used in opposite directions, so the
"switch" was a geometric artefact of the aircraft continuing west past the airfield; the METAR
at the time read `21008KT 180V250`, a southwesterly supporting westerly operations; and a
runway direction change is an airport-wide event that would show in other arrivals, not one
aircraft's own track. The analyst review corrected it to `other`.

The fix was to encode Heathrow's runway geometry and the wind-to-direction relationship into
the assessment prompt, and to version it (`PROMPT_VERSION = "prompt-v2"`). Because
`agent_assessments` is unique on `(event_id, model_version)`, both generations coexist and the
difference is measurable:

| Prompt version | Events reviewed | Agreement |
|---|---|---|
| `prompt-v1` | 1 | 0 / 1 |
| `prompt-v2` | 3 | 3 / 3 |

### 3.5 Analytics pipeline (`pipeline/analytics.py`)

Lakebase Change Data Feed is enabled on all four tables, streaming into
`lb_<table>_history` Delta tables in Unity Catalog. All four report `CDF_STATE_STREAMING`.

A separate Lakeflow pipeline, `squawk-analytics`, builds four materialized views from that
change history — never from the operational database directly:

- `event_lifecycle` — one row per event, reconstructing when it was detected, assessed and
  reviewed from the change stream, with derived latencies.
- `agent_agreement` — how often analysts agreed with the agent, by event type, cause and
  model version, with `agreement_rate` and `detection_precision`.
- `forecast_accuracy` — each scored forecast alongside a naive persistence baseline.
- `daily_summary` — daily KPIs for the application's Analytics tab.

Rebuilding current state from a change log is handled by a `latest_rows` helper that takes the
highest `_sort_by` per key and discards deletes.

**Measured results over the operating period:**

| Metric | Value |
|---|---|
| Events detected | 4 (3 go-arounds, 1 synthetic holding fixture) |
| Events reviewed | 4 of 4 |
| Agreement rate | 3 / 4 (75%) |
| Detection precision | 1.00 (no false positives) |
| Scored forecasts | 7 |
| Agent forecast MAE | 0.71 min |
| Naive baseline MAE | 0.00 min |

### 3.6 Frontend and core workflow (`app/app.py`)

A five-tab Streamlit application. **Live map** shows current traffic on a pydeck map with the
four stack rings drawn and aircraft currently holding highlighted. **Event queue** is the core
workflow: a status summary, a selectable time window, a table of events, and a detail panel
showing the altitude profile, the METARs around the event, the agent's cause, severity,
confidence, reasoning and MLflow trace, and the review controls. **Forecast** shows predictions
against actuals. **Ask Squawk** is a chat agent with the read-only tool subset — it can look
anything up and cannot change data. **Analytics** surfaces the materialized views.

The write path is real: submitting a review inserts into `analyst_reviews` and transitions the
event to `confirmed` or `rejected` in a single transaction, the reviewer is captured from the
forwarded authentication header rather than typed, and the new status is reflected immediately.
Empty states are handled explicitly (no events in window, no forecasts yet, agent not yet
assessed).

### 3.7 Deployed application

Deployed as a Databricks App at
`https://squawkv2-1352785079224954.aws.databricksapps.com`, not run locally.

Configuration is declarative in `app.yaml`, with four resources bound by key: a SQL warehouse
(`Can use`), the Lakebase database (`Can connect and create`), and the two model serving
endpoints (`Can query`). The app authenticates as its own service principal, which holds
`USE CATALOG` / `USE SCHEMA` / `SELECT` in Unity Catalog and the narrow Lakebase grants
described in §3.3. No API keys are involved, because the agent runs on workspace model serving.

### 3.8 Big data — volume and variety

**Volume (demonstrated).** 1,103,270 rows in `bronze.opensky_states` and 315,211 in
`silver.positions` over 48.9 hours, processed by a distributed Spark pipeline, partitioned by
ingest date, with Silver liquid-clustered on `(event_date, icao24)` — the access pattern the
detector uses when it reads a three-hour window per aircraft every 15 seconds.

**Variety (demonstrated).** METAR and TAF are unstructured coded text. Squawk parses them into
structured fields (wind direction and speed, gusts, visibility, ceiling, flight category),
retains the original `raw_text`, surfaces that raw text in the application next to each event,
and feeds it to the agent as reasoning evidence. The agent's explanations quote specific values
from it, as the examples in §3.4 show — this is unstructured data meaningfully transformed and
used in the workflow, not merely stored.

**Velocity (measured, not claimed).** Ingestion-to-queryable latency, sampled 30 times at
10-second intervals:

| min | median | p95 | max | samples under 60 s |
|---|---|---|---|---|
| 22 s | 87 s | 163 s | 165 s | 10 / 30 |

This does **not** meet the sub-minute bar and is not claimed as a third V. The remaining
latency decomposes into a 21.7-second poll interval, a 10-second streaming trigger interval,
and a Delta commit cadence of 20–140 seconds set by micro-batch duration — the stream-static
join, watermarked deduplication and clustered write. Reducing it further would mean smaller
micro-batches via `cloudFiles.maxFilesPerTrigger`, which pays the same fixed overhead more
often and can make matters worse; it was not attempted.

For context, this figure was **650–800 seconds** earlier the same day. Two causes were found:
a scheduler job that had failed on 28 consecutive runs while the system appeared healthy, and
an unset `pipelines.trigger.interval` defaulting to ten minutes for flows with non-Delta
sources. Both are described in §5.

---

## 4. Detection method and threshold validation

Holding is detected geometrically rather than by any published feed. For each aircraft the
detector computes distance to the four stack fixes, filters to positions within 12 NM and above
7,000 ft, and accumulates **signed heading change over an 8-minute rolling window**, ignoring
turns across data gaps longer than 120 seconds. A standard holding pattern is a racetrack with
two same-direction 180° turns per circuit, so one circuit accumulates 360°; the threshold is
set there. Go-arounds are detected as a descent below 1,500 ft within 6 NM of a runway
threshold followed by a climb through 2,500 ft at over 1,000 ft/min within five minutes.

Episode IDs are a deterministic hash of aircraft, type and start time, so re-detecting the same
ongoing episode every 15 seconds upserts rather than duplicates. (An earlier version did
duplicate: as an episode's start time slid out of the detector's three-hour lookback, a fresh ID
was generated each cycle, producing ~39 events for a single hold. The fix was a 30-minute margin
at the window edge, covered by a regression test that proves 11 IDs collapse to 1.)

**The thresholds were validated against real data rather than assumed.** A diagnostic
(`notebooks/99_threshold_probe.py`) examined every aircraft that entered a stack zone over a
12-hour window: **1,210 aircraft**, of which the maximum accumulated turn was **235°**, against
a 360° threshold. The distribution has a hard ceiling in the 195–235° band, which is the
signature of ordinary arrival geometry — a base turn plus a final turn in the same direction.
Nothing sits between 235° and 360°.

Two conclusions follow. The threshold has a clear margin over normal traffic and is not firing
spuriously. And **no stack holding occurred at Heathrow during the operating period** — an
absence that is measured, not assumed.

That is consistent with how Heathrow is run today: time-based separation and cross-border
arrival management absorb delay en route rather than over London, so stack holding has become a
bad-weather and peak-disruption phenomenon rather than a daily one. The TAF for the final day
(`21010KT 9999 FEW035`) confirms settled conditions throughout.

---

## 5. Honest disclosures

**Synthetic traffic was injected to validate the holding path.** With no real holding occurring,
the detection → queue → agent → review path for holding episodes could not otherwise be
exercised. `notebooks/90_inject_test_hold.py` writes landing files **in the poller's own format**
for one aircraft flying a textbook holding pattern over BNN. Only the sensor input is synthetic:
Auto Loader, Bronze, Silver, the detector, Lakebase, the agent and the app all ran on it
unmodified. The aircraft uses ICAO24 `ffff01`, in the reserved `ffff00–ffffff` range never
issued to a real airframe, so synthetic rows remain separable from real traffic permanently
(`WHERE icao24 NOT LIKE 'ffff%'`). The fixture contributes **48 of 1,103,270 positions** and 1
of 4 events. All three go-arounds are real aircraft.

**The forecast lost to its baseline.** Agent MAE 0.71 minutes against a naive persistence MAE of
0.00. The reason is structural: actual holding was zero in every scored hour, so a
"same as last hour" baseline is exactly right every time and cannot be beaten. The agent
predicted small positive values, which reveals a mild bias toward forecasting disruption when
asked to forecast disruption. A meaningful comparison requires a period with variance in the
target.

**Detection latency is inflated.** `event_lifecycle` reports a mean of 2,739 seconds for
go-arounds. This includes a period when the detector task was not running: a cell written for
interactive use called `display()` on an empty DataFrame, which raised `CANNOT_INFER_EMPTY_SCHEMA`
and failed the task on every job run — a textbook "works in a notebook, fails in production"
defect. Those events were found on catch-up rather than in near-real-time. Steady-state detection
latency is the 15-second detector cycle plus pipeline latency.

**A scheduler job failed 28 consecutive times while the system looked healthy.** A wrapper job
created by the pipeline scheduling UI had been failing in 2 seconds on every run for over a day,
because it was trying to start an update on a pipeline that was already running continuously.
Nothing surfaced it. This is the clearest gap in the project's observability and the first thing
to fix.

**Review records exceed events.** `analyst_reviews` contains 9 rows across 4 events, because
the table is append-only by design and some verdicts were revised during review. `agent_agreement`
uses the latest review per event, giving 4.

**Minor:** the METAR parser converts the coded value `9999`, meaning "10 km or more", to its
statute-mile equivalent of 6.2, so the agent reports a bounded number for an open-ended value.
The sample size — 4 events over 48.9 hours — is small, and every rate derived from it carries
correspondingly wide uncertainty.

---

## 6. What I would do next

Reduce ingestion latency below one minute by profiling the Silver micro-batch, which is the
binding constraint rather than ingestion or trigger interval. Add alerting on job failure and
on Silver freshness, so a dead scheduler cannot hide for a day. Collect through a period of
genuine disruption — wind or low visibility — so that the holding path, the cause taxonomy
beyond `other`, and the forecast comparison are all exercised against real variance. And expand
the review set substantially, since every agreement statistic here rests on four events.

---

## Appendix — evidence index

| Evidence | File |
|---|---|
| Scale — 1,103,270 Bronze / 315,211 Silver rows | `evidence/scales_1m_rows.png` |
| Guardrail proof — Postgres refuses the agent's writes | `evidence/guardrail.png` |
| MLflow agent trace — tool calls, arguments, returns | `evidence/mlflow_trace.png` |
| Agent quality — agreement by prompt version, forecast vs baseline | `evidence/agent_quality.png` |
| Synthetic fixture — injected holding pattern detected end to end | `evidence/synthetic_fixture.png` |

The live application at the URL above is itself verifiable evidence for the frontend, the
deployment, the Lakebase read and write path and the agent's read-only chat tools.
