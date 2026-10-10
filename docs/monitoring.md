# Monitoring (Task 21)

## Metrics — `GET /metrics` (Prometheus)

| Metric | Meaning |
|---|---|
| `owfd_http_requests_total{method,path,status}` | requests per endpoint (path templates, e.g. `/alerts/{alert_id}/ack`) |
| `owfd_http_request_seconds{method,path}` | latency histogram per endpoint |
| `owfd_predictions_total{class_name}` | predictions per predicted class |
| `owfd_alerts_opened_total{class_name,severity}` | alerts opened |
| `owfd_wells_streaming` | wells with a live streaming state |
| `owfd_drift_max_share_unusual` | largest share of wells unusual on one sensor at the last drift check |
| `owfd_model_info{version,model}` | serving bundle in use |

Logs are JSON lines on stdout (`request_id`, method, path, status, latency), level from
`OWFD_LOG_LEVEL` (default `INFO`). Every response carries `X-Request-ID`.

## Fleet data drift — `GET /monitoring/drift`

For every streaming well and sensor, the last 2 h of **deviations from the well's own
baseline** are summarised by their median (offset) and log inter-quartile range (spread). These
are compared with the range seen on 2-hour windows of real training and validation wells
(stored in the bundle's `reference.json` at export). Drift is flagged on a sensor when **at
least half of the wells (and at least 3) are outside that range** — a fleet-wide change such as
a recalibrated sensor, a unit change in the data pipeline, or a firmware update. A single faulty
well does not count as drift; that is what alerts are for.

`python scripts/drift_report.py --api <url>` writes an HTML report and exits with code 1 on
drift (usable as a scheduled job). The dashboard's **System** tab shows the same table.

### Why not PSI?

The first version used the Population Stability Index of the live values against the pooled
training distribution. On this data it does not work: wells differ so much that a single
normal well's 2-hour window already reaches the PSI maximum (about 8.3 with 10 bins), so
normal and shifted data look the same, and thresholds calibrated on unseen wells (95th
percentile 4.9–8.3) sit at that ceiling. The window-statistics design above was adopted instead.

### Verification on real data (demo wells from held-out test wells)

| Scenario | Fleet drift | Detail |
|---|---|---|
| 10 demo wells, as recorded | **no** | at most 1 of 10 wells unusual on any sensor |
| + 8 wells whose P-TPT is suddenly reported in bar instead of Pa | **yes, on P-TPT only** | 9 of 18 wells unusual on P-TPT; other sensors stable |

Limitation: shifts smaller than the natural well-to-well variation (e.g. a few bar on P-TPT)
are not detectable this way.

## Uptime monitoring

Point a free uptime monitor (e.g. UptimeRobot or Better Stack) at `GET /health` of the public
API every 5 minutes; it returns 200 with the model version when healthy. On free hosting tiers
this also keeps the API from sleeping.
