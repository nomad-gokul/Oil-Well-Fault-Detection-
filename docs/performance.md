# API performance (Task 15)

Measured locally on 4 CPU cores with the production bundle (XGBoost, 1,000 trees x 10 classes),
SQLite, one uvicorn worker. Load generator: `tests/load/locustfile.py` (Locust), running on the
**same 4 cores** as the API, so these are pessimistic.

    make serve
    OWFD_RATE_LIMIT_PER_MIN=0 make serve   # the default limit (600 POST/min per IP) would throttle the test
    locust -f tests/load/locustfile.py --headless -u 60 -r 20 -t 60s --host http://localhost:8000

## Single requests (no load)

| Endpoint | Latency |
|---|---:|
| `GET /health` | 2 ms |
| `POST /predict` (300 readings, features + model + SHAP drivers) | **43 ms** |
| Streaming update of one well with one reading (in-process) | 4.0 ms |

### With the long-context model (237 features)

Measured against the Docker image (`Dockerfile`, 1 worker) after the switch to the context
model: `POST /predict` 28 ms median (p95 39 ms, 30 requests). A streaming update with 150 new
readings (one demo tick) takes about 9 ms per well in process; the demo (10 wells at 30x)
keeps the container near 0.2% of one core between ticks, with 178 MB of memory.

## 60 concurrent users for 60 s (48 wells streaming 1 reading/s, 12 analysts)

| Endpoint | Requests | Failures | Median | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| `POST /ingest` (1 reading) | 2,832 | 0 | 250 ms | 400 ms | 470 ms |
| `POST /predict` (300 readings) | 480 | 0 | 49 ms | 330 ms | 390 ms |
| `GET /alerts?status=open` | 163 | 0 | 8 ms | 300 ms | 380 ms |
| **All** | **3,475** | **0** | 240 ms | 400 ms | 470 ms |

All 48 wells are kept up with in real time (47 ingest requests/s; every request finishes well
within its 1 s period).

## Bottleneck found and fixed

The first run had `/ingest` at 610 ms median and `/predict` at 1.3 s p95. Profiling showed
most of the time in the stuck-sensor rule, which used a pandas group-by over the whole
36-minute buffer for each of the 7 sensors on every reading. Rewritten in NumPy
(`owfd.data.clean._stuck` / `_ffill`): **3.2x faster** per reading (12.8 → 4.0 ms), verified
identical to the pandas version on random data and byte-identical to the stored training data
on real instances.

## Scaling notes

- Ingestion is serialised per process (one lock around the streaming state), so throughput
  scales with batching: clients that send 10–60 s of readings per request cut the per-request
  overhead proportionally. For many more wells, run several workers with wells partitioned
  across them (the streaming state is per well).
- Under concurrent load `/predict` p95 (330 ms) is above the 100 ms single-request target
  because CPU is shared with ingestion and the load generator; without load it is 43 ms.
