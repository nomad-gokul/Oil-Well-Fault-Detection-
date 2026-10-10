# 3W Dataset Report

## 1. Dataset structure (Task 2)

Source: [petrobras/3W](https://github.com/petrobras/3W), dataset version **2.0.0** (`dataset/dataset.ini`),
licensed CC BY 4.0. Fetched with `python -m owfd.data.download` (sparse, shallow clone of the
`dataset/` folder only: 2,228 files, about 1.8 GB on disk).

| Property | Value |
|---|---|
| File format | Parquet (pyarrow engine, brotli compression), one file per instance |
| Folder structure | `dataset/<event class 0..9>/<instance>.parquet` |
| Real instances | `WELL-<5-digit well id>_<YYYYMMDDhhmmss start>.parquet`, 1,119 files from 40 wells |
| Simulated instances | `SIMULATED_<5-digit n>.parquet`, 1,089 files |
| Hand-drawn instances | `DRAWN_<5-digit n>.parquet`, 20 files |
| Index | `timestamp` (`datetime64[ns]`), stored as the pandas index |
| Sampling rate | 1 Hz (1 s between consecutive rows in every file inspected) |
| Columns | 27 variables (18 continuous sensors + 9 valve states) + `class` + `state` |
| Labels | `class` (int16, nullable): 0 = normal, 1..9 = steady-state event, 101..109 = transient of that event |

### Variables

| Group | Columns |
|---|---|
| Pressures [Pa] | P-ANULAR, P-JUS-BS, P-JUS-CKGL, P-JUS-CKP, P-MON-CKGL, P-MON-CKP, P-MON-SDV-P, P-PDG, PT-P, P-TPT |
| Temperatures [degC] | T-JUS-CKP, T-MON-CKP, T-PDG, T-TPT |
| Flow rates [m3/s] | QBS, QGL |
| Choke openings [%] | ABER-CKGL, ABER-CKP |
| Valve states [0, 0.5, 1] | ESTADO-DHSV, -M1, -M2, -PXO, -SDV-GL, -SDV-P, -W1, -W2, -XO |

### Instances per event class and source

| Class | Event | Real | Simulated | Drawn | Total |
|---|---|---:|---:|---:|---:|
| 0 | Normal Operation | 594 | 0 | 0 | 594 |
| 1 | Abrupt Increase of BSW | 4 | 114 | 10 | 128 |
| 2 | Spurious Closure of DHSV | 22 | 16 | 0 | 38 |
| 3 | Severe Slugging | 32 | 74 | 0 | 106 |
| 4 | Flow Instability | 343 | 0 | 0 | 343 |
| 5 | Rapid Productivity Loss | 11 | 439 | 0 | 450 |
| 6 | Quick Restriction in PCK | 6 | 215 | 0 | 221 |
| 7 | Scaling in PCK | 36 | 0 | 10 | 46 |
| 8 | Hydrate in Production Line | 14 | 81 | 0 | 95 |
| 9 | Hydrate in Service Line | 57 | 150 | 0 | 207 |
| | **Total** | **1,119** | **1,089** | **20** | **2,228** |

### First observations (to be analysed in Task 3)

- Many variables are entirely missing in a given file (e.g. P-JUS-BS, P-JUS-CKP, PT-P, QBS in
  the real instances inspected). Simulated and drawn instances carry only a handful of
  variables (P-MON-CKP, P-PDG, P-TPT, T-JUS-CKP, T-TPT, and in some cases P-JUS-CKGL, QGL).
- Real instances contain unlabelled periods (`class` and `state` are null, e.g. 3,600 rows,
  one hour, in the instances inspected).
- Classes 1 and 6 have only 4 and 6 real instances, so a test set of held-out real instances
  will be very small for them. Per-class results must be reported with that caveat.
- Class 3 (Severe Slugging) and 4 (Flow Instability) have no transient label.
