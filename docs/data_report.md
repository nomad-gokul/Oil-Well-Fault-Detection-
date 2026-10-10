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

### First observations

- Many variables are entirely missing in a given file (e.g. P-JUS-BS, P-JUS-CKP, PT-P, QBS in
  the real instances inspected). Simulated and drawn instances carry only a handful of
  variables (P-MON-CKP, P-PDG, P-TPT, T-JUS-CKP, T-TPT, and in some cases P-JUS-CKGL, QGL).
- Real instances contain unlabelled periods (`class` and `state` are null, e.g. 3,600 rows,
  one hour, in the instances inspected).
- Classes 1 and 6 have only 4 and 6 real instances, so a test set of held-out real instances
  will be very small for them. Per-class results must be reported with that caveat.
- Class 3 (Severe Slugging) and 4 (Flow Instability) have no transient label.

## 2. Exploratory analysis (Task 3)

Notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), built on the per-instance profile
from `python -m owfd.data.profile` (`data/processed/profile.parquet`, about 70 s for all 2,228 files).

### Key findings

| Topic | Finding |
|---|---|
| Wells | 40 real wells; class 3 and 6 events come from only 2 wells each, class 1 and 5 from 3 |
| Duration | Real: median 5.9 h (1.5 h – 213 h); simulated: median 8.1 h; drawn: median 24 h |
| Sampling | Every file is a clean 1 Hz series: **0 timestamp gaps** in 2,228 files |
| Missing variables | No instance has all 27 variables; P-JUS-BS, P-MON-SDV-P, QBS, PT-P are absent everywhere |
| Frozen sensors | P-PDG is constant over the whole instance in 79% of real instances where present (QGL 67%, T-PDG 59%, ABER-CKGL 83%); never in simulated data |
| Unlabelled data | Every real instance starts with exactly 3,600 unlabelled rows (1 hour); simulated/drawn have none |
| Label order | No instance goes backwards (fault → normal): all follow normal → transient → steady state |
| Transients | All real instances of classes 1, 2, 5, 6, 7, 8 have a transient; classes 3, 4 never do; only 14 of 57 real class-9 instances do |
| Transient length (real, median) | 6: 12 min · 2: 86 min · 5: 153 min · 1: 164 min · 7: 12 h · 8: 48 h |
| Imbalance (real labelled rows) | normal 35%, class 7 27%, class 8 17%, class 4 8.5%, class 9 8.4%; classes 1, 2, 6 < 1% each |

Figures: [instances per well](figures/eda_instances_per_well.png) ·
[durations](figures/eda_durations.png) · [missing by source](figures/eda_missing_by_source.png) ·
[missing in real data by class](figures/eda_missing_real_by_class.png) ·
[fault signatures](figures/eda_fault_signatures.png)

These follow from the numbers above and are applied in Tasks 4 and 5.

1. **Variables: keep 7 core sensors** — `P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, QGL`.
   The first five are present in every simulated and hand-drawn instance and in 50–92% of real
   ones; P-JUS-CKGL and QGL are present in 62% / 40% of real instances. Every other variable is
   either absent from all simulated/drawn data (P-ANULAR, T-PDG, ABER-*) or almost never present
   in real data (T-MON-CKP, P-JUS-CKP, P-JUS-BS, P-MON-SDV-P, P-MON-CKGL, QBS, PT-P). Valve
   states are dropped: they are absent from simulated/drawn data.
2. **Missing values: mask + fill.** Whole-variable absence is the norm (no instance has all 7),
   so each variable gets a binary `missing` channel. Short NaN runs inside a series are
   forward-filled (max 60 s); the rest is filled with the training mean after scaling (i.e. 0).
   There are **no timestamp gaps**: every file is a clean 1 Hz series, so no resampling is needed.
3. **Frozen (stuck) sensors are treated as missing.** P-PDG is constant over the whole instance in
   79% of the real instances where it is present (QGL 67%, T-PDG 59%), but never in simulated
   data. A rule that also works online: a value unchanged for ≥ 30 min is masked as missing.
4. **The first hour of every real instance is unlabelled** (exactly 3,600 rows, always a leading
   block). It is used as history for windows but never as a training target.
5. **Sources: train on real + simulated + drawn; validate and test on real only**, on held-out
   wells. Caveat: classes 3 and 6 come from only 2 real wells and classes 1 and 5 from 3, so a
   strict per-well split cannot put every class in every split; Task 5 must handle this
   explicitly and report which classes are covered.
6. **Shortcut risk.** Missingness differs by source (e.g. QGL is present in 40% of real but 14% of
   simulated instances), so a model could learn "this looks simulated" instead of the fault. The
   real-only test set guards against this; Task 7 adds an ablation without missing masks.
7. **Imbalance.** Of the real labelled rows, normal is 35%, class 7 27% and class 8 17%, while
   classes 1, 2 and 6 are each under 1%. Use macro F1 as the main metric, class-weighted losses,
   and cap windows per instance so multi-day instances (classes 7, 8) do not dominate.
8. **Label caveats.** Classes 3 and 4 have no transient label, so early warning is undefined for
   them (detection only). 43 of the 57 real class-9 instances contain only normal labels; with
   labels taken at the window end these simply become normal windows.
