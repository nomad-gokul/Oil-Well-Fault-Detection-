# Chronic flow instability: per-well profiles from operator labels

Flow instability is 343 of the 482 fault episodes across all 40 wells, and the onset model
catches only 16% of them ([cross_validation.md](cross_validation.md)). This page explains why,
what does work, and how it is built into the live system.

## 1. Across wells it cannot be learned

A dedicated binary detector (unstable vs normal), trained on the other wells and tested on
held-out wells (same 5 folds as the cross-validation), reaches a window **ROC-AUC of 0.54–0.58**,
whatever the feature set (all 237 features, without baseline-relative features, or only the
30/120-minute context). On WELL-00002 and WELL-00010, which supply 201 of the episodes, unstable
periods look exactly like normal ones to it; on WELL-00004 everything, normal included, looks
unstable. Every well is unstable in its own way.

## 2. Within a well it is clear

For each well with both kinds of data, the single best feature separates its unstable windows
from its normal ones with AUC 0.97–1.00 (for example the 120-minute range of the subsea pressure
on WELL-00001, the 30-minute spread of the choke temperature on WELL-00005). Caveat: labels are
per recording, so part of this is "which period", not only "unstable or not". That is why the
evaluation below only ever uses the past.

## 3. Design: a per-well profile learned from labels

Once operators have labelled some of a well's past recordings as unstable or normal, a small
XGBoost model (100 trees, depth 3, all 237 features) is fitted for that well. In the live system
it scores every minute of that well; **P(unstable) > 0.8 for 20 consecutive minutes** opens a
"chronic" Flow Instability alert, alongside the onset model's alerts. The rule was fixed in
advance; every rule tried (0.5/10 min to 0.95/30 min) gives nearly the same results.

A well gets a profile once it has at least 60 labelled windows of each kind; it is refitted
after every 5 newly labelled recordings.

## 4. Evaluation: replaying each well's history in time order

`python -m owfd.models.well_profiles evaluate` replays every well's recordings in time order.
Each recording is scored by a profile trained **only on that well's recordings that ended
before it started** (consecutive 3W recordings overlap by about an hour, so overlapping ones
are excluded). Labels stand in for operator feedback; `--label-frac` keeps only a random share
of them. A well's first episodes, before any profile exists, count as misses. A profile whose
newest training data is more than **180 days** older than the recording is not used (it is
refitted as soon as newer labels exist): a well's normal operation changes over years.

| Profile age limit | Unstable episodes caught | Extra false alarms / day |
|---|---:|---:|
| none | 277 / 343 | 0.34 |
| 365 or **180 days (used)** | 277 / 343 | 0.31 |
| 90 days | 275 / 343 | 0.31 |
| 30 days | 274 / 343 | 0.24 |

Results with the 180-day limit:

| Labels available | Unstable episodes caught | ... once the well has a profile | Continuations | New onsets | Extra false alarms / day (those 7 wells) |
|---|---:|---:|---:|---:|---:|
| Every past recording | **277 / 343 (81%)** | 277 / 299 (93%) | 273 / 329 | 4 / 14 | 0.31 |
| A random 25% of them | 146 / 343 (43%) | 146 / 187 (78%) | 140 / 329 | 6 / 14 | 0.96 |

**Whole system across all 40 wells** (onset model cross-validation + profiles):

| | Fault episodes caught | False alarms / day |
|---|---:|---:|
| Onset model only | 27% (130 / 482) | 2.56 |
| + profiles, every past recording labelled | **78% (378 / 482)** | 2.70 |
| + profiles, 25% labelled | 52% (248 / 482) | 3.00 |

## 5. What this does, and does not, do

- **It tracks a known chronic condition.** 329 of the 343 unstable episodes directly continue
  an unstable recording of the same well: instability in 3W is mostly a regime that lasts days
  or weeks. Once it has been labelled, the profile flags the well while it lasts and when it
  comes back.
- **It does not discover a well's first instability.** Only 4 of 14 new onsets (unstable after
  normal operation) are caught, and a well without labels has no profile at all.
- **It depends on labels.** With a quarter of past recordings labelled, wells get profiles later
  and false alarms rise. The feedback loop below is what makes the labels exist.

## 6. In the live system

- `deploy/bundle/profiles/`: one gzipped model per well with enough labelled history (7 real
  wells), plus a profile for one demo well (slugging) trained only on its source well's
  recordings that ended before the demo recording starts. Demo profiles older than 180 days at
  the demo start are not built (four were skipped, the oldest 1,077 days). The instability demo
  has no profile (its well had no labelled normal data before it); the onset model alerts on it.
- **Age limit at serving:** a real well's profile is only used for data at most 180 days after
  its newest training data (`GET /wells/{id}/profile` shows when it expires). The bundled
  profiles end in 2017–2018, so on live data today they are expired until refreshed with new
  labels, which is the intended behaviour.
- Chronic alerts are stored and shown like other alerts, with `source: "chronic"`, the profile's
  top SHAP drivers, and a badge on the dashboard's fleet view.
- **Feedback loop:** `POST /alerts/{id}/feedback` (confirmed / false_alarm / flow_instability)
  and the dashboard's *Confirm* and *False alarm* buttons store operator verdicts
  (`GET /feedback`); they are the labels the next `python -m owfd.models.well_profiles build`
  learns from.
- `GET /wells/{id}/profile` shows whether a well has a profile and what it was trained on.

Why the age limit exists: in the first version the PCK-restriction demo (WELL-00004, 2017) had a
profile trained on that well's 2014 recordings, and during its labelled-normal first hours it
raised a chronic alert: a false alarm from an old profile misreading a newer operating regime.
With the 180-day limit that profile is not used; the replay above shows the limit costs no
detection and removes false alarms across all wells.

Reproduce:

```bash
python -m owfd.models.well_profiles evaluate               # every past recording labelled
python -m owfd.models.well_profiles evaluate --label-frac 0.25
python -m owfd.models.well_profiles build                  # -> artifacts/well_profiles/<version>
python -m owfd.models.export && make publish-bundle        # profiles go into deploy/bundle
```
