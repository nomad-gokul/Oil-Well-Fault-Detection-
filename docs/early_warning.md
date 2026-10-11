# Early-warning evaluation

`python -m owfd.models.evaluate` replays every held-out instance minute by minute, the way the
live system sees it, and scores **alerts**, not individual windows.

- **Alert rule:** P(fault) = 1 - P(normal) above a threshold for *k* consecutive minutes.
- **Per-well threshold (optional):** the first hour of each well's stream (unlabelled in 3W, and
  the same hour that defines the well's baseline) is a burn-in. Afterwards the well's threshold
  is `max(threshold, min(quantile(burn-in P(fault), q) + margin, 0.99))`, so a well whose model
  output is already high while it runs normally needs stronger evidence. Implemented the same
  way in the live alert engine.
- **Detected:** the alert is on during at least one fault-labelled minute of the instance.
- **Delay:** first alerting minute minus the first fault minute (transient or steady state).
- **Lead time:** start of the steady-state fault minus the first alerting minute
  (positive = warned before the fault fully developed; only for instances with a transient).
- **False alarms:** alerts that start on normal-labelled minutes, per 24 h of normal operation.

Probabilities are calibrated with temperature scaling fitted on validation. The model and the
alert rule (threshold, *k*, burn-in quantile and margin) are chosen **without the test wells**:
the highest detection rate with at most 1 false alarm per 24 h on the
[cross-validation](cross_validation.md) predictions of the 30 wells outside the test split
(`--select-on cv`). The first version of this project chose on the 7 validation wells only; those
choices did not transfer to the test wells, so selection now uses 30 wells. Test numbers are
reported once.

## Model comparison

Features: *window* = 125 statistics of the last 5 minutes (previous production model);
*context* = window + 112 statistics of the last 30 and 120 minutes (level, spread, trend,
oscillation: see `owfd.features.context`); *oscillation only* = window + the 56 oscillation
statistics of the context set.

| | window | window + per-well | oscillation only | **context** | **context + per-well (chosen)** |
|---|---:|---:|---:|---:|---:|
| Rule (threshold, k) | 0.90, 2 | 0.90, 2 | 0.85, 5 | 0.90, 2 | **0.85, 5; q 0.9 + 0.1** |
| Selection set (30 wells): detected / false alarms per day | 13% / 0.74 | 13% / 0.74 | 13% / 0.95 | 15% / 0.95 | **18% / 0.99** |
| Validation (7 wells): detected / false alarms per day | 21% / 1.06 | 21% / 1.06 | 16% / 1.14 | 14% / 2.20 | 16% / 1.23 |
| **Test (10 wells, 126 episodes): detected** | 57% | 57% | 55% | 52% | **52%** |
| Test: median delay after onset | 1 min | 1 min | 4 min | 1 min | 4 min |
| Test: median lead before steady state | 114 min | 114 min | 199 min | 113 min | **307 min** |
| **Test: false alarms per 24 h** | 5.3 | 5.3 | 1.7 | 2.6 | **1.8** |
| Window macro F1 (val / test) | 0.453 / 0.433 | | 0.488 / 0.463 | 0.498 / 0.464 | 0.498 / 0.464 |
| Calibration error, test | 0.110 | | 0.097 | 0.070 | 0.070 |

At a fixed false-alarm budget the context model is the better detector on the test wells
(e.g. 50% vs 46% at 1 false alarm per day, 71% vs 64% at 5) and across all 40 wells (31% vs
25% at 1 per day, 43% vs 35% at 3; see [cross_validation.md](cross_validation.md)). The
previous production model (window features, rule chosen on validation: 0.85, 5) gave 56% at
2.9 false alarms per day on the test wells; the new one gives 52% at 1.8 per day.

**Chosen: context features + per-well threshold** (`configs/serving.yaml`, `deploy/bundle/`).

## Production model per event class (test wells)

| Class | Instances | Detected | Median lead (min) | Median delay (min) |
|---|---:|---:|---:|---:|
| 1 Abrupt Increase of BSW | 2 | 0% | – | – |
| 2 Spurious Closure of DHSV | 3 | **100%** | 67 | 7 |
| 3 Severe Slugging | 31 | 10% | – (no transient) | 196 |
| 4 Flow Instability | 64 | 67% | – (no transient) | 4 |
| 5 Rapid Productivity Loss | 6 | 33% | 1,215 | 302 |
| 6 Quick Restriction in PCK | 3 | 33% | -2 | 14 |
| 7 Scaling in PCK | 10 | **100%** | 671 | 154 |
| 8 Hydrate in Production Line | 2 | **100%** | 1,666 | 34 |
| 9 Hydrate in Service Line | 5 | 40% | – | 303 |

Lead times are long for slowly developing faults (scaling and hydrates develop over a day or
more), and minutes for fast ones.

## Honest caveats

- **The test split is easier than average.** Across all 40 wells (cross-validation) about one
  in four fault episodes is detected at about one false alarm per day, mostly because of flow
  instability on two chronically unstable wells; excluding flow instability, 55%; excluding flow
  instability and severe slugging, 66% (test wells: 65%, 20 of 31). Details and
  95% intervals: [cross_validation.md](cross_validation.md).
- **Per-well thresholds are not a detection lever.** They were chosen here because they gave the
  best selection-set score, but in cross-validation they did not raise detection on average.
- **Severe slugging** is learned almost entirely from simulations (31 of 32 real episodes come
  from one well, which is in the test split) and is mostly missed.
- False-alarm rates differ between wells; rare classes (1, 2, 6, 8) have 2–3 test episodes each.

Figures: [operating points](figures/alert_tradeoff.png) ·
[reliability after calibration](figures/reliability.png). Tables:
`docs/early_warning_comparison.csv`, `docs/early_warning_per_class.csv`.
