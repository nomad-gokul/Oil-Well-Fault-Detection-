# Early-warning evaluation (Task 10)

`python -m owfd.models.evaluate` replays every held-out instance minute by minute, the way the
live system sees it, and scores **alerts**, not individual windows.

- **Alert rule:** P(fault) = 1 - P(normal) above a threshold for *k* consecutive minutes.
- **Detected:** the alert is on during at least one fault-labelled minute of the instance.
- **Delay:** first alerting minute minus the first fault minute (transient or steady state).
- **Lead time:** start of the steady-state fault minus the first alerting minute
  (positive = warned before the fault fully developed; only for instances with a transient).
- **False alarms:** alerts that start on normal-labelled minutes, per 24 h of normal operation.

Probabilities are calibrated with temperature scaling fitted on validation. The threshold and
*k* are tuned on **validation**: the highest detection rate with at most 1 false alarm per 24 h.
The production model is also chosen on validation; test numbers are reported once.

## Model comparison

| | XGBoost | CNN | Ensemble (mean) |
|---|---:|---:|---:|
| Val macro F1 (windows) | **0.440** | 0.075 | 0.350 |
| Test macro F1 (windows) | **0.446** | 0.218 | 0.413 |
| Calibration error, test (ECE, raw → calibrated) | 0.127 → **0.088** | 0.399 → 0.242 | 0.145 → 0.125 |
| Tuned rule (threshold, k) | 0.80, 5 | 0.90, 1 | 0.90, 5 |
| Val: faults detected / false alarms per day | 21% / 0.92 | 0% / 0.00 | 14% / 0.57 |
| **Test: faults detected** (126 instances) | **57%** | 3% | 44% |
| Test: median delay after fault onset | **4 min** | 424 min | 4 min |
| Test: median lead time before steady state | 559 min | 84 min | 854 min |
| Test: false alarms per 24 h of normal operation | 3.3 | 0.05 | 2.1 |

**Chosen: XGBoost** (`configs/serving.yaml`): best validation detection within the false-alarm
budget, and best macro F1. The ensemble does not beat XGBoost alone because the CNN adds noise.

## XGBoost per event class (test wells)

| Class | Instances | Detected | Median lead (min) | Median delay (min) |
|---|---:|---:|---:|---:|
| 1 Abrupt Increase of BSW | 2 | 50% | 379 | 272 |
| 2 Spurious Closure of DHSV | 3 | **100%** | 67 | 5 |
| 3 Severe Slugging | 31 | 23% | – (no transient) | 84 |
| 4 Flow Instability | 64 | 67% | – (no transient) | 4 |
| 5 Rapid Productivity Loss | 6 | 33% | 1,516 | 2 |
| 6 Quick Restriction in PCK | 3 | 33% | 12 | 0 |
| 7 Scaling in PCK | 10 | **100%** | 1,372 | 162 |
| 8 Hydrate in Production Line | 2 | **100%** | 1,669 | 31 |
| 9 Hydrate in Service Line | 5 | 60% | -3 | 71 |

Lead times are long for slowly developing faults (scaling and hydrates develop over a day or
more), and minutes for fast ones (DHSV closure: alert 5 min after onset, about an hour before the
steady state).

## Honest caveats

- **False alarms do not transfer perfectly across wells.** The rule was held to 1 false alarm per
  day on the validation wells but gives 3.3 per day on the test wells (959 h of normal
  operation). In production the threshold should be re-tuned per well or per field after a
  burn-in period; [the trade-off plot](figures/alert_tradeoff.png) shows the available
  operating points (e.g. lower thresholds detect 60–90% of faults at 10–25 false alarms/day).
- Validation has only 58 fault instances from 7 wells, so the tuned threshold is itself uncertain.
- Severe slugging is learned from simulations only and is mostly missed on the real slugging
  well (sim-to-real gap, see `docs/results.md`).
- Rare classes (1, 2, 6, 8) have 2–3 test instances each.

Figures: [operating points](figures/alert_tradeoff.png) ·
[reliability after calibration](figures/reliability.png). Tables:
`docs/early_warning_comparison.csv`, `docs/early_warning_per_class.csv`.
