# Cross-validation by well

The test split has 10 wells and the validation split 7, so their numbers depend a lot on
*which* wells landed in them. `python -m owfd.models.cv` measures the whole procedure on all
40 real wells:

- Wells are split into 5 folds (8 wells each, balanced by number of fault episodes).
- For each fold, XGBoost (same hyperparameters and tree count as the production run) is trained
  on the simulated and hand-drawn instances plus the real wells of the other 4 folds, and
  predicts every minute of the held-out wells, including their burn-in hour.
- The calibration temperature and the alert rule are tuned on the other folds' out-of-fold
  predictions (budget: 1 false alarm per 24 h) and applied to the held-out fold (nested).
- 95% intervals: bootstrap over wells (1,000 resamples).

482 fault episodes, 4,053 h of normal operation, 40 wells.

## Results

| Features | Alert rule | Episodes detected (95% CI) | False alarms / 24 h (95% CI) | Median delay (min) | Median lead before steady state (min) |
|---|---|---|---|---:|---:|
| 5-minute window (125, previous model) | global threshold | 24% (10–54%) | 1.4 (0.3–2.4) | 6 | 184 |
| 5-minute window | per-well threshold | 16% (8–37%) | 1.2 (0.2–2.1) | 32 | 184 |
| + oscillation context only (181) | global threshold | 24% (10–54%) | 1.1 (0.3–1.9) | 6 | 148 |
| **+ full 30/120-minute context (237, new model)** | global threshold | 27% (12–58%) | 2.6 (0.3–5.3) | 7 | 173 |
| + full 30/120-minute context | per-well threshold | 27% (12–58%) | 2.6 (0.3–5.3) | 7 | 173 |

The nested procedure lands each model at a slightly different false-alarm rate, so compare
them at *equal* false-alarm budgets (pooled out-of-fold predictions, best rule per budget;
analysis only, not used for selection):

| Detected at ≤ N false alarms / 24 h | 0.5 | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| 5-minute window features | 23% | 25% | 30% | 35% | 41% | 64% |
| + oscillation context only | 23% | 25% | 28% | 33% | 44% | 60% |
| **+ full 30/120-minute context** | **26%** | **31%** | **37%** | **43%** | **49%** | 61% |

**Long-context features help:** +6 to +8 points of detection at every budget from 1 to 5 false
alarms per day. The oscillation statistics alone do not; the gain needs the level, range and
trend over 30 and 120 minutes as well.

**Per-well thresholds do not help** on average: tuned inside the folds they either change
nothing or trade detection for fewer false alarms. They remain available (the production
rule, chosen on the 30 non-test wells, enables them) but are not a lever for detection.

## What the honest number looks like, and why

Across all 40 wells only about **one in four** fault episodes is detected at about one false
alarm per day, well below the 52–56% measured on the 10 test wells. The test split was easier
than average. The difference is almost entirely **flow instability**:

| Event class | Episodes (40 wells) | Detected (new model, CV) |
|---|---:|---:|
| Flow instability | **343** | 16% |
| Scaling in PCK | 36 | 61% |
| Severe slugging | 32 | 16% |
| Spurious closure of DHSV | 22 | 100% |
| Hydrate in production line | 14 | 79% |
| Hydrate in service line | 14 | 57% |
| Rapid productivity loss | 11 | 73% |
| Quick restriction in PCK | 6 | 0% |
| Abrupt increase of BSW | 4 | 0% |

- Flow instability is 71% of all fault episodes, and two wells contribute 201 of them
  (WELL-00002: 117, WELL-00010: 84) with almost none detected. These instances have no
  transient: the well is unstable from the first minute of the recording, so "deviation from
  this well's normal" has nothing to compare against, and an unstable well looks like its own
  normal.
- **Excluding flow instability**, 55% of the remaining 139 episodes are detected (previous
  model: 48%).
- **Excluding flow instability and severe slugging** (the two oscillating conditions), 66% of
  the remaining 107 episodes are detected (71 of 107, 95% CI 51–81%), at the same false-alarm
  rate; on the test wells, 65% (20 of 31).
- Severe slugging: 31 of the 32 real episodes come from one well (WELL-00014). When that well
  is held out, the model has one real slugging example plus simulations: 16% detected.

So the realistic summary is: the system reliably catches the sudden, well-defined events
(DHSV closure, hydrates, productivity loss, PCK scaling) and largely misses chronic oscillatory
conditions. Raising the headline number meaningfully needs labelled examples of those conditions
from more wells, or a different framing (flagging a well as chronically unstable, rather than
detecting the onset of an event).

Reproduce:

```bash
python -m owfd.models.cv --model artifacts/xgb/xgb_ctx-<version>    # ~30 min on 4 cores
python -m owfd.models.cv --model artifacts/xgb/xgb_tuned-<version>
python -m owfd.models.evaluate --select-on cv --models artifacts/xgb/xgb_tuned-<v> artifacts/xgb/xgb_ctx-<v>
```
