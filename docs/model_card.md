# Model card — oil well fault early warning (tuned XGBoost)

| | |
|---|---|
| **Model** | XGBoost multiclass (10 event classes), depth 3, 1,000 trees (tree count chosen on validation), temperature-calibrated |
| **Bundle** | `deploy/bundle/` (`manifest.json` has version, git commit and test metrics) |
| **Input** | 1 Hz readings of 7 sensors (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, QGL) from one well; any of them may be missing |
| **Output** | every minute: probabilities of 10 classes for the last 5 minutes, P(fault) = 1 − P(normal), top SHAP drivers; an alert when P(fault) > 0.85 for 5 consecutive minutes |

## Intended use

Decision support for production engineers monitoring offshore wells: an early, explained
warning that a well's behaviour looks like one of the 3W undesirable events, so a human can
look sooner. **Not** for autonomous control or safety-critical shutdown decisions: the model
misses a large share of some fault types (below) and raises a few false alarms per day.

## Training data

Petrobras 3W v2.0.0 (CC BY 4.0): 680 real instances from 23 wells plus 1,089 simulated and 20
hand-drawn instances. 5-minute windows every minute, at most 300 per instance (480,070
training windows). Class weights balance rare events (capped at 30x). Details:
[data_report.md](data_report.md).

## Evaluation

Split **by well**: validation 7 wells (160 real instances), test 10 wells (279 real
instances), no overlap. Model, tree count, calibration and alert rule chosen on validation only.

**Per window (test):** macro F1 0.433, balanced accuracy 0.505, fault-vs-normal ROC-AUC 0.811,
PR-AUC 0.857, F1 0.810; calibration error 0.110 after temperature scaling.

**As an alerting system (test, 126 fault episodes, 959 h of normal operation):** 56% of
episodes detected, median delay 4 min after onset, 91% of detections before the steady state
(median 217 min ahead), 2.9 false alarms per 24 h of normal operation.

| Event class | Test episodes | Detected |
|---|---:|---:|
| Spurious closure of DHSV | 3 | 100% |
| Scaling in PCK | 10 | 100% |
| Hydrate in production line | 2 | 100% |
| Flow instability | 64 | 67% |
| Hydrate in service line | 5 | 60% |
| Rapid productivity loss | 6 | 33% |
| Quick restriction in PCK | 3 | 33% |
| Severe slugging | 31 | 19% |
| Abrupt increase of BSW | 2 | 0% |

## Known failure modes

- **Sim-to-real gap:** severe slugging has no real training well; the model learned it from
  simulations and misses most real slugging. Rapid productivity loss and BSW have 1 real
  training instance each.
- **New wells:** false-alarm rates differ between wells (0.6/day on validation wells vs 2.9/day
  on test wells). Thresholds should be re-tuned per field after a burn-in period.
- **Class confusion during transients:** P(fault) often rises before the predicted class is
  right (e.g. a DHSV closure first reported as BSW).
- **Sensor availability:** absent sensors are handled (masks, baseline-relative features), but
  wells with very few sensors carry less signal.
- **Small test counts** for classes 1, 2, 6 and 8 (2–3 episodes): their rates are very uncertain.

## Explainability and monitoring

Every alert stores its top 5 SHAP drivers; most drivers match the physics of the event (see
[results.md](results.md#explainability-task-11)). The API exposes Prometheus metrics and a
fleet-level drift check ([monitoring.md](monitoring.md)).

## Ethical and safety considerations

Missed detections are possible for every class; the system must complement, not replace,
existing safety systems and operator judgement. The data comes from one operator's wells;
behaviour on other fields, equipment or operating practices is unknown.
