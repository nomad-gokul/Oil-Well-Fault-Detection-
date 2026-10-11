# Model card — oil well fault early warning (XGBoost with long-context features)

| | |
|---|---|
| **Model** | XGBoost multiclass (10 event classes), depth 3, tree count chosen on validation, temperature-calibrated (T = 1.84) |
| **Bundle** | `deploy/bundle/` (`manifest.json` has version, git commit, alert rule and test metrics) |
| **Input** | 1 Hz readings of 7 sensors (P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, QGL) from one well; any of them may be missing |
| **Features** | 237: statistics of the last 5 minutes (125, incl. deviation from the well's first-hour baseline) and of one-minute averages over the last 30 and 120 minutes (112: level, spread, trend, oscillation period and strength) |
| **Output** | every minute: probabilities of 10 classes, P(fault) = 1 − P(normal), top SHAP drivers; an alert when P(fault) exceeds the well's threshold for 5 consecutive minutes (0.85, raised per well after a one-hour burn-in to the burn-in 90th percentile + 0.1, at most 0.99) |

## Intended use

Decision support for production engineers monitoring offshore wells: an early, explained
warning that a well's behaviour looks like one of the 3W undesirable events, so a human can
look sooner. **Not** for autonomous control or safety-critical shutdown decisions: the model
misses a large share of some fault types (below) and raises a few false alarms per day.

## Training data

Petrobras 3W v2.0.0 (CC BY 4.0): 680 real instances from 23 wells plus 1,089 simulated and 20
hand-drawn instances. Windows every minute, at most 300 per instance (480,070 training windows).
Class weights balance rare events (capped at 30x). Details: [data_report.md](data_report.md).

## Evaluation

Split **by well**: train 23 wells, validation 7 wells (160 real instances), test 10 wells (279
real instances), no overlap. Tree count and calibration on validation; model and alert rule on
the cross-validation predictions of the 30 non-test wells; test reported once. Plus 5-fold
cross-validation by well over all 40 real wells ([cross_validation.md](cross_validation.md)).

**Per window (test):** macro F1 0.464, fault-vs-normal ROC-AUC 0.852, PR-AUC 0.898; calibration
error 0.070 after temperature scaling.

**As an alerting system:**

| | Test (10 wells, 126 episodes, 959 h normal) | Cross-validation (40 wells, 482 episodes, 4,053 h normal) |
|---|---:|---:|
| Episodes detected | 52% | 27% (95% CI 12–58%) at 2.6 false alarms / day; 31% at 1 / day (pooled) |
| Episodes detected, excluding flow instability and severe slugging | 65% (20 of 31) | 66% (71 of 107; 95% CI 51–81%) |
| False alarms per 24 h of normal operation | 1.8 | see left |
| Median delay after onset | 4 min | 7 min |
| Median lead before the steady state | 307 min | 173 min |

| Event class | Test episodes | Detected (test) | CV episodes | Detected (CV) |
|---|---:|---:|---:|---:|
| Spurious closure of DHSV | 3 | 100% | 22 | 100% |
| Scaling in PCK | 10 | 100% | 36 | 61% |
| Hydrate in production line | 2 | 100% | 14 | 79% |
| Flow instability | 64 | 67% | 343 | 16% |
| Hydrate in service line | 5 | 40% | 14 | 57% |
| Rapid productivity loss | 6 | 33% | 11 | 73% |
| Quick restriction in PCK | 3 | 33% | 6 | 0% |
| Severe slugging | 31 | 10% | 32 | 16% |
| Abrupt increase of BSW | 2 | 0% | 4 | 0% |

Previous model (5-minute features only): 56% on test at 2.9 false alarms / day; in
cross-validation 6–8 points lower detection than this model at equal false-alarm rates.

## Chronic-instability profiles (second, per-well model)

Wells with labelled history also have a small per-well model for flow instability (7 real wells
in the bundle), raising separate "chronic" alerts at P(unstable) > 0.8 for 20 minutes. Replaying
every well's history in time order, with each recording scored only by a profile trained on that
well's earlier recordings: 81% of unstable episodes caught (93% once a well has a profile) at
0.31 extra false alarms per day on those wells (profiles expire 180 days after their newest
training data); the whole system then catches 78% of the 482
fault episodes across 40 wells (onset model alone: 27%). It tracks continuing instability (329
of 343 episodes) and catches only 4 of 14 new onsets; it needs operator labels. Details:
[chronic_instability.md](chronic_instability.md).

## Known failure modes

- **Chronic flow instability** is rarely detected across wells (16% in cross-validation): two
  wells that are unstable from the first minute of every recording supply 201 of the 482 CV
  episodes, and there is no normal period to compare them with. The 10 test wells happen to
  contain easier instability episodes (67%).
- **Sim-to-real gap:** 31 of the 32 real severe-slugging episodes come from one well (in the
  test split), so the model learned slugging from simulations and misses most real slugging.
  Rapid productivity loss and BSW have 1 real training instance each.
- **New wells:** false-alarm rates differ between wells. Per-well thresholds after a one-hour
  burn-in are in place, but in cross-validation they did not raise detection on average.
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
