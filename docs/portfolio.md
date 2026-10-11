# Portfolio notes

## Resume bullets

- Built an ML early-warning system for **offshore oil well faults** on the Petrobras 3W dataset
  (2,228 sensor time series, 40 wells, 10 event types) with leak-free, **well-level evaluation
  and 5-fold cross-validation by well**; tuned **XGBoost** reached fault-vs-normal **ROC-AUC
  0.85** on unseen wells and caught **52% of fault episodes at 1.8 false alarms/day** (4-minute
  median delay, 100% of valve closures and hydrates).
- Engineered **baseline-relative** and **30/120-minute context features** (trend, oscillation
  period and strength) that added **6–8 points of detection at equal false-alarm rates across
  all 40 wells**; diagnosed that the remaining misses are chronic flow instability on two wells,
  and documented two negative results (a PyTorch CNN that memorised wells; per-well thresholds).
- Shipped it as a **FastAPI** service (streaming ingestion with verified training/serving parity,
  alert lifecycle, SHAP drivers per alert, Prometheus metrics, drift check) with a live
  **Streamlit** dashboard; ~30 ms inference, Docker, **GitHub Actions CI/CD**, 151 tests;
  deployed on Railway + Streamlit Cloud.

- Reframed the largest failure mode after error analysis: chronic flow instability cannot be
  learned across wells (AUC 0.57) but is separable within each well, so I built **per-well
  profile models trained from operator feedback**, evaluated by replaying each well's history in
  time order: system-wide detection rose from **27% to 78%** of fault episodes at +0.2 false
  alarms/day (52% with only a quarter of past recordings labelled).

## Likely interview questions

**1. Why split by well instead of randomly?**
Consecutive windows from the same well are nearly identical; a random split puts near-copies in
train and test and inflates scores. Splitting by well measures what matters in production:
performance on a well the model has never seen.

**2. What was the single most important modelling decision?**
Deviation-from-baseline features. Each well runs at its own pressures and temperatures, so
absolute levels identify the well rather than the fault. Measuring each sensor against that
well's own first hour made the model transfer across wells (validation macro F1 0.20 → 0.35).

**3. Why did the deep model do worse than XGBoost?**
It memorised training wells: it labelled 92% of the validation wells' normal windows as a fault
class. With only 23 training wells, hand-crafted baseline-relative statistics generalise better
than learned filters. A well-invariant input representation helped only slightly on a small
budget. I kept it as a documented negative result.

**4. Why did early stopping hurt XGBoost at first?**
Validation log-loss bottomed out after 16 trees because probabilities on unseen wells become
over-confident long before the ranking stops improving. Choosing the tree count by validation
macro F1 instead raised validation F1 from 0.14 to 0.44 (together with the new features).

**5. How do you handle class imbalance?**
Class weights (inverse frequency, capped at 30x), macro F1 as the selection metric, at most 300
windows per instance in training, and alert-level metrics that count fault *episodes* rather
than windows.

**6. How is the alert threshold chosen, and is it realistic?**
On the cross-validation predictions of the 30 non-test wells: the highest detection rate with
at most 1 false alarm per day (P(fault) > 0.85 for 5 consecutive minutes, raised per well after a
one-hour burn-in). Choosing on the 7 validation wells alone did not transfer to the test wells,
which is why selection moved to 30 wells. On test it gives 1.8 false alarms/day.

**7. How do you make sure online predictions match offline ones?**
The serving code reuses the training preprocessing, and tests stream instances in random chunk
sizes and compare against the batch pipeline. That caught two real bugs: the streaming buffer
needed extra history for forward-fill after a stuck-sensor mask, and serving had to round
inputs to float16 like the training window store (a 0.11 probability skew before the fix).

**8. How would you detect data drift here?**
Pooled-distribution PSI does not work: one normal well already saturates it. I compare each
well's recent deviation statistics (median offset, spread) with the range seen on real
training/validation wells and flag drift only when at least half of the fleet is unusual on the
same sensor, e.g. a unit change in the data pipeline. A single faulty well is an alert, not
drift.

**9. What are the biggest limitations?**
Chronic flow instability on two wells (most of the misses across wells), the sim-to-real gap
for severe slugging (31 of 32 real episodes come from one well), tiny counts for rare classes,
and class confusion during transients.

**10. How would you scale it to hundreds of wells?**
Readings already come in per-well batches; partition wells across API workers (the streaming
state is per well), move persistence to Postgres, and batch model calls across wells. The
profiling-driven NumPy rewrite of preprocessing made a streamed reading 3.2x cheaper.


**Follow-up: why is cross-validation so much lower than the test split?**
The test split has 10 wells and happened to be easier than average. Across all 40 wells (5-fold
CV by well, nested tuning, bootstrap intervals), about one in four fault episodes is caught at
about one false alarm per day. 71% of all episodes are flow instability, mostly on two wells that
are unstable from the first minute of every recording, so "deviation from this well's normal"
has nothing to compare with. Excluding those, 55% are caught. Reporting both numbers, and why
they differ, is more useful than quoting the best split.

**Follow-up: how did you get from 27% to 78%, and is that real?**
Most misses were flow instability. A detector trained on other wells was barely better than
chance (AUC 0.57): each well is unstable in its own way. Within a well it separates almost
perfectly. So the product needs a feedback loop: operators label episodes, and each well gets a
small profile model. I measured it the way it would be used: replaying each well's recordings in
time order, every recording scored only by a profile trained on earlier, non-overlapping
recordings, first episodes counted as misses. 78% is real for *tracking a known condition*:
329 of 343 unstable episodes continue an earlier one, and only 4 of 14 brand-new onsets are
caught. With only 25% of recordings labelled it is 52%.
