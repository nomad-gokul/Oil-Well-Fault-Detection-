# Portfolio notes

## Resume bullets

- Built an ML early-warning system for **offshore oil well faults** on the Petrobras 3W dataset
  (2,228 sensor time series, 40 wells, 10 event types): well-level, leak-free evaluation;
  **XGBoost tuned with Optuna** reached fault-vs-normal **ROC-AUC 0.81** on wells never seen in
  training and detected **56% of fault episodes with a 4-minute median delay**.
- Engineered **baseline-relative features** (deviation from each well's own normal level) that
  lifted cross-well macro F1 from 0.20 to 0.35 in a controlled comparison, and diagnosed why a
  **PyTorch CNN** memorised training wells (validation F1 0.08), with SHAP explanations that
  match the physics of each fault.
- Shipped it as a **FastAPI** service (streaming ingestion, alert lifecycle, SHAP drivers per
  alert, Prometheus metrics, fleet drift check) with a live **Streamlit** dashboard; 43 ms
  inference, 0 failures under a 60-user load test, PyTorch-free 531 MB Docker image,
  **GitHub Actions CI/CD** and 134 tests; deployed on Render + Streamlit Cloud.

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
On validation: the highest detection rate with at most 1 false alarm per day (P(fault) > 0.85
for 5 consecutive minutes). On test wells it gives 2.9 false alarms/day, so false-alarm rates
vary by well; in production I would re-tune per field after a burn-in period.

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
Sim-to-real gap (severe slugging is learned from simulations only and detected 19% of the
time), tiny test counts for rare classes, and class confusion during transients.

**10. How would you scale it to hundreds of wells?**
Readings already come in per-well batches; partition wells across API workers (the streaming
state is per well), move persistence to Postgres, and batch model calls across wells. The
profiling-driven NumPy rewrite of preprocessing made a streamed reading 3.2x cheaper.
