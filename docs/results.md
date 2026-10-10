# Results

All models are trained on the train split (real + simulated + drawn instances) and evaluated
on **held-out real wells only** (see [data_report.md](data_report.md) §3). Metrics are
window-level over the 10 event classes; macro F1 and balanced accuracy average over the classes
present in each split. "Fault" metrics treat classes 1–9 as one class versus normal (0).

<!-- results:begin -->
| Model | Split | Macro F1 | Balanced acc. | Fault ROC-AUC | Fault PR-AUC | Fault F1 | Windows |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb_baseline | val | 0.440 | 0.504 | 0.798 | 0.592 | 0.414 | 41,673 |
| xgb_baseline | test | 0.446 | 0.516 | 0.824 | 0.862 | 0.810 | 133,627 |
| xgb_no_missing_features | val | 0.420 | 0.476 | 0.799 | 0.591 | 0.387 | 41,673 |
| xgb_no_missing_features | test | 0.421 | 0.491 | 0.815 | 0.857 | 0.799 | 133,627 |
<!-- results:end -->

## XGBoost baseline (Task 7)

`python -m owfd.models.train_xgb` — 125 features per window (97 window statistics + 28
deviations from the well's own baseline), class-balanced sample weights (capped at 30x),
tracked in MLflow (`mlflow.db`; browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
Artifacts: `artifacts/xgb/<name>-<timestamp>/` (model, feature list, scaler, metrics,
confusion matrices).

### What changed during the self-test loop

| Version | Val macro F1 | Test macro F1 | Why |
|---|---:|---:|---|
| Early stopping on val log-loss, 97 features | 0.139 | 0.263 | Val log-loss bottomed out after **16 trees**: probabilities on unseen wells become over-confident long before the ranking stops improving |
| Fixed 600 trees, regularised (depth 4, min_child_weight 20) | 0.200 | 0.426 | Early stopping was the main problem |
| + deviation from the well's own baseline, trees picked by val macro F1 | **0.440** | **0.446** | Each well runs at different pressures/temperatures; "how far from this well's normal level" transfers across wells |

Dropping the absolute-level features instead made things worse (test 0.175), so levels and
deviations are both useful. The baseline is causal: the running mean over the first hour of the
stream, using only data up to the window end (`owfd.data.windows.causal_baseline`).

### Shortcut check (EDA decision 6)

Without the missing-data features the model scores 0.420 (val) / 0.421 (test), only 0.02–0.03
lower: it is **not** relying on "this looks like a simulated instance".

### Per-class test results (held-out wells)

| Class | Precision | Recall | F1 | Test windows |
|---|---:|---:|---:|---:|
| 0 Normal | 0.73 | 0.84 | 0.78 | 57,565 |
| 1 Abrupt Increase of BSW | 0.15 | 0.24 | 0.19 | 852 |
| 2 Spurious Closure of DHSV | 0.19 | 0.73 | 0.31 | 320 |
| 3 Severe Slugging | 0.01 | 0.00 | 0.00 | 9,202 |
| 4 Flow Instability | 0.62 | 0.67 | 0.65 | 7,657 |
| 5 Rapid Productivity Loss | 0.53 | 0.07 | 0.13 | 5,124 |
| 6 Quick Restriction in PCK | 0.04 | 0.35 | 0.07 | 43 |
| 7 Scaling in PCK | 0.84 | 0.90 | 0.87 | 47,801 |
| 8 Hydrate in Production Line | 0.98 | 0.72 | 0.83 | 3,685 |
| 9 Hydrate in Service Line | 0.66 | 0.65 | 0.65 | 1,378 |

Fault vs normal on test: ROC-AUC 0.824, PR-AUC 0.862, F1 0.810.

**Takeaways.** Severe slugging is learned from simulated data only (no real training well) and
does **not** transfer to the real slugging well (00014): a clear sim-to-real gap. Rapid
productivity loss is also learned mostly from simulations (1 real training instance) and is
mostly missed. The rare classes (1, 2, 6) have high recall but low precision, and only 2–3 real
test instances each, so their scores are noisy. Validation macro F1 was still rising at the
1,000-tree limit; Task 9 tunes this.
