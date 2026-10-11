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
| dl_cnn | val | 0.075 | 0.218 | 0.350 | 0.267 | 0.318 | 41,673 |
| dl_cnn | test | 0.218 | 0.386 | 0.432 | 0.557 | 0.541 | 133,627 |
| xgb_tuned | val | 0.453 | 0.523 | 0.774 | 0.600 | 0.483 | 41,673 |
| xgb_tuned | test | 0.433 | 0.505 | 0.811 | 0.857 | 0.810 | 133,627 |
| xgb_osc | val | 0.488 | 0.555 | 0.737 | 0.501 | 0.424 | 41,673 |
| xgb_osc | test | 0.463 | 0.509 | 0.799 | 0.840 | 0.755 | 133,627 |
| xgb_ctx | val | 0.498 | 0.571 | 0.656 | 0.450 | 0.383 | 41,673 |
| xgb_ctx | test | 0.464 | 0.507 | 0.852 | 0.898 | 0.800 | 133,627 |
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

## Deep sequence model (Task 8)

`python -m owfd.models.train_dl --arch cnn|gru` — a residual 1D-CNN or a GRU on the raw
5-minute windows (`configs/dl.yaml`): class-weighted cross-entropy, AdamW, one-cycle learning
rate, gradient clipping, early stopping on validation macro F1, CPU-only (about 2.5 min per
epoch of 120k windows on 4 cores). Both architectures pass an over-fitting sanity check
(about 100% accuracy on 64 windows) and a `--fast-dev-run` smoke test.

**Result: the network does not generalise to unseen wells.**

| Run | Val macro F1 | Test macro F1 | Val fault ROC-AUC |
|---|---:|---:|---:|
| CNN, absolute inputs, early-stopped after 9 epochs | 0.075 | 0.218 | 0.350 |
| XGBoost (for comparison) | 0.440 | 0.446 | 0.798 |

Training loss fell steadily (0.32 → 0.06) while validation F1 stayed below 0.08. The
validation confusion matrix shows why: 92% of the *normal* windows from the two validation
wells were predicted as "Scaling in PCK" (mean P(normal) 0.06). The network memorises each
training well's operating levels (normal operation in training comes from a handful of real
wells), so a new well's normal operation looks like something else.

A well-independent input representation (`input_mode: relative` — each sensor minus its window
mean, plus the deviation from the well's own baseline; no absolute levels) was tried on a small
budget (4 epochs x 40k windows): best validation macro F1 0.083 vs 0.055 for absolute inputs.
Not enough to compete. With 40 wells and only 23 for training, the hand-crafted, baseline-relative
statistics used by XGBoost transfer across wells far better than learned filters; more wells,
per-well normalisation learned from the first hours of each stream, or domain-adversarial
training would be the next things to try. XGBoost remains the production model; the deep model
and the comparison stay in the project as an honest negative result.

## Explainability (Task 11)

`python -m owfd.models.explain` computes exact TreeSHAP values natively in XGBoost
(`pred_contribs`) on a class-balanced sample of test windows: [importance per class](figures/shap_importance_per_class.png),
`docs/shap_importance.csv`. `Explainer.explain_window` returns the top 5 drivers of a single
prediction in **33 ms** on one CPU thread (1,000 trees x 10 classes; target < 50 ms, tested), and
the API attaches them to every alert. SHAP values are verified to add up to the model's raw
output.

**Do the drivers make physical sense?** Mostly, yes:

| Class | Top drivers | Physical reading |
|---|---|---|
| Spurious Closure of DHSV | P-TPT / T-TPT drop vs baseline | the safety valve shuts, flow and heat stop reaching the tree |
| Severe Slugging | std of P-PDG and T-TPT, P-PDG − P-TPT | large periodic pressure oscillations |
| Flow Instability | P-MON-CKP mean and std, P-TPT range | unstable pressure upstream of the production choke |
| Quick Restriction in PCK | P-MON-CKP min/max vs baseline | upstream choke pressure jumps when the choke closes |
| Scaling in PCK | P-MON-CKP mean/max vs baseline | upstream pressure creeps up as deposits narrow the choke |
| Hydrate in Production Line | P-TPT and T-JUS-CKP vs baseline | a blockage changes pressure and temperature along the line |
| Hydrate in Service Line | QGL and P-JUS-CKGL (gas lift) | the service line is the gas-lift line |
| Rapid Productivity Loss | P-PDG and T-JUS-CKP vs baseline | bottom-hole pressure rises, flowing temperature falls |

Weaker: Abrupt Increase of BSW and Scaling in PCK also lean on absolute pressure levels
(P-JUS-CKGL mean, P-PDG min), which are partly well-specific rather than physical.

## Hyperparameter tuning (Task 9)

`python -m owfd.models.tune --model xgb --trials 15` — Optuna (TPE sampler, median pruning on
validation macro F1 reported every 50 trees) on a stratified 30% sample of the training
windows; the study is stored in `artifacts/optuna.db` and resumes where it stopped. The
objective only sees train and validation data (tested). Best of 15 trials (5 completed, 10
pruned): validation macro F1 **0.475** on the sample with `max_depth 3, learning_rate 0.147,
min_child_weight 15.9, subsample 0.85, colsample_bytree 0.31, reg_lambda 7.6`
(`configs/xgb_tuned.yaml`): a shallower, more regularised model than the baseline, consistent
with the cross-well story.

Retrained on all training windows (`--config configs/xgb_tuned.yaml --name xgb_tuned`):

| | Val macro F1 | Test macro F1 | Test fault ROC-AUC |
|---|---:|---:|---:|
| xgb_baseline | 0.440 | **0.446** | **0.824** |
| xgb_tuned | **0.453** | 0.433 | 0.811 |

Better on validation, slightly worse on test: with 7 validation wells the selection signal is
noisy. The tuned model was still chosen for production because the choice is made on
validation by a fixed rule (see `docs/early_warning.md`); switching back after looking at the
test results would be tuning on the test set. The deep model was not tuned: at about 2.5 min
per epoch on CPU and with validation F1 below 0.1 there was nothing worth tuning.
