"""Cross-validation by well: early-warning numbers over all real wells, with intervals.

Usage::

    python -m owfd.models.cv --model artifacts/xgb/xgb_ctx-<version> [--folds 5]

All real wells are split into folds (balanced by number of fault instances). For each fold an
XGBoost model with the hyperparameters and number of trees of ``--model`` (so nothing is
selected on the held-out fold) is trained on the simulated and hand-drawn instances plus the
real wells of the *other* folds (at most ``max_train_windows_per_instance`` windows per
instance, as in training). It predicts every window of the held-out wells, including their
burn-in hour.

The calibration temperature and the alert rule are chosen by nested cross-validation: for each
fold they are tuned on the out-of-fold predictions of the other folds and then applied to it.
Detection rate and false alarms per day get 95% bootstrap intervals (resampling wells).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from owfd.config import PROJECT_ROOT, get_settings
from owfd.models.evaluate import (
    AlertRule,
    Replay,
    apply_temperature,
    fit_temperature,
    per_class_summary,
    rule_grid,
    summarize,
    tune_rule,
)
from owfd.models.metrics import N_CLASSES
from owfd.models.train_xgb import balanced_weights

logger = logging.getLogger(__name__)

PROBA = [f"p{c}" for c in range(N_CLASSES)]
EVAL_META = ["instance_id", "well_id", "end_time", "end_pos", "event_class", "is_transient"]


def assign_folds(fault_counts: pd.Series, n_folds: int, seed: int = 0) -> dict[str, int]:
    """Wells -> fold, spreading wells with many fault instances evenly (snake order over wells
    sorted by fault count, ties broken randomly)."""
    rng = np.random.default_rng(seed)
    order = sorted(fault_counts.index, key=lambda w: (-fault_counts[w], rng.random()))
    folds = {}
    for i, well in enumerate(order):
        cycle, pos = divmod(i, n_folds)
        folds[str(well)] = pos if cycle % 2 == 0 else n_folds - 1 - pos
    return folds


def cap_per_instance(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """At most ``cap`` random windows per instance (like the training window store)."""
    rng = np.random.default_rng(seed)
    keep = [
        idx if len(idx) <= cap else np.sort(rng.choice(idx, size=cap, replace=False))
        for idx in df.groupby("instance_id").indices.values()
    ]
    return df.iloc[np.concatenate(keep)] if keep else df.iloc[:0]


def bootstrap_by_well(
    report: pd.DataFrame, n: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% intervals of detection rate and false alarms per day, resampling wells."""
    rng = np.random.default_rng(seed)
    per_well = report.groupby("well_id").agg(
        faults=("has_fault", "sum"),
        detected=("detected", "sum"),
        false_alarms=("false_alarms", "sum"),
        hours=("normal_hours", "sum"),
    )
    arr = per_well.to_numpy(dtype=float)
    det, fa = [], []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        det.append(s[1] / s[0] if s[0] else np.nan)
        fa.append(s[2] / s[3] * 24 if s[3] else np.nan)
    pct = (2.5, 97.5)
    lo_d, hi_d = np.nanpercentile(det, pct)
    lo_f, hi_f = np.nanpercentile(fa, pct)
    return {"detection_rate": (float(lo_d), float(hi_d)), "false_alarms_per_day": (lo_f, hi_f)}


def out_of_fold(
    tables: dict[str, pd.DataFrame],
    columns: list[str],
    config: dict[str, Any],
    n_trees: int,
    folds: dict[str, int],
    cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold probabilities for every labelled window and burn-in window of real wells."""
    synthetic = tables["train"][tables["train"]["source"] != "real"]
    real_capped = pd.concat(
        [tables["train"][tables["train"]["source"] == "real"]]
        + [cap_per_instance(tables[s], cap, seed) for s in ("val", "test")]
    )
    real_all = pd.concat([tables[s] for s in ("train_full", "val", "test")], ignore_index=True)
    burnin = tables["burnin"]
    oof: list[pd.DataFrame] = []
    oof_burn: list[pd.DataFrame] = []
    for f in sorted(set(folds.values())):
        held = {w for w, k in folds.items() if k == f}
        train = pd.concat([synthetic, real_capped[~real_capped["well_id"].isin(held)]])
        y = train["event_class"].to_numpy(dtype=int)
        params = {**config["model"], "n_estimators": n_trees}
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            random_state=seed,
            n_jobs=config["n_jobs"],
            **params,
        )
        model.fit(train[columns], y, sample_weight=balanced_weights(y, config["class_weight_cap"]))
        for src, out in ((real_all, oof), (burnin, oof_burn)):
            rows = src[src["well_id"].isin(held)]
            proba = pd.DataFrame(model.predict_proba(rows[columns]), columns=PROBA)
            meta = rows[[c for c in EVAL_META if c in rows]].reset_index(drop=True)
            out.append(pd.concat([meta.assign(fold=f), proba], axis=1))
        logger.info("fold %d: %d wells held out, %d training windows", f, len(held), len(train))
    return pd.concat(oof, ignore_index=True), pd.concat(oof_burn, ignore_index=True)


def nested_reports(
    oof: pd.DataFrame,
    oof_burn: pd.DataFrame,
    variants: dict[str, list[AlertRule]],
    max_false_alarms_per_day: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]]]:
    """Per-instance reports of every rule variant; temperature and rule tuned on other folds."""
    reports: dict[str, list[pd.DataFrame]] = {v: [] for v in variants}
    chosen: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    wells = oof.groupby("instance_id")["well_id"].first()
    for f in sorted(oof["fold"].unique()):
        inner = (oof["fold"] != f).to_numpy()
        t = fit_temperature(
            oof.loc[inner, PROBA].to_numpy(), oof.loc[inner, "event_class"].to_numpy()
        )
        p_fault = 1.0 - apply_temperature(oof[PROBA].to_numpy(), t)[:, 0]
        preds = oof[["instance_id", "end_time", "event_class", "is_transient"]].assign(
            p_fault=p_fault
        )
        b_fault = 1.0 - apply_temperature(oof_burn[PROBA].to_numpy(), t)[:, 0]
        burnin = {str(i): b_fault[ix] for i, ix in oof_burn.groupby("instance_id").indices.items()}
        tune, test = Replay(preds[inner], burnin), Replay(preds[~inner], burnin)
        for variant, rules in variants.items():
            rule, _ = tune_rule(
                tune, max_false_alarms_per_day=max_false_alarms_per_day, rules=rules
            )
            rep = test.report(rule)
            rep["well_id"] = rep["instance_id"].map(wells)
            rep["fold"] = f
            reports[variant].append(rep)
            chosen[variant].append({"fold": int(f), "temperature": t, **rule.__dict__})
    return {v: pd.concat(r, ignore_index=True) for v, r in reports.items()}, chosen


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cross-validation by well.")
    parser.add_argument("--model", type=Path, required=True, help="artifact folder (xgb)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-false-alarms", type=float, default=1.0, help="per 24 h")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    model_dir = args.model.resolve()
    config = json.loads((model_dir / "config.json").read_text())
    columns = json.loads((model_dir / "feature_names.json").read_text())
    features = settings.resolve(settings.processed_data_dir) / "features"
    need = [*EVAL_META, "source", *columns]
    tables = {
        name: pd.read_parquet(features / f"{name}.parquet", columns=need)
        for name in ("train", "val", "test", "train_full", "burnin")
    }
    real = pd.concat([tables[s] for s in ("train_full", "val", "test")])
    faults = real[real["event_class"] > 0].groupby("well_id")["instance_id"].nunique()
    fault_counts = faults.reindex(sorted(real["well_id"].unique()), fill_value=0)
    folds = assign_folds(fault_counts, args.folds, settings.random_seed)

    oof, oof_burn = out_of_fold(
        tables, columns, config, int(config["n_trees"]), folds,
        settings.windows.max_train_windows_per_instance, settings.random_seed,
    )  # fmt: skip
    thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)
    ks = (1, 2, 3, 5)
    variants = {
        "global threshold": rule_grid(thresholds, ks),
        "per-well threshold": rule_grid(
            thresholds, ks, quantiles=(None, 0.5, 0.9, 1.0), margins=(0.0, 0.05, 0.1)
        ),
    }
    reports, chosen = nested_reports(oof, oof_burn, variants, args.max_false_alarms)

    name = model_dir.name.rsplit("-", 2)[0]
    out_dir = settings.resolve(settings.artifacts_dir) / "cv" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / "oof.parquet")
    oof_burn.to_parquet(out_dir / "oof_burnin.parquet")
    summary = {"model": name, "folds": folds, "n_wells": len(folds), "variants": {}}
    for variant, rep in reports.items():
        rep.to_parquet(out_dir / f"report_{variant.replace(' ', '_')}.parquet")
        per_class = per_class_summary(rep)
        summary["variants"][variant] = {
            **summarize(rep),
            "ci95": bootstrap_by_well(rep),
            "rules": chosen[variant],
            "per_class": per_class.reset_index().to_dict("records"),
        }
        logger.info("%s / %s: %s", name, variant, summarize(rep))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    logger.info("-> %s", out_dir.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
