"""Hyperparameter search with Optuna (objective: validation macro F1; test is never read).

Usage::

    python -m owfd.models.tune --model xgb --trials 25      # then retrains the best on all data
    python -m owfd.models.tune --model dl --trials 8

Studies are stored in ``artifacts/optuna.db`` and resume where they stopped.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from owfd.config import PROJECT_ROOT, get_settings
from owfd.models.metrics import N_CLASSES, classification_metrics
from owfd.models.train_xgb import balanced_weights

logger = logging.getLogger(__name__)


def stratified_subsample(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Sorted row indices keeping ``frac`` of every class (at least one row each)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        keep.append(rng.choice(rows, size=max(1, round(frac * len(rows))), replace=False))
    return np.sort(np.concatenate(keep))


class F1Pruning(TrainingCallback):
    """Report validation macro F1 to Optuna every ``every`` rounds; prune bad trials."""

    def __init__(self, trial: optuna.Trial, x_val: pd.DataFrame, y_val: np.ndarray, every: int):
        self.trial, self.x_val, self.y_val, self.every = trial, x_val, y_val, every
        self.best = (0.0, 0)

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        k = epoch + 1
        if k % self.every:
            return False
        from xgboost import DMatrix

        proba = model.predict(DMatrix(self.x_val), iteration_range=(0, k))
        f1 = classification_metrics(self.y_val, proba.reshape(-1, N_CLASSES))["macro_f1"]
        self.best = max(self.best, (f1, -k))
        self.trial.report(f1, k)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return False


def xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoost search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def xgb_objective(
    trial: optuna.Trial,
    train: tuple[pd.DataFrame, np.ndarray],
    val: tuple[pd.DataFrame, np.ndarray],
    base: dict[str, Any],
    seed: int,
) -> float:
    """Best validation macro F1 over the boosting rounds of one configuration."""
    (x, y), (x_val, y_val) = train, val
    params = {**base["model"], **xgb_space(trial)}
    pruning = F1Pruning(trial, x_val, y_val, base["select_every"])
    model = XGBClassifier(objective="multi:softprob", num_class=N_CLASSES, random_state=seed,
                          n_jobs=base["n_jobs"], callbacks=[pruning], **params)  # fmt: skip
    model.fit(x, y, sample_weight=balanced_weights(y, base["class_weight_cap"]))
    f1, neg_k = pruning.best
    trial.set_user_attr("n_trees", -neg_k)
    return f1


def dl_space(trial: optuna.trial.BaseTrial, base: dict[str, Any]) -> dict[str, Any]:
    """Deep-model search space applied on top of ``base`` (configs/dl.yaml)."""
    cfg = copy.deepcopy(base)
    cfg["train"]["lr"] = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
    drop = trial.suggest_float("dropout", 0.0, 0.4)
    if cfg["arch"] == "cnn":
        width = trial.suggest_categorical("width", [16, 32, 48])
        kernel = trial.suggest_categorical("kernel_size", [5, 7, 11])
        cfg["cnn"].update(widths=[width, 2 * width, 3 * width], dropout=drop, kernel_size=kernel)
    else:
        cfg["gru"].update(hidden=trial.suggest_categorical("hidden", [32, 64, 96]), dropout=drop)
    return cfg


def make_study(name: str, storage: str, seed: int) -> optuna.Study:
    """Create or resume a maximisation study with median pruning."""
    return optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
    )  # fmt: skip


def remaining(study: optuna.Study, target: int) -> int:
    """Trials still to run so that the study holds ``target`` finished trials."""
    done = [t for t in study.trials if t.state.is_finished()]
    return max(0, target - len(done))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters on validation data.")
    parser.add_argument("--model", choices=["xgb", "dl"], required=True)
    parser.add_argument("--trials", type=int, default=25, help="target number of trials")
    parser.add_argument("--train-frac", type=float, default=0.3, help="xgb: train subsample")
    parser.add_argument("--arch", choices=["cnn", "gru"], default="gru", help="dl only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    settings = get_settings()
    seed = settings.random_seed
    artifacts = settings.resolve(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{artifacts / 'optuna.db'}"
    processed = settings.resolve(settings.processed_data_dir)

    if args.model == "xgb":
        base = yaml.safe_load((PROJECT_ROOT / "configs" / "xgb.yaml").read_text())
        names = json.loads((processed / "features" / "feature_names.json").read_text())
        tr = pd.read_parquet(processed / "features" / "train.parquet")
        va = pd.read_parquet(processed / "features" / "val.parquet")
        keep = stratified_subsample(tr["event_class"].to_numpy(), args.train_frac, seed)
        train = (tr.iloc[keep][names], tr["event_class"].to_numpy()[keep])
        val = (va[names], va["event_class"].to_numpy())
        study = make_study("xgb", storage, seed)
        study.optimize(lambda t: xgb_objective(t, train, val, base, seed),
                       n_trials=remaining(study, args.trials))  # fmt: skip
        best = {**base["model"], **study.best_params}
        tuned = {**base, "model": best}
        out = PROJECT_ROOT / "configs" / "xgb_tuned.yaml"
        out.write_text(
            "# Written by python -m owfd.models.tune --model xgb\n" + yaml.safe_dump(tuned)
        )
    else:
        from owfd.data.windows import WindowStore
        from owfd.models.train_dl import train_dl, windows_for

        base = yaml.safe_load((PROJECT_ROOT / "configs" / "dl.yaml").read_text())
        base["arch"] = args.arch
        base["train"].update(epochs=4, windows_per_epoch=40000, patience=4)
        train_w = windows_for(WindowStore(processed / "windows" / "train"), base)
        val_w = windows_for(WindowStore(processed / "windows" / "val"), base)

        def objective(trial: optuna.Trial) -> float:
            cfg = dl_space(trial, base)

            def report(epoch: int, f1: float) -> None:
                trial.report(f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            out_dir = artifacts / "optuna_dl" / str(trial.number)
            _, history = train_dl(cfg, train_w, val_w, out_dir, n_channels=train_w.n_channels,
                                  seed=seed, on_epoch=report)  # fmt: skip
            return max(h["val_macro_f1"] for h in history)

        study = make_study(f"dl_{args.arch}", storage, seed)
        study.optimize(objective, n_trials=remaining(study, args.trials))
        out = PROJECT_ROOT / "configs" / f"dl_{args.arch}_tuned.yaml"
        tuned = dl_space(optuna.trial.FixedTrial(study.best_params), base)
        # Full-length training settings from dl.yaml, with the tuned learning rate.
        tuned["train"].update(
            yaml.safe_load((PROJECT_ROOT / "configs" / "dl.yaml").read_text())["train"]
        )
        tuned["train"]["lr"] = study.best_params["lr"]
        out.write_text(
            "# Written by python -m owfd.models.tune --model dl\n" + yaml.safe_dump(tuned)
        )

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    logger.info("Best val macro F1 %.3f with %s -> %s\n%s", study.best_value, study.best_params,
                out, trials.sort_values("value", ascending=False).head(10).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
