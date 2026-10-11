import inspect
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from owfd.models import tune
from owfd.models.tune import make_study, remaining, stratified_subsample, xgb_objective

BASE = {
    "model": {"n_estimators": 40, "tree_method": "hist"},
    "select_every": 10,
    "class_weight_cap": 30.0,
    "n_jobs": 2,
}


def data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 10
    return pd.DataFrame({"a": y + rng.normal(0, 0.5, n), "b": rng.normal(size=n)}), y


def test_stratified_subsample_keeps_every_class() -> None:
    y = np.array([0] * 100 + [1] * 10 + [2])
    keep = stratified_subsample(y, 0.3, seed=0)
    assert (np.diff(keep) > 0).all()
    counts = np.bincount(y[keep])
    assert counts.tolist() == [30, 3, 1]


def test_study_runs_and_resumes(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    train, val = data(400, 0), data(200, 1)
    study = make_study("xgb", storage, seed=0)
    study.optimize(lambda t: xgb_objective(t, train, val, BASE, 0), n_trials=remaining(study, 2))
    assert len(study.trials) == 2
    # "kill" and resume: a new study object on the same storage continues to 4 trials
    resumed = make_study("xgb", storage, seed=0)
    assert remaining(resumed, 4) == 2
    resumed.optimize(
        lambda t: xgb_objective(t, train, val, BASE, 0), n_trials=remaining(resumed, 4)
    )
    assert len(resumed.trials) == 4
    assert 0 < resumed.best_value <= 1
    assert "n_trees" in resumed.best_trial.user_attrs


def test_tuning_never_reads_the_test_split() -> None:
    source = inspect.getsource(tune)
    assert "test" not in source.replace("tests", "").replace("Test", "").replace(
        "test is never", ""
    )


def test_pruned_trials_are_counted_as_finished(tmp_path: Path) -> None:
    study = make_study("x", f"sqlite:///{tmp_path / 'o.db'}", seed=0)

    def obj(trial: optuna.Trial) -> float:
        if trial.number == 1:
            raise optuna.TrialPruned()
        return 0.5

    study.optimize(obj, n_trials=3)
    assert remaining(study, 3) == 0
