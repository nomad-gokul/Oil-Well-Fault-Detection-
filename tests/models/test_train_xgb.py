import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.features.build import META_COLUMNS
from owfd.models.report import results_rows, update_results
from owfd.models.train_xgb import balanced_weights, select_columns, train_xgb

CONFIG = {
    "model": {"n_estimators": 30, "learning_rate": 0.3, "max_depth": 3, "tree_method": "hist"},
    "select_every": 10,
    "class_weight_cap": 30.0,
    "n_jobs": 2,
}


def make_features(tmp_path: Path) -> Path:
    """Separable synthetic features: feature f0 encodes the class."""
    tmp_path.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = ["P-PDG__mean", "P-PDG__std", "P-PDG__frac_missing"]
    for split, n in (("train", 600), ("val", 200), ("test", 200)):
        y = np.arange(n) % 10
        df = pd.DataFrame({c: 0 for c in META_COLUMNS}, index=range(n))
        df["event_class"] = y
        df[names[0]] = y + rng.normal(0, 0.2, n)
        df[names[1]] = rng.normal(size=n)
        df[names[2]] = rng.random(n)
        df.to_parquet(tmp_path / f"{split}.parquet")
    (tmp_path / "feature_names.json").write_text(json.dumps(names))
    return tmp_path


def test_balanced_weights() -> None:
    y = np.array([0] * 90 + [1] * 9 + [2])
    w = balanced_weights(y, cap=5.0)
    assert w.mean() == pytest.approx(1.0)
    assert w[y == 1][0] / w[y == 0][0] == pytest.approx(5.0)  # 10x rarer, capped at 5
    assert w[y == 2][0] / w[y == 0][0] == pytest.approx(5.0)


def test_select_columns_ablation() -> None:
    names = ["a__mean", "a__frac_missing", "b__last"]
    assert select_columns(names, True) == names
    assert select_columns(names, False) == ["a__mean", "b__last"]


def test_train_xgb_learns_saves_and_is_deterministic(tmp_path: Path) -> None:
    features = make_features(tmp_path / "f")
    _, m1 = train_xgb(CONFIG, features, tmp_path / "a")
    _, m2 = train_xgb(CONFIG, features, tmp_path / "b")
    assert m1["test"]["macro_f1"] > 0.9
    assert m1 == m2  # same seed -> identical results
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    curve = {int(k): v for k, v in cfg["val_macro_f1_curve"].items()}
    assert cfg["n_trees"] in curve and curve[cfg["n_trees"]] == max(curve.values())
    for f in (
        "model.json",
        "feature_names.json",
        "metrics.json",
        "config.json",
        "confusion_test.png",
    ):
        assert (tmp_path / "a" / f).exists()
    _, m3 = train_xgb(CONFIG, features, tmp_path / "c", use_missing_features=False)
    assert "P-PDG__frac_missing" not in json.loads(
        (tmp_path / "c" / "feature_names.json").read_text()
    )


def test_update_results_replaces_rows(tmp_path: Path) -> None:
    m = {"macro_f1": 0.5, "balanced_accuracy": 0.5, "fault_roc_auc": 0.9, "fault_pr_auc": 0.8,
         "fault_f1": 0.7, "n": 1000}  # fmt: skip
    path = tmp_path / "results.md"
    update_results(path, "xgb", {"val": m, "test": m})
    update_results(path, "other", {"test": m})
    update_results(path, "xgb", {"val": {**m, "macro_f1": 0.6}, "test": m})
    rows = results_rows(path)
    assert len(rows) == 3
    assert sum("| xgb | val | 0.600 |" in r for r in rows) == 1
