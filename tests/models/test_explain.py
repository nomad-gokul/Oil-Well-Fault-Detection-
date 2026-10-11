import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from owfd.config import get_settings
from owfd.models.explain import Explainer, plot_importance
from owfd.models.train_xgb import train_xgb
from tests.models.test_train_xgb import CONFIG, make_features


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("xgb")
    train_xgb(CONFIG, make_features(tmp / "f"), tmp / "m")
    return tmp / "m"


def test_shap_values_are_additive(model_dir: Path) -> None:
    ex = Explainer(model_dir)
    x = pd.read_parquet(model_dir.parent / "f" / "test.parquet")[ex.features].iloc[:20]
    contrib = ex.contributions(x)
    full = ex.booster.predict(xgb.DMatrix(x.to_numpy(np.float32), feature_names=ex.features),
                              pred_contribs=True)  # fmt: skip
    margin = ex.booster.predict(xgb.DMatrix(x.to_numpy(np.float32), feature_names=ex.features),
                                output_margin=True)  # fmt: skip
    bias = np.asarray(full).reshape(20, 10, -1)[:, :, -1]
    np.testing.assert_allclose(contrib.sum(axis=2) + bias, margin, rtol=1e-4, atol=1e-4)


def test_explain_window_names_the_informative_feature(model_dir: Path) -> None:
    ex = Explainer(model_dir)
    x = pd.read_parquet(model_dir.parent / "f" / "test.parquet")[ex.features].iloc[7]
    drivers = ex.explain_window(x, top=2)
    assert drivers[0].feature == "P-PDG__mean"  # the only feature that encodes the class
    assert abs(drivers[0].contribution) >= abs(drivers[1].contribution)
    assert drivers[0].value == pytest.approx(float(x["P-PDG__mean"]))


def test_mean_abs_and_plot(model_dir: Path, tmp_path: Path) -> None:
    ex = Explainer(model_dir)
    x = pd.read_parquet(model_dir.parent / "f" / "test.parquet")[ex.features]
    imp = ex.mean_abs(x)
    assert imp.shape == (len(ex.features), 10)
    assert imp.idxmax().eq("P-PDG__mean").sum() >= 8
    plot_importance(imp, tmp_path / "p.png")
    assert (tmp_path / "p.png").exists()


def test_explain_window_is_fast_on_the_real_model() -> None:
    settings = get_settings()
    runs = sorted((settings.resolve(settings.artifacts_dir) / "xgb").glob("xgb_baseline-*"))
    if not runs:
        pytest.skip("no trained XGBoost model")
    ex = Explainer(runs[-1])
    x = np.random.default_rng(0).normal(size=(100, len(ex.features))).astype(np.float32)
    ex.explain_window(x[0])  # warm-up
    t0 = time.perf_counter()
    for row in x:
        ex.explain_window(row)
    mean_ms = (time.perf_counter() - t0) / len(x) * 1000
    assert mean_ms < 50, mean_ms
