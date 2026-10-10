"""SHAP explanations for the XGBoost model (exact TreeSHAP, computed natively by XGBoost).

Usage::

    python -m owfd.models.explain [--model artifacts/xgb/<run>]  # global plots -> docs/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from owfd.data.schema import CLASS_NAMES
from owfd.models.metrics import N_CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Driver:
    """One feature's contribution to a prediction (in log-odds of the explained class)."""

    feature: str
    value: float
    contribution: float


class Explainer:
    """Per-window SHAP values for a saved XGBoost artifact folder."""

    def __init__(self, model_dir: Path) -> None:
        features = json.loads((model_dir / "feature_names.json").read_text())
        self._init(model_dir / "model.json", features)

    @classmethod
    def from_files(cls, model_file: Path, features: list[str]) -> Explainer:
        """Explainer for a model JSON file and its feature names (e.g. in a serving bundle)."""
        ex = cls.__new__(cls)
        ex._init(model_file, features)
        return ex

    def _init(self, model_file: Path, features: list[str]) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(model_file)
        self.booster.set_param({"nthread": 1})  # one window at a time: avoid thread start-up
        self.features: list[str] = list(features)

    def contributions(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """SHAP values [n, n_classes, n_features] (bias column dropped)."""
        values = np.asarray(x, dtype=np.float32).reshape(-1, len(self.features))
        dm = xgb.DMatrix(values, feature_names=self.features)
        contribs = self.booster.predict(dm, pred_contribs=True)
        return np.asarray(contribs).reshape(len(values), N_CLASSES, -1)[:, :, :-1]

    def explain_window(
        self, x: pd.Series | np.ndarray, top: int = 5, target: int | None = None
    ) -> list[Driver]:
        """The ``top`` features pushing the prediction towards ``target`` (default: the
        predicted class), largest absolute contribution first."""
        values = np.asarray(x, dtype=np.float32).reshape(1, -1)
        contrib = self.contributions(values)[0]
        if target is None:
            logits = contrib.sum(axis=1)
            target = int(np.argmax(logits))
        order = np.argsort(-np.abs(contrib[target]))[:top]
        return [Driver(self.features[i], float(values[0, i]), float(contrib[target, i]))
                for i in order]  # fmt: skip

    def mean_abs(self, x: pd.DataFrame) -> pd.DataFrame:
        """Global importance: mean |SHAP| per feature (rows) and class (columns)."""
        c = np.abs(self.contributions(x)).mean(axis=0)  # [classes, features]
        return pd.DataFrame(
            c.T, index=self.features, columns=[CLASS_NAMES[i] for i in range(N_CLASSES)]
        )


def plot_importance(importance: pd.DataFrame, path: Path, top: int = 8) -> None:
    """Top features per fault class (one small bar chart per class)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(12, 17))
    for ax, cls in zip(axes.ravel(), importance.columns, strict=True):
        s = importance[cls].sort_values().iloc[-top:]
        ax.barh(s.index, s.values, color="#3b6ea5")
        ax.set_title(cls, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Mean |SHAP| per class (test windows, XGBoost)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main(argv: list[str] | None = None) -> None:
    """Global explanation plots on a class-stratified sample of test windows."""
    from owfd.config import PROJECT_ROOT, get_settings

    parser = argparse.ArgumentParser(description="Global SHAP importance for XGBoost.")
    parser.add_argument("--model", type=Path, help="xgb artifact folder (default: latest)")
    parser.add_argument("--per-class", type=int, default=300, help="test windows per class")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    model_dir = args.model or sorted((settings.resolve(settings.artifacts_dir) / "xgb").glob(
        "xgb_baseline-*"))[-1]  # fmt: skip
    explainer = Explainer(model_dir)
    test = pd.read_parquet(
        settings.resolve(settings.processed_data_dir) / "features" / "test.parquet"
    )
    sample = test.groupby("event_class", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.per_class), random_state=0), include_groups=False
    )
    importance = explainer.mean_abs(sample[explainer.features])
    figs = PROJECT_ROOT / "docs" / "figures"
    plot_importance(importance, figs / "shap_importance_per_class.png")
    importance.to_csv(PROJECT_ROOT / "docs" / "shap_importance.csv", float_format="%.4f")
    top = {c: importance[c].sort_values(ascending=False).index[:5].tolist() for c in importance}
    logger.info("Top drivers per class:\n%s", json.dumps(top, indent=1))


if __name__ == "__main__":
    main()
