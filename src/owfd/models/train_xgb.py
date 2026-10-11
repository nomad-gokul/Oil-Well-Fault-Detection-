"""Train and evaluate the XGBoost baseline, tracked with MLflow.

Usage::

    python -m owfd.models.train_xgb [--config configs/xgb.yaml] [--name xgb_baseline]
                                    [--no-missing-features] [--drop REGEX]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from owfd.config import PROJECT_ROOT, get_settings
from owfd.models.metrics import N_CLASSES, classification_metrics, flat_metrics, plot_confusion
from owfd.models.report import update_results

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def load_split(
    features_dir: Path, split: str, columns: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix (``columns`` only) and event-class labels of one split."""
    df = pd.read_parquet(features_dir / f"{split}.parquet")
    return df[columns], df["event_class"].to_numpy(dtype=int)


def balanced_weights(y: np.ndarray, cap: float) -> np.ndarray:
    """Inverse-frequency sample weights, at most ``cap`` x apart, with mean 1."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    present = per_class > 0
    per_class[present] = np.minimum(per_class[present] / per_class[present].min(), cap)
    w = per_class[y]
    return w / w.mean()


def select_columns(
    names: list[str], use_missing_features: bool, drop: str | None = None
) -> list[str]:
    """Feature columns, optionally without the missing-data indicators (shortcut ablation) and
    without names matching the regular expression ``drop`` (feature-group ablations)."""
    cols = names if use_missing_features else [n for n in names if not n.endswith("frac_missing")]
    return [n for n in cols if not re.search(drop, n)] if drop else cols


def select_n_trees(
    model: XGBClassifier, x_val: pd.DataFrame, y_val: np.ndarray, every: int
) -> tuple[int, dict[int, float]]:
    """Number of boosting rounds with the best validation macro F1 (checked every ``every``)."""
    total = model.get_booster().num_boosted_rounds()
    curve = {}
    for k in [*range(every, total, every), total]:
        proba = model.predict_proba(x_val, iteration_range=(0, k))
        curve[k] = classification_metrics(y_val, proba)["macro_f1"]
    best = max(curve, key=lambda k: (curve[k], -k))  # ties -> fewer trees
    return best, curve


def truncate(model: XGBClassifier, n_trees: int) -> XGBClassifier:
    """A classifier holding only the first ``n_trees`` boosting rounds of ``model``."""
    booster = model.get_booster()[:n_trees]
    small = XGBClassifier()
    small.load_model(bytearray(booster.save_raw("json")))
    return small


def train_xgb(
    config: dict[str, Any],
    features_dir: Path,
    out_dir: Path,
    *,
    use_missing_features: bool = True,
    seed: int = 42,
    drop: str | None = None,
) -> tuple[XGBClassifier, dict[str, dict[str, Any]]]:
    """Fit on train, keep the tree count with the best val macro F1, evaluate, save artifacts."""
    names = json.loads((features_dir / "feature_names.json").read_text())
    columns = select_columns(names, use_missing_features, drop)
    x_train, y_train = load_split(features_dir, "train", columns)
    x_val, y_val = load_split(features_dir, "val", columns)
    cap = config["class_weight_cap"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        random_state=seed,
        n_jobs=config["n_jobs"],
        **config["model"],
    )
    model.fit(x_train, y_train, sample_weight=balanced_weights(y_train, cap))
    n_trees, curve = select_n_trees(model, x_val, y_val, config["select_every"])
    model = truncate(model, n_trees)

    metrics = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        x, y = (x_val, y_val) if split == "val" else load_split(features_dir, split, columns)
        metrics[split] = classification_metrics(y, model.predict_proba(x))
        fig = plot_confusion(metrics[split], f"XGBoost — {split}")
        fig.savefig(out_dir / f"confusion_{split}.png", dpi=110)

    model.save_model(out_dir / "model.json")
    (out_dir / "feature_names.json").write_text(json.dumps(columns, indent=1))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **config,
                "use_missing_features": use_missing_features,
                "drop": drop,
                "seed": seed,
                "n_trees": n_trees,
                "val_macro_f1_curve": curve,
            },
            indent=1,
        )  # fmt: skip
    )
    return model, metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the XGBoost baseline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "xgb.yaml")
    parser.add_argument("--name", default="xgb_baseline")
    parser.add_argument("--no-missing-features", action="store_true")
    parser.add_argument("--drop", help="regex: leave out matching features (ablation)")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    config = yaml.safe_load(args.config.read_text())
    features_dir = settings.resolve(settings.processed_data_dir) / "features"
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = settings.resolve(settings.artifacts_dir) / "xgb" / f"{args.name}-{version}"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("oil-well-fault-detection")
    with mlflow.start_run(run_name=args.name):
        _, metrics = train_xgb(
            config, features_dir, out_dir,
            use_missing_features=not args.no_missing_features, seed=settings.random_seed,
            drop=args.drop,
        )  # fmt: skip
        shutil.copy(settings.resolve(settings.artifacts_dir) / "preprocessing" / "scaler.json",
                    out_dir / "scaler.json")  # fmt: skip
        mlflow.log_params({**config["model"], "use_missing_features": not args.no_missing_features,
                           "class_weight_cap": config["class_weight_cap"]})  # fmt: skip
        for split, m in metrics.items():
            mlflow.log_metrics(flat_metrics(m, split))
        mlflow.log_artifacts(str(out_dir))

    update_results(PROJECT_ROOT / "docs" / "results.md", args.name, metrics)
    logger.info("Saved %s | val macro F1 %.3f | test macro F1 %.3f", out_dir,
                metrics["val"]["macro_f1"], metrics["test"]["macro_f1"])  # fmt: skip


if __name__ == "__main__":
    main()
