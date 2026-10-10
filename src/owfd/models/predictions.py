"""Per-window class probabilities of trained models, cached next to the model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from owfd.data.windows import WindowStore
from owfd.models.metrics import N_CLASSES
from owfd.models.nets import build_model
from owfd.models.train_dl import predict_proba, windows_for

META = ["instance_id", "well_id", "end_time", "event_class", "is_transient"]
PROBA = [f"p{c}" for c in range(N_CLASSES)]


def model_kind(artifact_dir: Path) -> str:
    """'xgb' or 'dl', from the files in an artifact folder."""
    if (artifact_dir / "model.json").exists():
        return "xgb"
    if (artifact_dir / "model.pt").exists():
        return "dl"
    raise FileNotFoundError(f"No model in {artifact_dir}")


def _xgb_proba(
    artifact_dir: Path, features_dir: Path, split: str
) -> tuple[pd.DataFrame, np.ndarray]:
    cols = json.loads((artifact_dir / "feature_names.json").read_text())
    df = pd.read_parquet(features_dir / f"{split}.parquet")
    model = XGBClassifier()
    model.load_model(artifact_dir / "model.json")
    return df[META], model.predict_proba(df[cols])


def load_dl(artifact_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild a trained deep model from its artifact folder."""
    config = json.loads((artifact_dir / "config.json").read_text())
    n_channels = len(config["channels"]) * (3 if config.get("use_baseline", True) else 2) // 2
    model = build_model(config, n_channels, N_CLASSES)
    model.load_state_dict(torch.load(artifact_dir / "model.pt"))
    model.eval()
    return model, config


def _dl_proba(artifact_dir: Path, windows_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    model, config = load_dl(artifact_dir)
    store = WindowStore(windows_dir / split)
    data = windows_for(store, config)
    return store.windows[META], predict_proba(model, data)


def predictions(artifact_dir: Path, processed_dir: Path, split: str) -> pd.DataFrame:
    """Window metadata + probabilities ``p0..p9`` for one split (cached as parquet)."""
    cache = artifact_dir / f"predictions_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if model_kind(artifact_dir) == "xgb":
        meta, proba = _xgb_proba(artifact_dir, processed_dir / "features", split)
    else:
        meta, proba = _dl_proba(artifact_dir, processed_dir / "windows", split)
    out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(proba, columns=PROBA)], axis=1)
    out.to_parquet(cache)
    return out
