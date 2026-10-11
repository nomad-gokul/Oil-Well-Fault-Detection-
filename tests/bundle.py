"""Shared serving-bundle fixture: an ensemble (XGBoost + ONNX CNN) built from synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from owfd.config import PreprocessingConfig
from owfd.data.clean import Scaler, clean_instance
from owfd.data.schema import CLASS_NAMES
from owfd.data.windows import add_deviation_channels
from owfd.features.stats import feature_names, window_features
from owfd.models.export import export_onnx
from owfd.models.nets import build_model
from tests.conftest import make_instance

PRE = PreprocessingConfig()  # production defaults: 7 sensors, 30 min stuck rule, 60 s ffill
W, STRIDE, BASE_S = 300, 60, 3600
VARS = PRE.variables


def raw_instance(n: int = 4500, seed: int = 0) -> pd.DataFrame:
    df = make_instance([0] * n, seed=seed).drop(columns=["class", "state"])
    rng = np.random.default_rng(seed)
    df["P-PDG"] = 2.2e7 + np.cumsum(rng.normal(0, 1e3, n))
    df.loc[df.index[1000:3200], "P-TPT"] = 1.4e7  # stuck for 2200 s
    df.loc[df.index[500:540], "T-TPT"] = np.nan  # short gap (forward-filled)
    df.loc[df.index[2000:2400], "T-JUS-CKP"] = np.nan  # long gap (stays missing)
    df["QGL"] = np.nan  # absent sensor
    return df


def build_bundle(out: Path) -> Path:
    """Write a bundle into ``out`` and return it."""
    out.mkdir(parents=True, exist_ok=True)
    raw = raw_instance(seed=1)
    scaler = Scaler.fit([clean_instance(raw, PRE)], PRE)
    scaler.save(out / "scaler.json")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, W, 2 * len(VARS))).astype(np.float32)
    x[:, :, len(VARS) :] = rng.random((300, W, len(VARS))) < 0.1
    base = rng.normal(size=(300, len(VARS))).astype(np.float32)
    names = feature_names(VARS)
    feats = window_features(x, VARS, base)
    y = np.arange(300) % len(CLASS_NAMES)
    keep = [n for n in names if not n.endswith("frac_missing")]  # a subset, like the ablation
    model = xgb.XGBClassifier(
        n_estimators=20, max_depth=3, num_class=10, objective="multi:softprob"
    )
    model.fit(pd.DataFrame(feats, columns=names)[keep], y)
    model.save_model(out / "xgb_0.json")

    torch.manual_seed(0)
    cfg = {"arch": "cnn", "cnn": {"widths": [8, 16], "kernel_size": 5, "dropout": 0.0}}
    net = build_model(cfg, 3 * len(VARS), len(CLASS_NAMES))
    diff = export_onnx(net, add_deviation_channels(x[:8], base[:8]), out / "dl_1.onnx")
    assert diff < 1e-4

    manifest = {
        "version": "test", "model": "ensemble", "temperature": 1.7,
        "members": [
            {"kind": "xgb", "file": "xgb_0.json", "features": keep},
            {"kind": "onnx", "file": "dl_1.onnx", "use_baseline": True},
        ],
        "alert": {"threshold": 0.5, "k": 2}, "preprocessing": PRE.model_dump(),
        "window_s": W, "stride_s": STRIDE, "baseline_s": BASE_S,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out
