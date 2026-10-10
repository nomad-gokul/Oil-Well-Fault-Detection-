"""Export the model chosen in ``configs/serving.yaml`` to a self-contained serving bundle.

Usage::

    python -m owfd.models.export   # -> artifacts/serving/<version>/

The bundle needs no PyTorch: deep models are exported to ONNX (outputs verified against
PyTorch), XGBoost models are copied as JSON. Everything the online pipeline needs (scaler,
preprocessing, window and baseline settings, calibration, alert rule) is in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from owfd.config import PROJECT_ROOT, get_settings
from owfd.data.schema import CLASS_NAMES
from owfd.data.windows import WindowStore
from owfd.models.predictions import load_dl
from owfd.models.train_dl import windows_for

logger = logging.getLogger(__name__)


def export_onnx(
    model: torch.nn.Module, example: np.ndarray, path: Path, atol: float = 1e-4
) -> float:
    """Export ``model`` with a dynamic batch axis; return the max |ONNX - PyTorch| on ``example``.

    Raises ``AssertionError`` if the outputs differ by more than ``atol``.
    """
    import onnxruntime as ort

    model.eval()
    x = torch.from_numpy(example.astype(np.float32))
    torch.onnx.export(
        model, (x,), str(path), input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False,
    )  # fmt: skip
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"x": example.astype(np.float32)})[0]
    with torch.no_grad():
        torch_out = model(x).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    assert diff <= atol, f"ONNX output differs from PyTorch by {diff}"
    return diff


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_bundle(
    serving: dict[str, Any], out_dir: Path, example_windows: Path | None
) -> dict[str, Any]:
    """Write the bundle for ``serving`` (contents of configs/serving.yaml); return the manifest."""
    settings = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    scaler_src = None
    for i, rel in enumerate(serving["artifacts"]):
        src = PROJECT_ROOT / rel
        scaler_src = src / "scaler.json"
        if (src / "model.json").exists():
            shutil.copy(src / "model.json", out_dir / f"xgb_{i}.json")
            features = json.loads((src / "feature_names.json").read_text())
            members.append({"kind": "xgb", "file": f"xgb_{i}.json", "features": features,
                            "source": rel})  # fmt: skip
        else:
            model, config = load_dl(src)
            use_baseline = config.get("use_baseline", True)
            if example_windows is not None:
                data = windows_for(WindowStore(example_windows), config)
                example = data.batch(np.arange(min(64, len(data))))
            else:
                n_ch = len(config["channels"]) * (3 if use_baseline else 2) // 2
                example = np.random.default_rng(0).normal(size=(8, 300, n_ch)).astype(np.float32)
            diff = export_onnx(model, example, out_dir / f"dl_{i}.onnx")
            logger.info("ONNX export of %s: max |diff| vs PyTorch = %.2e", rel, diff)
            members.append({"kind": "onnx", "file": f"dl_{i}.onnx", "arch": config["arch"],
                            "use_baseline": use_baseline, "onnx_max_abs_diff": diff,
                            "input_mode": config.get("input_mode", "absolute"),
                            "source": rel})  # fmt: skip
    assert scaler_src is not None
    shutil.copy(scaler_src, out_dir / "scaler.json")
    metrics_file = PROJECT_ROOT / "docs" / "early_warning_comparison.csv"
    metrics: dict[str, Any] = {}
    if metrics_file.exists():
        import pandas as pd

        table = pd.read_csv(metrics_file, index_col="model")
        if serving["model"] in table.index:
            metrics = {
                k: (None if pd.isna(v) else v) for k, v in table.loc[serving["model"]].items()
            }
    manifest = {
        "version": out_dir.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": serving["model"],
        "members": members,
        "temperature": serving["temperature"],
        "alert": serving["alert"],
        "preprocessing": settings.preprocessing.model_dump(),
        "window_s": settings.windows.window_s,
        "stride_s": settings.windows.stride_s,
        "baseline_s": settings.baseline_s,
        "class_names": CLASS_NAMES,
        "metrics": metrics,
    }
    reference = training_reference(settings, PROJECT_ROOT / rel_scaler(serving))
    if reference is not None:
        (out_dir / "reference.json").write_text(json.dumps(reference))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def rel_scaler(serving: dict[str, Any]) -> str:
    """Scaler of the first member (all members share the training scaler)."""
    return f"{serving['artifacts'][0]}/scaler.json"


def training_reference(
    settings: Any, scaler_path: Path, n_instances: int = 400
) -> dict[str, Any] | None:
    """Fleet-drift reference: ranges of window statistics over 2-hour windows of real training
    and validation wells (see ``owfd.monitoring.drift``); None if processed data is missing."""
    import pandas as pd

    from owfd.data.clean import Scaler, mask_column
    from owfd.data.split import load_splits
    from owfd.monitoring.drift import build_reference, deviations_from_baseline

    clean_dir = settings.resolve(settings.processed_data_dir) / "clean"
    splits_file = settings.resolve(settings.splits_file)
    index_file = settings.resolve(settings.processed_data_dir) / "instances.parquet"
    if not (clean_dir.exists() and splits_file.exists() and index_file.exists()):
        logger.warning("No processed training data: bundle without drift reference")
        return None
    scaler = Scaler.load(scaler_path)
    splits = load_splits(splits_file)
    real = set(pd.read_parquet(index_file).query("source == 'real'")["instance_id"])
    ids = sorted(real & set(splits["instances"]["train"] + splits["instances"]["val"]))
    rng = np.random.default_rng(settings.random_seed)
    windows = []
    for iid in rng.choice(ids, size=min(n_instances, len(ids)), replace=False):
        df = scaler.transform(pd.read_parquet(clean_dir / f"{iid}.parquet"))
        for v in scaler.variables:
            df[v] = df[v].where(~df[mask_column(v)])
        dev = deviations_from_baseline(df, scaler.variables, settings.baseline_s)
        start = int(rng.integers(0, max(1, len(dev) - 7200)))
        windows.append(dev.iloc[start : start + 7200])
    return build_reference(windows, scaler.variables)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export the serving bundle.")
    parser.add_argument("--serving", type=Path, default=PROJECT_ROOT / "configs" / "serving.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    serving = yaml.safe_load(args.serving.read_text())
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = settings.resolve(settings.artifacts_dir) / "serving" / version
    val_windows = settings.resolve(settings.processed_data_dir) / "windows" / "val"
    manifest = build_bundle(serving, out_dir, val_windows if val_windows.exists() else None)
    logger.info("Bundle %s (%s) -> %s", manifest["version"], manifest["model"], out_dir)


if __name__ == "__main__":
    main()
