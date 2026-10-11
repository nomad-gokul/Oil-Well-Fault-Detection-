"""Train the deep sequence model on raw windows, tracked with MLflow.

Usage::

    python -m owfd.models.train_dl [--config configs/dl.yaml] [--name dl_cnn] [--fast-dev-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import mlflow
import numpy as np
import torch
import yaml
from torch import nn

from owfd.config import PROJECT_ROOT, get_settings
from owfd.data.windows import INPUT_MODES, WindowStore
from owfd.models.metrics import N_CLASSES, classification_metrics, flat_metrics, plot_confusion
from owfd.models.nets import build_model
from owfd.models.report import update_results
from owfd.models.train_xgb import DEFAULT_TRACKING_URI

logger = logging.getLogger(__name__)


class Windows(Protocol):
    """What training needs from a window source (``WindowStore`` or a test double)."""

    def __len__(self) -> int: ...
    def batch(self, idx: np.ndarray) -> np.ndarray: ...
    def labels(self) -> np.ndarray: ...


class StoreWindows:
    """Adapter exposing a ``WindowStore`` through the :class:`Windows` protocol.

    Batches are returned in sorted index order (fast memmap reads); labels follow suit.
    """

    def __init__(
        self, store: WindowStore, use_baseline: bool = True, input_mode: str = "absolute"
    ) -> None:
        self.store = store
        self.use_baseline = use_baseline
        self.transform = INPUT_MODES[input_mode]

    def __len__(self) -> int:
        return len(self.store)

    @property
    def n_channels(self) -> int:
        """Channels per time step fed to the network."""
        n = len(self.store.channels)
        return n + n // 2 if self.use_baseline else n

    def batch(self, idx: np.ndarray) -> np.ndarray:
        idx = np.sort(idx)
        x = self.store.batch(idx)
        return self.transform(x, self.store.baselines(idx)) if self.use_baseline else x

    def labels(self) -> np.ndarray:
        return self.store.windows["event_class"].to_numpy(dtype=np.int64)


def windows_for(store: WindowStore, config: dict[str, Any]) -> StoreWindows:
    """The model input a config asks for (baseline channels and input representation)."""
    return StoreWindows(
        store, config.get("use_baseline", True), config.get("input_mode", "absolute")
    )


def class_weights(y: np.ndarray, cap: float) -> torch.Tensor:
    """Inverse-frequency class weights (classes absent from ``y`` get 0), at most ``cap`` apart."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    present = w > 0
    w[present] = np.minimum(w[present] / w[present].min(), cap)
    w[present] /= (w[present] * counts[present]).sum() / counts[present].sum()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model: nn.Module, data: Windows, batch_size: int = 1024,
                  limit: int | None = None) -> np.ndarray:  # fmt: skip
    """Softmax probabilities [n, n_classes] for the first ``limit`` (default: all) windows."""
    model.eval()
    n = len(data) if limit is None else min(limit, len(data))
    out = []
    for lo in range(0, n, batch_size):
        idx = np.arange(lo, min(lo + batch_size, n))
        out.append(torch.softmax(model(torch.from_numpy(data.batch(idx))), dim=1).numpy())
    return np.concatenate(out) if out else np.empty((0, N_CLASSES), np.float32)


def train_dl(
    config: dict[str, Any],
    train: Windows,
    val: Windows,
    out_dir: Path,
    *,
    n_channels: int,
    seed: int = 42,
    fast_dev_run: bool = False,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train with early stopping on validation macro F1; keeps the best checkpoint.

    ``on_epoch(epoch, val_macro_f1)`` is called after every epoch (e.g. to prune a tuning
    trial by raising an exception).
    """
    tc = config["train"]
    torch.manual_seed(seed)
    torch.set_num_threads(tc["threads"])
    rng = np.random.default_rng(seed)
    model = build_model(config, n_channels, N_CLASSES)
    y_train, y_val = train.labels(), val.labels()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, tc["class_weight_cap"]))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = 1 if fast_dev_run else tc["epochs"]
    per_epoch = 2 * tc["batch_size"] if fast_dev_run else min(tc["windows_per_epoch"], len(train))
    steps = epochs * -(-per_epoch // tc["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=steps,
                                                pct_start=0.1)  # fmt: skip
    val_limit = 2 * tc["batch_size"] if fast_dev_run else None
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, bad_epochs, history = -1.0, 0, []

    for epoch in range(epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        order = rng.choice(len(train), size=per_epoch, replace=False)
        for lo in range(0, per_epoch, tc["batch_size"]):
            idx = order[lo : lo + tc["batch_size"]]
            x, y = torch.from_numpy(train.batch(idx)), torch.from_numpy(y_train[np.sort(idx)])
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        proba = predict_proba(model, val, limit=val_limit)
        m = classification_metrics(y_val[: len(proba)], proba)
        history.append(
            {"epoch": epoch, "train_loss": total / per_epoch, "val_macro_f1": m["macro_f1"],
             "seconds": time.perf_counter() - t0}
        )  # fmt: skip
        logger.info("epoch %d  loss %.4f  val macro F1 %.3f  (%.0f s)", epoch,
                    history[-1]["train_loss"], m["macro_f1"], history[-1]["seconds"])  # fmt: skip
        if on_epoch is not None:
            on_epoch(epoch, m["macro_f1"])
        if m["macro_f1"] > best_f1:
            best_f1, bad_epochs = m["macro_f1"], 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                logger.info("early stopping after epoch %d", epoch)
                break
    model.load_state_dict(torch.load(out_dir / "model.pt"))
    return model, history


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the deep sequence model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dl.yaml")
    parser.add_argument("--name", default=None, help="run name (default: dl_<arch>)")
    parser.add_argument("--fast-dev-run", action="store_true", help="2 batches, smoke test")
    parser.add_argument("--arch", choices=["cnn", "gru"], help="override the config's arch")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    config = yaml.safe_load(args.config.read_text())
    if args.arch:
        config["arch"] = args.arch
    name = args.name or f"dl_{config['arch']}"
    windows_dir = settings.resolve(settings.processed_data_dir) / "windows"
    stores = {s: WindowStore(windows_dir / s) for s in ("train", "val", "test")}
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = settings.resolve(settings.artifacts_dir) / "dl" / f"{name}-{version}"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("oil-well-fault-detection")
    with mlflow.start_run(run_name=name + ("-fast-dev-run" if args.fast_dev_run else "")):
        model, history = train_dl(
            config, windows_for(stores["train"], config), windows_for(stores["val"], config),
            out_dir, n_channels=windows_for(stores["train"], config).n_channels,
            seed=settings.random_seed,
            fast_dev_run=args.fast_dev_run,
        )  # fmt: skip
        limit = 2048 if args.fast_dev_run else None
        metrics = {}
        for split in ("val", "test"):
            data = windows_for(stores[split], config)
            proba = predict_proba(model, data, limit=limit)
            metrics[split] = classification_metrics(data.labels()[: len(proba)], proba)
            plot_confusion(metrics[split], f"{name} — {split}").savefig(
                out_dir / f"confusion_{split}.png", dpi=110
            )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        (out_dir / "config.json").write_text(
            json.dumps(
                {**config, "channels": stores["train"].channels, "seed": settings.random_seed},
                indent=1,
            )
        )
        shutil.copy(settings.resolve(settings.artifacts_dir) / "preprocessing" / "scaler.json",
                    out_dir / "scaler.json")  # fmt: skip
        mlflow.log_params({"arch": config["arch"], **config["train"], **config[config["arch"]]})
        for split, m in metrics.items():
            mlflow.log_metrics(flat_metrics(m, split))
        mlflow.log_artifacts(str(out_dir))

    if not args.fast_dev_run:
        update_results(PROJECT_ROOT / "docs" / "results.md", name, metrics)
    logger.info("Saved %s | val macro F1 %.3f | test macro F1 %.3f", out_dir,
                metrics["val"]["macro_f1"], metrics["test"]["macro_f1"])  # fmt: skip


if __name__ == "__main__":
    main()
