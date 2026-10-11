import json
from pathlib import Path

import numpy as np
import pytest
import torch

from owfd.data.windows import add_deviation_channels
from owfd.models.metrics import N_CLASSES
from owfd.models.nets import build_model
from owfd.models.train_dl import class_weights, predict_proba, train_dl

T, C = 60, 6


class ArrayWindows:
    """In-memory stand-in for a window store (the class is encoded in channel 0's level)."""

    def __init__(self, n: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.y = np.arange(n) % N_CLASSES
        self.x = rng.normal(0, 0.3, (n, T, C)).astype(np.float32)
        self.x[:, :, 0] += self.y[:, None]

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, idx: np.ndarray) -> np.ndarray:
        return self.x[np.sort(idx)]

    def labels(self) -> np.ndarray:
        return self.y.astype(np.int64)


def config(arch: str, epochs: int = 40, lr: float = 0.01) -> dict:
    return {
        "arch": arch,
        "cnn": {"widths": [16, 32], "kernel_size": 5, "dropout": 0.0},
        "gru": {"hidden": 32, "layers": 1, "pool": 5, "dropout": 0.0},
        "train": {
            "epochs": epochs,
            "windows_per_epoch": 64,
            "batch_size": 16,
            "lr": lr,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "patience": 100,
            "class_weight_cap": 30.0,
            "threads": 2,
        },  # fmt: skip
    }


@pytest.mark.parametrize("arch", ["cnn", "gru"])
def test_models_output_shape(arch: str) -> None:
    model = build_model(config(arch), C, N_CLASSES)
    assert model(torch.zeros(3, T, C)).shape == (3, N_CLASSES)


def test_unknown_arch() -> None:
    with pytest.raises(ValueError):
        build_model({"arch": "x"}, C, N_CLASSES)


@pytest.mark.parametrize("arch", ["cnn", "gru"])
def test_overfits_a_tiny_dataset(arch: str, tmp_path: Path) -> None:
    """Sanity check: the training loop can drive 64 windows to ~100% train accuracy."""
    data = ArrayWindows(64)
    epochs = 40 if arch == "cnn" else 100
    model, history = train_dl(config(arch, epochs), data, data, tmp_path, n_channels=C)
    acc = (predict_proba(model, data).argmax(1) == data.labels()).mean()
    assert acc >= 0.95, (acc, history[-1])
    assert (tmp_path / "model.pt").exists()


def test_fast_dev_run_and_determinism(tmp_path: Path) -> None:
    train, val = ArrayWindows(200), ArrayWindows(50, seed=1)
    cfg = config("cnn", epochs=5)
    _, h1 = train_dl(cfg, train, val, tmp_path / "a", n_channels=C, fast_dev_run=True)
    _, h2 = train_dl(cfg, train, val, tmp_path / "b", n_channels=C, fast_dev_run=True)
    assert len(h1) == 1  # one epoch only
    assert h1[0]["train_loss"] == pytest.approx(h2[0]["train_loss"])


def test_class_weights() -> None:
    w = class_weights(np.array([0] * 90 + [1] * 10), cap=5.0)
    assert w[1] / w[0] == pytest.approx(5.0)
    assert w[2:].sum() == 0  # absent classes


def test_add_deviation_channels() -> None:
    x = np.ones((2, 4, 4), np.float32)  # 2 sensors + 2 masks
    base = np.array([[0.5, np.nan], [1.0, 2.0]], np.float32)
    out = add_deviation_channels(x, base)
    assert out.shape == (2, 4, 6)
    np.testing.assert_allclose(out[0, :, 4], 0.5)
    np.testing.assert_allclose(out[0, :, 5], 0.0)  # unknown baseline -> 0
    np.testing.assert_allclose(out[1, :, 5], -1.0)
    assert json.dumps(out.dtype.name) == '"float32"'
