from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.features.build import META_COLUMNS, build_features
from owfd.features.stats import STATS, feature_names, window_features

VARS = ["P-PDG", "P-TPT", "T-TPT"]
T = 100


def window(*columns: np.ndarray, missing: list[np.ndarray] | None = None) -> np.ndarray:
    values = np.stack(columns, axis=1)
    masks = np.stack(missing, axis=1) if missing else np.zeros_like(values)
    return np.concatenate([values, masks], axis=1)[None].astype(np.float32)


def feats(x: np.ndarray) -> pd.Series:
    return pd.Series(window_features(x, VARS)[0], index=feature_names(VARS))


def test_names_match_output_width() -> None:
    names = feature_names(VARS)
    assert len(names) == len(VARS) * len(STATS) + 3  # one pair (P-PDG - P-TPT) applies
    assert len(set(names)) == len(names)
    x = np.zeros((5, T, 2 * len(VARS)), np.float32)
    assert window_features(x, VARS).shape == (5, len(names))
    assert len(feature_names(["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP"])) == 4 * len(STATS) + 6


def test_ramp_constant_and_alternating() -> None:
    ramp = np.arange(T, dtype=float)  # 0..99
    const = np.full(T, 3.0)
    alt = np.tile([1.0, -1.0], T // 2)
    f = feats(window(ramp, const, alt))
    assert f["P-PDG__mean"] == pytest.approx(49.5)
    assert f["P-PDG__min"] == 0 and f["P-PDG__max"] == 99 and f["P-PDG__last"] == 99
    assert f["P-PDG__delta"] == 99
    assert f["P-PDG__slope"] == pytest.approx(T)  # 1 per second over a 100 s window
    assert f["P-PDG__p50"] == pytest.approx(49.5)
    assert f["P-PDG__p05"] == pytest.approx(4.95)
    assert f["P-PDG__vol_ratio"] == pytest.approx(0.0, abs=1e-6)  # same spread first/last 25%
    assert f["P-PDG__turns"] == 0  # monotonic: no local extrema
    assert f["P-TPT__std"] == 0 and f["P-TPT__slope"] == pytest.approx(0)
    assert f["T-TPT__turns"] == pytest.approx((T - 2) / T)  # every inner point is an extremum
    assert f["dp_pdg_tpt__last"] == pytest.approx(96)
    assert f["dp_pdg_tpt__slope"] == pytest.approx(T)


def test_volatility_ratio_detects_growing_oscillation() -> None:
    rng = np.random.default_rng(0)
    growing = rng.normal(0, 1, T) * np.linspace(0.1, 3, T)
    f = feats(window(growing, np.zeros(T), np.zeros(T)))
    assert f["P-PDG__vol_ratio"] > 0.5


def test_missing_variable_gives_nan_stats() -> None:
    zeros = np.zeros(T)
    half = np.r_[np.ones(T // 2), np.zeros(T // 2)]
    f = feats(window(zeros, zeros, zeros, missing=[np.ones(T), half, zeros]))
    assert f["P-PDG__frac_missing"] == 1 and np.isnan(f["P-PDG__mean"])
    assert np.isnan(f["dp_pdg_tpt__mean"])  # pair uses a fully missing variable
    assert f["P-TPT__frac_missing"] == 0.5 and not np.isnan(f["P-TPT__mean"])
    assert f["T-TPT__frac_missing"] == 0


def test_batch_rows_are_independent() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, T, 2 * len(VARS))).astype(np.float32)
    x[:, :, len(VARS) :] = 0
    batch = window_features(x, VARS)
    for i in range(4):
        np.testing.assert_allclose(batch[i], window_features(x[i : i + 1], VARS)[0], rtol=1e-6)


class FakeStore:
    def __init__(self, x: np.ndarray) -> None:
        self._x = x
        self.windows = pd.DataFrame({c: range(len(x)) for c in META_COLUMNS})

    def __len__(self) -> int:
        return len(self._x)

    def batch(self, idx: np.ndarray) -> np.ndarray:
        return self._x[idx]


def test_build_features_chunks_consistently(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, T, 2 * len(VARS))).astype(np.float32)
    x[:, :, len(VARS) :] = 0
    store = FakeStore(x)
    a = build_features(store, VARS, chunk=3)  # type: ignore[arg-type]
    b = build_features(store, VARS, chunk=100)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == META_COLUMNS + feature_names(VARS)
