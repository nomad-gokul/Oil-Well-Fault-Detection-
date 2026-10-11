import numpy as np
import pytest

from owfd.features.context import (
    BLOCK_S,
    CONTEXT_STATS,
    MAX_BLOCKS,
    SCALES,
    all_feature_names,
    block_means,
    context_feature_names,
    context_features,
)
from owfd.features.stats import feature_names


def _features(signal: np.ndarray, end: int | None = None) -> dict[str, float]:
    values = signal.reshape(len(signal), -1)
    end = len(values) - 1 if end is None else end
    blocks = block_means(values, np.zeros_like(values, bool), np.array([end]))
    out = context_features(blocks, np.zeros((1, values.shape[1])))[0]
    return dict(zip(context_feature_names(["s"]), out.tolist(), strict=True))


def test_names() -> None:
    names = context_feature_names(["a", "b"])
    assert len(names) == len(SCALES) * 2 * len(CONTEXT_STATS) == len(set(names))
    assert all_feature_names(["a", "b"]) == feature_names(["a", "b"]) + names


def test_block_means_are_causal_minute_means_of_observed_rows() -> None:
    n = BLOCK_S * 5
    v = np.arange(n, dtype=float)[:, None]
    missing = np.zeros_like(v, bool)
    missing[BLOCK_S * 4 : BLOCK_S * 4 + 10] = True  # excluded from the last block's mean
    b = block_means(v, missing, np.array([n - 1, n - 1 - BLOCK_S]), n_blocks=6)
    assert b.shape == (2, 6, 1)
    assert np.isnan(b[0, 0, 0])  # would start before row 0
    np.testing.assert_allclose(b[0, 1:, 0][:-1], [v[i * 60 : i * 60 + 60].mean() for i in range(4)])
    assert b[0, -1, 0] == pytest.approx(v[250:300].mean())
    np.testing.assert_allclose(b[1, 2:, 0], b[0, 1:-1, 0])  # one minute earlier
    # Rows after the end never matter.
    later = v.copy()
    later[-BLOCK_S:] = 1e6
    np.testing.assert_allclose(block_means(later, missing, np.array([n - 1 - BLOCK_S]), 6), b[1:2])
    # A block without observed rows is NaN.
    all_missing = np.ones_like(v, bool)
    assert np.isnan(block_means(v, all_missing, np.array([n - 1]))).all()


@pytest.mark.parametrize("period_min", [8, 20])
def test_detects_oscillation_period(period_min: int) -> None:
    t = np.arange(BLOCK_S * MAX_BLOCKS * 2)
    f = _features(np.sin(2 * np.pi * t / (period_min * BLOCK_S)))
    assert f["s__c120_period"] == period_min
    assert f["s__c120_acf"] > 0.8
    assert f["s__c120_cross"] == pytest.approx(2 / period_min, abs=0.02)  # two per cycle
    assert (f["s__c120_hf"] > 0.5) == (period_min < 10)
    if period_min * 2 <= 30:  # visible at the 30-minute scale too
        assert f["s__c30_period"] == period_min


def test_trend_level_and_no_cycle() -> None:
    t = np.arange(BLOCK_S * MAX_BLOCKS * 2, dtype=float)
    f = _features(0.5 + t / 3600)  # rises one unit per hour
    assert f["s__c120_slope"] == pytest.approx(1 / 60, rel=1e-6)  # per minute
    assert f["s__c30_dev"] == pytest.approx(0.5 + (len(t) - 15 * 60) / 3600, rel=1e-3)
    assert f["s__c120_acf"] == 0 and f["s__c120_cross"] == 0
    assert np.isnan(f["s__c120_period"]) and np.isnan(f["s__c120_hf"])


def test_too_little_history_is_nan() -> None:
    t = np.arange(BLOCK_S * 40, dtype=float)
    f = _features(np.sin(t / 100))
    assert not np.isnan(f["s__c30_std"])  # 40 blocks >= half of 30
    assert np.isnan(f["s__c120_std"])  # 40 < half of 120
    # Missing sensor: everything NaN.
    v = np.zeros((BLOCK_S * MAX_BLOCKS, 1))
    blocks = block_means(v, np.ones_like(v, bool), np.array([len(v) - 1]))
    assert np.isnan(context_features(blocks, np.zeros((1, 1)))).all()
