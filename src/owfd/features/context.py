"""Long-context features: how each sensor behaved over the last 30 and 120 minutes.

The 5-minute window statistics (:mod:`owfd.features.stats`) cannot see slow oscillations
such as severe slugging, whose cycles are often longer than five minutes. Here each sensor
is summarised as one-minute block means (observed rows only) ending at the window end, and
the last ``scale`` blocks are described by spread, level, trend and periodicity.

Blocks end at the window end and every :data:`BLOCK_S` rows before it, so offline (from a
whole instance, :func:`block_means`) and online (one block per minute, see
``owfd.inference.predictor``) see exactly the same blocks.
"""

from __future__ import annotations

import numpy as np

#: Rows (seconds at 1 Hz) per block.
BLOCK_S = 60
#: Context lengths, in blocks (minutes).
SCALES = (30, 120)
MAX_BLOCKS = max(SCALES)
#: Per-variable statistics for each scale, in output order.
CONTEXT_STATS = ("std", "range", "dev", "slope", "cross", "acf", "period", "hf")
#: Share of a scale's blocks that must be observed; otherwise its statistics are NaN.
MIN_OBSERVED = 0.5
#: Oscillations faster than this (minutes per cycle) count as high frequency.
HF_PERIOD_MIN = 10.0


def all_feature_names(variables: list[str]) -> list[str]:
    """Every tabular feature: 5-minute window statistics, then long-context features."""
    from owfd.features.stats import feature_names

    return feature_names(variables) + context_feature_names(variables)


def context_feature_names(variables: list[str]) -> list[str]:
    """Names of the columns returned by :func:`context_features`, in order."""
    return [f"{v}__c{s}_{st}" for s in SCALES for v in variables for st in CONTEXT_STATS]


def block_means(
    values: np.ndarray, missing: np.ndarray, ends: np.ndarray, n_blocks: int = MAX_BLOCKS
) -> np.ndarray:
    """Block means [len(ends), n_blocks, k], oldest block first.

    Block ``j`` (counting back from the newest) covers rows ``e - (j + 1) * BLOCK_S + 1 .. e - j
    * BLOCK_S`` for a window ending at row ``e``. Only observed rows (``missing`` False) count;
    a block with none, or starting before row 0, is NaN. Causal: uses rows ``<= e`` only.
    """
    observed = ~np.asarray(missing, dtype=bool)
    k = values.shape[1]
    csum = np.zeros((len(values) + 1, k))
    csum[1:] = np.cumsum(np.where(observed, values, 0.0), axis=0, dtype=np.float64)
    ccnt = np.zeros((len(values) + 1, k), dtype=np.int64)
    ccnt[1:] = np.cumsum(observed, axis=0)
    back = BLOCK_S * np.arange(n_blocks - 1, -1, -1)
    stop = np.asarray(ends)[:, None] - back + 1  # index into the padded cumsums
    start = stop - BLOCK_S
    complete = start >= 0
    s0, s1 = np.clip(start, 0, None), np.clip(stop, 0, None)
    sums, counts = csum[s1] - csum[s0], ccnt[s1] - ccnt[s0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(complete[:, :, None] & (counts > 0), sums / counts, np.nan)


def _scale_stats(b: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Statistics [n, k, n_stats] of blocks ``b`` [n, L, k] (NaN = unobserved)."""
    n, length, _ = b.shape
    obs = ~np.isnan(b)
    n_obs = obs.sum(axis=1)
    enough = n_obs >= max(4, int(np.ceil(MIN_OBSERVED * length)))
    safe = np.maximum(n_obs, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(obs, b, 0.0)
        mean = filled.sum(axis=1) / safe
        t = np.where(obs, np.arange(length, dtype=np.float64)[None, :, None], 0.0)
        t_mean = t.sum(axis=1) / safe
        dt = np.where(obs, t - t_mean[:, None, :], 0.0)
        dx = np.where(obs, b - mean[:, None, :], 0.0)
        var_t = (dt * dt).sum(axis=1)
        slope = np.where(var_t > 0, (dt * dx).sum(axis=1) / var_t, 0.0)  # per minute
        resid = np.where(obs, dx - slope[:, None, :] * dt, 0.0)  # detrended, 0 where missing
        # Round-off of an exact line is not an oscillation.
        scale = np.abs(dx).max(axis=1, keepdims=True)
        resid = np.where(np.abs(resid) <= 1e-9 * np.maximum(scale, 1e-12), 0.0, resid)
        std = np.sqrt((dx * dx).sum(axis=1) / safe)
        rng = np.where(obs, b, -np.inf).max(axis=1) - np.where(obs, b, np.inf).min(axis=1)

        # Mean crossings of the detrended series (oscillation count), per minute.
        sign = np.sign(resid)
        cross = (sign[:, 1:] * sign[:, :-1] < 0).sum(axis=1) / (length - 1)

        # Autocorrelation (biased estimator via FFT). A cycle shows up as a peak after the
        # autocorrelation has first dropped to zero (before that, it only reflects smoothness).
        spec = np.fft.rfft(resid, n=2 * length, axis=1)
        acov = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :length]
        acov0 = acov[:, 0]
        lags = np.arange(1, length // 2 + 1)
        acf = acov[:, lags] / np.where(acov0 > 0, acov0, 1.0)[:, None, :]
        after_zero = np.maximum.accumulate(acf <= 0, axis=1)
        candidates = np.where(after_zero, acf, -np.inf)
        best = candidates.argmax(axis=1)
        acf_max = np.take_along_axis(candidates, best[:, None, :], axis=1)[:, 0]
        no_cycle = ~after_zero.any(axis=1)
        acf_max = np.where(no_cycle, 0.0, acf_max)
        period = np.where(no_cycle, np.nan, lags[best].astype(np.float64))

        # Share of (non-DC) power in cycles faster than HF_PERIOD_MIN minutes.
        power = np.abs(np.fft.rfft(resid, axis=1)) ** 2
        freqs = np.fft.rfftfreq(length)  # cycles per minute
        total = power[:, 1:].sum(axis=1)
        hf = power[:, freqs > 1.0 / HF_PERIOD_MIN].sum(axis=1) / np.where(total > 0, total, 1.0)

    flat = acov0 <= 1e-12  # no variation: periodicity is undefined
    acf_max = np.where(flat, 0.0, acf_max)
    period = np.where(flat, np.nan, period)
    hf = np.where(flat, np.nan, hf)
    stats = np.stack(
        [std, rng, mean - baseline, slope, cross, acf_max, period, hf], axis=2
    )  # [n, k, n_stats]
    return np.where(enough[:, :, None], stats, np.nan)


def context_features(blocks: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Compute :func:`context_feature_names` from blocks [n, >= MAX_BLOCKS, k] (oldest first,
    NaN = unobserved) and stream baselines [n, k] (NaN where unknown)."""
    blocks = np.asarray(blocks, dtype=np.float64)
    n, _, k = blocks.shape
    base = np.asarray(baseline, dtype=np.float64)
    out = [_scale_stats(blocks[:, -s:, :], base).reshape(n, k * len(CONTEXT_STATS)) for s in SCALES]
    return np.concatenate(out, axis=1).astype(np.float32)
