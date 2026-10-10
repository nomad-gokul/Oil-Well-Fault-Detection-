"""Vectorized per-window statistics for tree models (XGBoost).

Input: windows ``x`` of shape [n, window_s, 2 * n_vars] (scaled sensors, then missing masks,
see :func:`owfd.data.windows.channels`). Output: [n, n_features] float32.
"""

from __future__ import annotations

import numpy as np

#: Per-variable statistics, in output order.
STATS = (
    "mean", "std", "min", "max", "last", "slope", "delta",
    "p05", "p50", "p95", "vol_ratio", "turns", "frac_missing",
)  # fmt: skip

#: Cross-variable features: (name, variable a, variable b) -> a - b.
PAIRS = (("dp_pdg_tpt", "P-PDG", "P-TPT"), ("dp_tpt_ckp", "P-TPT", "P-MON-CKP"))
PAIR_STATS = ("mean", "last", "slope")


def feature_names(variables: list[str]) -> list[str]:
    """Names of the columns returned by :func:`window_features`, in order."""
    names = [f"{v}__{s}" for v in variables for s in STATS]
    names += [
        f"{p}__{s}" for p, a, b in PAIRS if a in variables and b in variables for s in PAIR_STATS
    ]
    return names


def _slope(x: np.ndarray) -> np.ndarray:
    """Least-squares slope along axis 1 (per second), for x of shape [n, t, k]."""
    t = np.arange(x.shape[1], dtype=np.float64)
    t -= t.mean()
    return np.einsum("ntk,t->nk", x - x.mean(axis=1, keepdims=True), t) / (t @ t)


def window_features(x: np.ndarray, variables: list[str]) -> np.ndarray:
    """Compute :func:`feature_names` for a batch of windows.

    Statistics of a variable that is missing for the whole window are NaN (XGBoost handles
    NaN natively); partially missing variables use the scaled fill value (0).
    """
    k = len(variables)
    v = x[:, :, :k].astype(np.float64)
    m = x[:, :, k : 2 * k] > 0.5
    n, t, _ = v.shape
    q = max(1, t // 4)

    first_std = v[:, :q].std(axis=1)
    last_std = v[:, -q:].std(axis=1)
    turns = (np.diff(np.sign(np.diff(v, axis=1)), axis=1) != 0).sum(axis=1) / t
    p05, p50, p95 = np.percentile(v, [5, 50, 95], axis=1)
    per_stat = {
        "mean": v.mean(axis=1),
        "std": v.std(axis=1),
        "min": v.min(axis=1),
        "max": v.max(axis=1),
        "last": v[:, -1],
        "slope": _slope(v) * t,  # change over the whole window, in scaled units
        "delta": v[:, -1] - v[:, 0],
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "vol_ratio": np.log1p(last_std) - np.log1p(first_std),
        "turns": turns,
        "frac_missing": m.mean(axis=1),
    }
    all_missing: np.ndarray = np.asarray(m.all(axis=1))
    blocks = []
    for s in STATS:
        f = per_stat[s]
        if s != "frac_missing":
            f = np.where(all_missing, np.nan, f)
        blocks.append(f)
    # [n, n_stats, k] -> variable-major order matching feature_names
    out = [np.stack(blocks, axis=1).transpose(0, 2, 1).reshape(n, k * len(STATS))]

    for _, a, b in PAIRS:
        if a in variables and b in variables:
            ia, ib = variables.index(a), variables.index(b)
            d = v[:, :, ia] - v[:, :, ib]
            missing = all_missing[:, ia] | all_missing[:, ib]
            pair = np.stack([d.mean(axis=1), d[:, -1], _slope(d[:, :, None])[:, 0] * t], axis=1)
            out.append(np.where(missing[:, None], np.nan, pair))
    return np.concatenate(out, axis=1).astype(np.float32)
