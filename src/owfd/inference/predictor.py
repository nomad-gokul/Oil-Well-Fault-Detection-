"""Production inference: the exact training-time preprocessing, without PyTorch.

A serving bundle (``artifacts/serving/<version>/``, written by ``owfd.models.export``) holds
the scaler, the model(s) (XGBoost JSON and/or ONNX), the calibration temperature and the
alert rule. :class:`StreamingPredictor` turns 1 Hz readings per well into a prediction every
``stride_s`` seconds.

Usage::

    python -m owfd.inference.predictor --instance data/raw/3W/dataset/7/WELL-00022_....parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import PreprocessingConfig
from owfd.data.clean import Scaler, clean_instance
from owfd.data.schema import CLASS_NAMES
from owfd.data.windows import INPUT_MODES, channels
from owfd.features.context import (
    BLOCK_S,
    MAX_BLOCKS,
    all_feature_names,
    block_means,
    context_feature_names,
    context_features,
)
from owfd.features.stats import window_features

N_CLASSES = len(CLASS_NAMES)


@dataclass
class Prediction:
    """Model output for the window ending at ``timestamp``."""

    well_id: str
    timestamp: pd.Timestamp
    predicted_class: int
    class_name: str
    p_fault: float
    probabilities: list[float]
    #: (scaled window [window_s, 2k], baseline [k], context blocks [MAX_BLOCKS, k]) when
    #: requested, e.g. for explanations.
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (without ``inputs``)."""
        d = {k: v for k, v in asdict(self).items() if k != "inputs"}
        d["timestamp"] = self.timestamp.isoformat()
        return d


def load_booster(path: Path) -> Any:
    """Load an XGBoost model from ``.json``/``.ubj``, or a gzipped ``.json.gz``."""
    import gzip

    import xgboost as xgb

    booster = xgb.Booster()
    if path.suffix == ".gz":
        booster.load_model(bytearray(gzip.decompress(path.read_bytes())))
    else:
        booster.load_model(path)
    return booster


def _mean_or_nan(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Element-wise ``total / count`` as float32, NaN where ``count`` is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan).astype(np.float32)


def _softmax_t(proba: np.ndarray, temperature: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-9, 1)) / temperature
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


class Predictor:
    """Loads a serving bundle and scores preprocessed windows."""

    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.manifest: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
        self.scaler = Scaler.load(bundle / "scaler.json")
        self.preprocessing = PreprocessingConfig(**self.manifest["preprocessing"])
        self.window_s: int = self.manifest["window_s"]
        self.stride_s: int = self.manifest["stride_s"]
        self.baseline_s: int = self.manifest["baseline_s"]
        self.temperature: float = self.manifest["temperature"]
        self.variables = list(self.preprocessing.variables)
        self.channels = channels(self.variables)
        self._all_features = all_feature_names(self.variables)
        self._members: list[tuple[str, Any, dict[str, Any]]] = []
        for member in self.manifest["members"]:
            if member["kind"] == "xgb":
                booster = load_booster(bundle / member["file"])
                cols = [self._all_features.index(f) for f in member["features"]]
                self._members.append(("xgb", booster, {**member, "cols": cols}))
            elif member["kind"] == "onnx":
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                session = ort.InferenceSession(str(bundle / member["file"]), opts,
                                               providers=["CPUExecutionProvider"])  # fmt: skip
                self._members.append(("onnx", session, member))
            else:
                raise ValueError(f"Unknown member kind {member['kind']}")
        context = set(context_feature_names(self.variables))
        #: Whether any member uses long-context features (older bundles do not).
        self.uses_context = any(
            context & set(meta["features"]) for kind, _, meta in self._members if kind == "xgb"
        )
        if self.uses_context and (self.window_s % BLOCK_S or self.stride_s % BLOCK_S):
            raise ValueError(f"window_s and stride_s must be multiples of {BLOCK_S}")

    @classmethod
    def load(cls, bundle: Path) -> Predictor:
        """Load a bundle folder."""
        return cls(bundle)

    def _features(
        self, x: np.ndarray, baseline: np.ndarray, blocks: np.ndarray | None
    ) -> np.ndarray:
        """All tabular features [n, n_features]; context features are NaN without ``blocks``
        (or skipped entirely when no member uses them)."""
        short = window_features(x, self.variables, baseline)
        n_context = len(self._all_features) - short.shape[1]
        if not self.uses_context:
            return np.concatenate([short, np.full((len(x), n_context), np.nan, np.float32)], 1)
        if blocks is None:
            blocks = np.full((len(x), MAX_BLOCKS, len(self.variables)), np.nan)
        return np.concatenate([short, context_features(blocks, baseline)], axis=1)

    def predict_proba(
        self, x: np.ndarray, baseline: np.ndarray, blocks: np.ndarray | None = None
    ) -> np.ndarray:
        """Calibrated class probabilities [n, 10] for scaled windows ``x`` [n, window_s, 2k],
        stream baselines [n, k] and one-minute context blocks [n, MAX_BLOCKS, k] (see
        :func:`owfd.features.context.block_means`; computed from float16-rounded values)."""
        # The training window store keeps scaled values as float16 (owfd.data.windows), so the
        # models were fit on float16-rounded inputs: round identically to avoid serving skew.
        x = x.astype(np.float16).astype(np.float32)
        probas = []
        feats_all: np.ndarray | None = None
        for kind, model, meta in self._members:
            if kind == "xgb":
                import xgboost as xgb

                if feats_all is None:
                    feats_all = self._features(x, baseline, blocks)
                feats = feats_all[:, meta["cols"]]
                dm = xgb.DMatrix(feats, feature_names=meta["features"])
                probas.append(np.asarray(model.predict(dm)).reshape(len(x), N_CLASSES))
            else:
                transform = INPUT_MODES[meta.get("input_mode", "absolute")]
                inp = transform(x, baseline) if meta.get("use_baseline", True) else x
                logits = model.run(None, {model.get_inputs()[0].name: inp.astype(np.float32)})[0]
                logits = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(logits)
                probas.append(e / e.sum(axis=1, keepdims=True))
        return _softmax_t(np.mean(probas, axis=0), self.temperature)

    def features(
        self, x: np.ndarray, baseline: np.ndarray, blocks: np.ndarray | None = None
    ) -> np.ndarray:
        """Every tabular feature [n, n_features] (``owfd.features.context.all_feature_names``
        order) of windows ``x`` [n, window_s, 2k], rounded to float16 like ``predict_proba``."""
        x = x.astype(np.float16).astype(np.float32)
        if blocks is None:
            blocks = np.full((len(x), MAX_BLOCKS, len(self.variables)), np.nan)
        short = window_features(x, self.variables, baseline)
        return np.concatenate([short, context_features(blocks, baseline)], axis=1)

    def xgb_features(
        self, x: np.ndarray, baseline: np.ndarray, blocks: np.ndarray | None = None
    ) -> np.ndarray:
        """Features of the first XGBoost member (its column subset), for explanations."""
        meta = next(m for kind, _, m in self._members if kind == "xgb")
        x = x.astype(np.float16).astype(np.float32)
        return self._features(x, baseline, blocks)[:, meta["cols"]]

    def predict_window(
        self, raw: pd.DataFrame, well_id: str = "unknown", keep_inputs: bool = False
    ) -> Prediction:
        """Score the last ``window_s`` rows of ``raw`` (1 Hz readings of one well, oldest first).

        All of ``raw`` is used as history: for stuck-sensor detection and for the stream
        baseline (its first ``baseline_s`` rows). Raises ``ValueError`` if too short.
        """
        if len(raw) < self.window_s:
            raise ValueError(f"Need at least {self.window_s} rows, got {len(raw)}")
        scaled = self.scaler.transform(clean_instance(raw, self.preprocessing))
        values = scaled[self.channels].to_numpy(np.float32)
        k = len(self.variables)
        head = values[: min(len(values), self.baseline_s)]
        observed = head[:, k:] < 0.5
        base = _mean_or_nan((head[:, :k] * observed).sum(axis=0), observed.sum(axis=0))
        x = values[-self.window_s :]
        v16 = values.astype(np.float16).astype(np.float64)
        blocks = block_means(v16[:, :k], v16[:, k:] > 0.5, np.array([len(values) - 1]))
        proba = self.predict_proba(x[None], base[None], blocks)[0]
        inputs = (x, base, blocks[0]) if keep_inputs else None
        return self._prediction(well_id, raw.index[-1], proba, inputs)

    def _prediction(
        self,
        well_id: str,
        ts: Any,
        proba: np.ndarray,
        inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> Prediction:
        c = int(np.argmax(proba))
        probs = [float(p) for p in proba]
        return Prediction(well_id, pd.Timestamp(ts), c, CLASS_NAMES[c], 1 - probs[0], probs, inputs)


@dataclass
class _WellState:
    buffer: pd.DataFrame
    n_seen: int = 0
    base_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    base_cnt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Means of the last MAX_BLOCKS completed one-minute blocks [n, k], oldest first.
    blocks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


class StreamingPredictor:
    """Per-well rolling buffers; emits a :class:`Prediction` every ``stride_s`` readings.

    Readings must arrive in time order at 1 Hz. The buffer keeps
    ``window_s + stuck_after_s + ffill_limit_s + 1`` rows, enough for every preprocessing step
    to see the same history as in training, so online predictions equal offline ones. Long
    context is kept as one mean per completed minute (computed while its rows are in the
    buffer, so with the same history as offline).
    """

    def __init__(self, predictor: Predictor) -> None:
        self.p = predictor
        pre = predictor.preprocessing
        # A stuck run is masked once it is `stuck_after_s` long, and forward-fill then patches
        # up to `ffill_limit_s` rows after the mask starts: both must be visible in the buffer.
        self.keep = predictor.window_s + pre.stuck_after_s + pre.ffill_limit_s + 1
        self.wells: dict[str, _WellState] = {}

    def reset(self, well_id: str) -> None:
        """Forget a well (e.g. after a restart of its data stream)."""
        self.wells.pop(well_id, None)

    def update(
        self, well_id: str, readings: pd.DataFrame, keep_inputs: bool = False
    ) -> list[Prediction]:
        """Add readings (DatetimeIndex, sensor columns) and return any new predictions.

        With ``keep_inputs`` each prediction carries its window and baseline.
        """
        if readings.empty:
            return []
        k = len(self.p.variables)
        state = self.wells.get(well_id)
        if state is None:
            state = _WellState(readings.iloc[:0], 0, np.zeros(k), np.zeros(k), np.zeros((0, k)))
            self.wells[well_id] = state
        old_len = len(state.buffer)
        frame = pd.concat([state.buffer, readings]) if old_len else readings
        scaled = self.p.scaler.transform(clean_instance(frame, self.p.preprocessing))
        values = scaled[self.p.channels].to_numpy(np.float32)
        first_pos = state.n_seen  # stream position of readings[0]

        positions = np.arange(first_pos, first_pos + len(readings))
        new = values[old_len:]
        # Baseline: observed values in the first `baseline_s` rows of the stream (causal).
        counted = (positions < self.p.baseline_s)[:, None] & (new[:, k:] < 0.5)
        cum_sum = state.base_sum + np.cumsum(np.where(counted, new[:, :k], 0.0), axis=0)
        cum_cnt = state.base_cnt + np.cumsum(counted, axis=0)
        # Rows that end a window, aligned like the training windows.
        w, stride = self.p.window_s, self.p.stride_s
        is_end = (positions >= w - 1) & ((positions - (w - 1)) % stride == 0)
        out: list[Prediction] = []
        js = np.flatnonzero(is_end)
        history = state.blocks
        ctx: np.ndarray | None = None
        if self.p.uses_context:
            # One-minute blocks ending in this batch (window ends are block ends too).
            bj = np.flatnonzero(((positions + 1) % BLOCK_S == 0) & (positions >= BLOCK_S - 1))
            v16 = values.astype(np.float16).astype(np.float64)
            new_blocks = block_means(v16[:, :k], v16[:, k:] > 0.5, old_len + bj, n_blocks=1)
            history = np.concatenate([state.blocks, new_blocks[:, 0, :]])
            # Index of each window end's block in `history`, then its last MAX_BLOCKS blocks.
            at = len(state.blocks) + np.searchsorted(bj, js)
            padded = np.concatenate([np.full((MAX_BLOCKS - 1, k), np.nan), history])
            ctx = np.stack([padded[i : i + MAX_BLOCKS] for i in at]) if js.size else None
        if js.size:
            rows = old_len + js
            windows = np.stack([values[r - w + 1 : r + 1] for r in rows])
            bases = _mean_or_nan(cum_sum[js], cum_cnt[js])
            proba = self.p.predict_proba(windows, bases, ctx)
            blank = np.full((MAX_BLOCKS, k), np.nan)
            out = [
                self.p._prediction(
                    well_id, frame.index[r], pr,
                    (windows[n], bases[n], ctx[n] if ctx is not None else blank)
                    if keep_inputs else None,
                )
                for n, (r, pr) in enumerate(zip(rows, proba, strict=True))
            ]  # fmt: skip
        state.blocks = history[-MAX_BLOCKS:]
        state.base_sum, state.base_cnt = cum_sum[-1], cum_cnt[-1]
        state.n_seen += len(readings)
        state.buffer = frame.iloc[-self.keep :]
        return out


def main(argv: list[str] | None = None) -> None:
    """Replay one 3W instance through the streaming predictor and print predictions."""
    from owfd.config import get_settings
    from owfd.data.load import load_instance

    parser = argparse.ArgumentParser(description="Stream a 3W instance through the model.")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, help="serving bundle (default: latest)")
    args = parser.parse_args(argv)
    settings = get_settings()
    bundle = (
        args.bundle or sorted((settings.resolve(settings.artifacts_dir) / "serving").iterdir())[-1]
    )
    stream = StreamingPredictor(Predictor.load(bundle))
    df = load_instance(args.instance)
    print(f"{'time':19s}  {'label':>5s}  {'predicted':28s}  P(fault)")
    for lo in range(0, len(df), 600):
        chunk = df.iloc[lo : lo + 600]
        for p in stream.update(
            args.instance.stem, chunk.drop(columns=["class", "state"], errors="ignore")
        ):
            label = df.loc[p.timestamp, "class"]
            print(f"{p.timestamp:%Y-%m-%d %H:%M:%S}  {str(label):>5s}  "
                  f"{p.predicted_class} {p.class_name:26s}  {p.p_fault:.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
