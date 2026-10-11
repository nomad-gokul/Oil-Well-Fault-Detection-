from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.data.clean import clean_instance
from owfd.data.schema import CLASS_NAMES
from owfd.data.windows import causal_baseline, channels
from owfd.inference.predictor import Predictor, StreamingPredictor
from tests.bundle import BASE_S, PRE, STRIDE, VARS, W, build_bundle, raw_instance


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_bundle(tmp_path_factory.mktemp("bundle"))


def offline_predictions(p: Predictor, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The training pipeline on the whole instance at once: (window end rows, probabilities)."""
    scaled = p.scaler.transform(clean_instance(raw, PRE))
    values = scaled[channels(VARS)].to_numpy(np.float32)
    ends = np.arange(W - 1, len(raw), STRIDE)
    k = len(VARS)
    base = causal_baseline(values[:, :k].astype(np.float64), values[:, k:], ends, BASE_S)
    x = np.stack([values[e - W + 1 : e + 1] for e in ends])
    return ends, p.predict_proba(x, base.astype(np.float32))


@pytest.mark.parametrize("chunk_seed", [0, 1])
def test_streaming_equals_offline(bundle: Path, chunk_seed: int) -> None:
    p = Predictor.load(bundle)
    raw = raw_instance()
    ends, expected = offline_predictions(p, raw)

    stream = StreamingPredictor(p)
    rng = np.random.default_rng(chunk_seed)
    got, lo = [], 0
    while lo < len(raw):  # random chunk sizes: 1 reading up to 15 minutes at a time
        n = int(rng.integers(1, 900))
        got += stream.update("w1", raw.iloc[lo : lo + n])
        lo += n
    assert [g.timestamp for g in got] == list(raw.index[ends])
    np.testing.assert_allclose([g.probabilities for g in got], expected, atol=1e-5)
    assert len(stream.wells["w1"].buffer) == stream.keep  # bounded memory
    assert stream.keep == W + PRE.stuck_after_s + PRE.ffill_limit_s + 1


def test_predict_window_matches_last_streaming_prediction(bundle: Path) -> None:
    p = Predictor.load(bundle)
    raw = raw_instance().iloc[: W + 60 * 40]  # ends exactly on a window boundary
    last = StreamingPredictor(p).update("w", raw)[-1]
    single = p.predict_window(raw, well_id="w")
    assert single.timestamp == last.timestamp == raw.index[-1]
    np.testing.assert_allclose(single.probabilities, last.probabilities, atol=1e-5)
    assert 0 <= single.p_fault <= 1 and abs(sum(single.probabilities) - 1) < 1e-5
    assert single.to_dict()["class_name"] == CLASS_NAMES[single.predicted_class]


def test_wells_are_independent_and_short_input_is_rejected(bundle: Path) -> None:
    p = Predictor.load(bundle)
    stream = StreamingPredictor(p)
    a, b = raw_instance(1000, seed=2), raw_instance(1000, seed=3)
    solo = StreamingPredictor(p).update("a", a)
    stream.update("b", b.iloc[:500])
    mixed = stream.update("a", a)
    np.testing.assert_allclose([m.probabilities for m in mixed], [s.probabilities for s in solo])
    with pytest.raises(ValueError):
        p.predict_window(a.iloc[:100])
    assert stream.update("a", a.iloc[:0]) == []


def test_serving_features_equal_training_store_features(bundle: Path, tmp_path: Path) -> None:
    """Guards against training/serving skew: the window store (float16) and the predictor
    must produce identical XGBoost features for the same window."""
    from owfd.config import WindowConfig
    from owfd.data.load import build_index, load_instance
    from owfd.data.windows import WindowStore, build_split

    p = Predictor.load(bundle)
    root = tmp_path / "dataset" / "0"
    root.mkdir(parents=True)
    raw = raw_instance(2000)
    df = raw.assign(**{"class": pd.array([None] * 400 + [0] * 1600, dtype="Int16")})
    df.to_parquet(root / "WELL-00001_20170101000000.parquet")
    index = build_index(tmp_path / "dataset")
    clean_dir = tmp_path / "clean"
    (clean_dir / "0").mkdir(parents=True)
    clean_instance(load_instance(root / "WELL-00001_20170101000000.parquet"), PRE).to_parquet(
        clean_dir / "0" / "WELL-00001_20170101000000.parquet"
    )
    build_split(
        index["instance_id"].tolist(),
        index,
        clean_dir,
        p.scaler,
        WindowConfig(window_s=W, stride_s=STRIDE),
        tmp_path / "w",
        baseline_s=BASE_S,
    )
    store = WindowStore(tmp_path / "w")
    i = len(store) - 1
    end = int(store.windows["end_pos"].iloc[i])
    train_feats = p.xgb_features(store.batch(np.array([i])), store.baselines(np.array([i])))
    serving = p.predict_window(raw.iloc[: end + 1], keep_inputs=True)
    assert serving.inputs is not None
    x, base, blocks = serving.inputs
    x16 = x.astype(np.float16).astype(np.float32)
    serve_feats = p.xgb_features(x16[None], base[None], blocks[None])
    np.testing.assert_allclose(serve_feats, train_feats, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def context_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_bundle(tmp_path_factory.mktemp("ctx_bundle"), context=True)


@pytest.mark.parametrize("chunk_seed", [0, 1])
def test_streaming_context_features_equal_training_features(
    context_bundle: Path, chunk_seed: int
) -> None:
    """Long-context features online (one block per minute, bounded buffer) must equal the
    training pipeline's (whole instance at once, owfd.features.build)."""
    from owfd.features.build import instance_features
    from owfd.features.context import MAX_BLOCKS

    p = Predictor.load(context_bundle)
    assert p.uses_context
    raw = raw_instance(n=W + 60 * (MAX_BLOCKS + 40), seed=4)
    scaled = p.scaler.transform(clean_instance(raw, PRE))
    ends = np.arange(W - 1, len(raw), STRIDE)
    expected = instance_features(scaled, ends, VARS, W, BASE_S)
    cols = [p._all_features.index(f) for f in p.manifest["members"][0]["features"]]

    stream = StreamingPredictor(p)
    rng = np.random.default_rng(chunk_seed)
    got, lo = [], 0
    while lo < len(raw):
        n = int(rng.integers(1, 900))
        got += stream.update("w", raw.iloc[lo : lo + n], keep_inputs=True)
        lo += n
    assert [g.timestamp for g in got] == list(raw.index[ends])
    feats = np.concatenate([p.xgb_features(*(a[None] for a in g.inputs)) for g in got])
    np.testing.assert_allclose(feats, expected[:, cols], rtol=1e-5, atol=1e-5, equal_nan=True)
    assert not np.isnan(feats[-1]).all() and len(stream.wells["w"].blocks) == MAX_BLOCKS
    # Same probabilities as scoring the training features directly.
    single = p.predict_window(raw, well_id="w")
    np.testing.assert_allclose(single.probabilities, got[-1].probabilities, atol=1e-5)
