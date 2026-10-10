from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.config import PreprocessingConfig, WindowConfig
from owfd.data.clean import Scaler, clean_instance
from owfd.data.load import build_index, load_instance
from owfd.data.windows import (
    WindowStore,
    base_column,
    build_split,
    causal_baseline,
    channels,
    map_labels,
    window_ends,
)

PRE = PreprocessingConfig(
    variables=["P-PDG", "T-TPT", "P-JUS-BS"], ffill_limit_s=2, stuck_after_s=10_000,
    clip_quantiles=(0.0, 1.0), scaler_sample_every_s=1,
)  # fmt: skip
WIN = WindowConfig(window_s=4, stride_s=2, max_train_windows_per_instance=2)


def test_map_labels() -> None:
    event, transient = map_labels(np.array([0, 1, 9, 101, 109]))
    assert event.tolist() == [0, 1, 9, 1, 9]
    assert transient.tolist() == [False, False, False, True, True]


def test_window_ends_are_end_aligned_labelled_and_full_length() -> None:
    labels = pd.Series([None, None, None, 0, 0, 0, 101, 101, 1, 1], dtype="Int16")
    ends = window_ends(labels, WIN)
    assert ends.tolist() == [3, 5, 7, 9]  # aligned to the last row, every 2 rows
    labels[3] = pd.NA
    assert window_ends(labels, WIN).tolist() == [5, 7, 9]  # unlabelled ends are skipped
    assert window_ends(labels.iloc[:3], WIN).size == 0  # shorter than one window


def test_window_ends_cap() -> None:
    labels = pd.Series([0] * 40, dtype="Int16")
    rng = np.random.default_rng(0)
    capped = window_ends(labels, WIN, cap=5, rng=rng)
    assert len(capped) == 5 and (np.diff(capped) > 0).all()
    with pytest.raises(ValueError):
        window_ends(labels, WIN, cap=5)


def test_build_split_and_store_match_scaled_data(tiny_dataset: Path, tmp_path: Path) -> None:
    index = build_index(tiny_dataset)
    clean_dir = tmp_path / "clean"
    for rel, iid in zip(index["path"], index["instance_id"], strict=True):
        (clean_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        clean_instance(load_instance(tiny_dataset / rel), PRE).to_parquet(
            clean_dir / f"{iid}.parquet"
        )
    ids = index["instance_id"].tolist()
    scaler = Scaler.fit([pd.read_parquet(clean_dir / f"{i}.parquet") for i in ids], PRE)
    table = build_split(ids, index, clean_dir, scaler, WIN, tmp_path / "out")
    store = WindowStore(tmp_path / "out")
    assert len(store) == len(table) > 0
    assert store.channels == channels(PRE.variables)

    # Every stored window equals the matching slice of the scaled instance.
    for i, row in table.iterrows():
        df = scaler.transform(pd.read_parquet(clean_dir / f"{row['instance_id']}.parquet"))
        expected = df[store.channels].to_numpy(np.float16)[row["end_pos"] - 3 : row["end_pos"] + 1]
        np.testing.assert_array_equal(store.x(int(i)), expected.astype(np.float32))
        assert row["label_raw"] == df["class"].iloc[row["end_pos"]]
    np.testing.assert_array_equal(store.batch(np.array([0, 1]))[1], store.x(1))

    real1 = table[table["instance_id"] == "1/WELL-00002_20170102000000"]
    # labels 5x0, 5x101, 5x1; window 4, stride 2 -> ends at rows 4, 6, 8, 10, 12, 14
    assert real1["end_pos"].tolist() == [4, 6, 8, 10, 12, 14]
    assert real1["label_raw"].tolist() == [0, 101, 101, 1, 1, 1]
    assert real1["is_transient"].tolist() == [False, True, True, False, False, False]
    assert real1["is_fault"].tolist() == [False, True, True, True, True, True]
    # instance 0 has 5 unlabelled trailing rows: windows ending there are skipped
    assert (table.loc[table["instance_id"].str.startswith("0/"), "label_raw"] == 0).all()


def test_build_split_cap(tiny_dataset: Path, tmp_path: Path) -> None:
    index = build_index(tiny_dataset)
    clean_dir = tmp_path / "clean"
    for rel, iid in zip(index["path"], index["instance_id"], strict=True):
        (clean_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        clean_instance(load_instance(tiny_dataset / rel), PRE).to_parquet(
            clean_dir / f"{iid}.parquet"
        )
    ids = index["instance_id"].tolist()
    scaler = Scaler.fit([pd.read_parquet(clean_dir / f"{i}.parquet") for i in ids], PRE)
    table = build_split(ids, index, clean_dir, scaler, WIN, tmp_path / "o", cap=2, seed=0)
    assert table.groupby("instance_id").size().max() <= 2


def test_causal_baseline() -> None:
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [100.0, 5.0], [100.0, 5.0]])
    masks = np.array([[0, 1], [0, 1], [0, 0], [0, 0], [0, 0]])
    base = causal_baseline(values, masks, np.array([0, 1, 2, 4]), baseline_s=3)
    np.testing.assert_allclose(base[:, 0], [1.0, 2.0, 3.0, 3.0])  # frozen after 3 rows
    assert np.isnan(base[0, 1]) and np.isnan(base[1, 1])  # nothing observed yet
    np.testing.assert_allclose(base[2:, 1], [5.0, 5.0])
    # causal: appending future rows does not change earlier baselines
    short = causal_baseline(values[:2], masks[:2], np.array([0, 1]), baseline_s=3)
    np.testing.assert_allclose(short[:, 0], base[:2, 0])


def test_store_exposes_baselines(tiny_dataset: Path, tmp_path: Path) -> None:
    index = build_index(tiny_dataset)
    clean_dir = tmp_path / "clean"
    for rel, iid in zip(index["path"], index["instance_id"], strict=True):
        (clean_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        clean_instance(load_instance(tiny_dataset / rel), PRE).to_parquet(
            clean_dir / f"{iid}.parquet"
        )
    ids = index["instance_id"].tolist()
    scaler = Scaler.fit([pd.read_parquet(clean_dir / f"{i}.parquet") for i in ids], PRE)
    table = build_split(ids, index, clean_dir, scaler, WIN, tmp_path / "o", baseline_s=5)
    store = WindowStore(tmp_path / "o")
    row = table.iloc[2]
    df = scaler.transform(pd.read_parquet(clean_dir / f"{row['instance_id']}.parquet"))
    expected = df["P-PDG"].iloc[: min(row["end_pos"] + 1, 5)].mean()
    assert store.baselines(np.array([2]))[0, 0] == pytest.approx(expected, rel=1e-5)
    assert np.isnan(table[base_column("P-JUS-BS")]).all()  # never observed
