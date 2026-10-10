import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.config import SplitConfig, get_settings
from owfd.data.split import (
    SPLITS,
    classes_without_synthetic,
    load_splits,
    make_splits,
    save_splits,
    validate_splits,
)


def fake_index() -> pd.DataFrame:
    """12 real wells x a few classes, plus synthetic instances of classes 1 and 2."""
    rows = []
    rng = np.random.default_rng(0)
    for w in range(12):
        for c in (0, 4, 1 + w % 2):
            for k in range(rng.integers(1, 4)):
                rows.append((f"{c}/WELL-{w:05d}_{k}", c, "real", f"WELL-{w:05d}"))
    for k in range(5):
        rows.append((f"1/SIMULATED_{k:05d}", 1, "simulated", None))
        rows.append((f"2/DRAWN_{k:05d}", 2, "drawn", None))
    return pd.DataFrame(rows, columns=["instance_id", "event_class", "source", "well_id"])


CFG = SplitConfig(val_frac=0.2, test_frac=0.25, n_trials=500)


def test_splits_have_no_leakage_and_test_is_real_only() -> None:
    index = fake_index()
    splits = make_splits(index, CFG, seed=1)
    validate_splits(splits, index)
    by_id = index.set_index("instance_id")
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not set(splits["instances"][a]) & set(splits["instances"][b])
                wells_a = set(by_id.loc[splits["instances"][a], "well_id"].dropna())
                wells_b = set(by_id.loc[splits["instances"][b], "well_id"].dropna())
                assert not wells_a & wells_b
    for s in ("val", "test"):
        assert (by_id.loc[splits["instances"][s], "source"] == "real").all()
    synthetic = index.loc[index["source"] != "real", "instance_id"]
    assert set(synthetic) <= set(splits["instances"]["train"])


def test_real_only_classes_keep_a_training_well() -> None:
    index = fake_index()
    assert classes_without_synthetic(index) == {0, 4}
    splits = make_splits(index, CFG, seed=3)
    train = index[index["instance_id"].isin(splits["instances"]["train"])]
    assert {0, 4} <= set(train.loc[train["source"] == "real", "event_class"])


def test_splits_are_deterministic(tmp_path: Path) -> None:
    index = fake_index()
    a, b = make_splits(index, CFG, seed=7), make_splits(index, CFG, seed=7)
    assert a == b
    save_splits(a, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == a


@pytest.mark.parametrize("corrupt", ["shared_instance", "shared_well", "synthetic_in_test"])
def test_validate_splits_catches_leakage(corrupt: str) -> None:
    index = fake_index()
    splits = copy.deepcopy(make_splits(index, CFG, seed=1))
    if corrupt == "shared_instance":
        splits["instances"]["test"].append(splits["instances"]["train"][0])
    elif corrupt == "shared_well":
        splits["wells"]["test"].append(splits["wells"]["train"][0])
    else:
        sim = "1/SIMULATED_00000"
        splits["instances"]["train"].remove(sim)
        splits["instances"]["test"].append(sim)
    with pytest.raises(ValueError):
        validate_splits(splits, index)


def test_committed_split_manifest_is_valid() -> None:
    """The real manifest (configs/splits.json) has no leakage, if the dataset index exists."""
    settings = get_settings()
    index_path = settings.resolve(settings.processed_data_dir) / "instances.parquet"
    if not index_path.exists():
        pytest.skip("dataset not downloaded")
    index = pd.read_parquet(index_path)
    splits = load_splits(settings.resolve(settings.splits_file))
    validate_splits(splits, index)
    test = index[index["instance_id"].isin(splits["instances"]["test"])]
    assert set(test["event_class"]) == set(range(10))  # every class is tested
