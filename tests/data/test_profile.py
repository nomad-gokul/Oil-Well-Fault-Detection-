from pathlib import Path

import numpy as np
import pandas as pd

from owfd.data.load import build_index
from owfd.data.profile import profile_dataset, profile_instance
from tests.conftest import make_instance


def test_profile_phases_and_labels() -> None:
    df = make_instance([None] * 2 + [0] * 3 + [101] * 4 + [1] * 5)
    p = profile_instance(df)
    assert (p["n_unlabelled"], p["n_normal"], p["n_transient"], p["n_steady"]) == (2, 3, 4, 5)
    assert p["transient_start_s"] == 5 and p["steady_start_s"] == 9
    assert p["phases_ordered"] is True
    assert p["duration_s"] == 13 and p["n_gaps"] == 0


def test_profile_detects_out_of_order_phases() -> None:
    assert profile_instance(make_instance([0, 1, 1, 0]))["phases_ordered"] is False


def test_profile_missing_frozen_and_gaps() -> None:
    df = make_instance([0] * 6)
    df["P-PDG"] = 5.0  # frozen sensor
    df.loc[df.index[:3], "T-TPT"] = np.nan  # half missing
    df = df.drop(index=df.index[2])  # one 2-second gap
    p = profile_instance(df)
    assert p["frozen__P-PDG"] is True and p["frozen__T-TPT"] is False
    assert p["missing__P-JUS-BS"] == 1.0 and p["frozen__P-JUS-BS"] is False
    assert p["missing__T-TPT"] == 2 / 5
    assert p["n_gaps"] == 1 and p["max_gap_s"] == 2.0


def test_profile_dataset_joins_index(tiny_dataset: Path) -> None:
    index = build_index(tiny_dataset)
    profile = profile_dataset(index, tiny_dataset, workers=2)
    assert len(profile) == len(index)
    assert profile["instance_id"].tolist() == index["instance_id"].tolist()
    drawn = profile.set_index("source").loc["drawn"]
    assert drawn["n_transient"] == 6 and pd.isna(drawn["steady_start_s"])
