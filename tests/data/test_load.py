from pathlib import Path

import pandas as pd
import pytest

from owfd.data.load import (
    INDEX_COLUMNS,
    build_index,
    load_instance,
    parse_instance_name,
    summarize_index,
)
from owfd.data.schema import Source, base_class, is_transient


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("WELL-00001_20170201010207.parquet", (Source.REAL, "WELL-00001")),
        ("SIMULATED_00042.parquet", (Source.SIMULATED, None)),
        ("DRAWN_00003.parquet", (Source.DRAWN, None)),
    ],
)
def test_parse_instance_name(name: str, expected: tuple[Source, str | None]) -> None:
    assert parse_instance_name(name) == expected


@pytest.mark.parametrize("name", ["foo.parquet", "WELL-1_2017.parquet", "SIMULATED_00001.csv"])
def test_parse_instance_name_rejects_unknown(name: str) -> None:
    with pytest.raises(ValueError):
        parse_instance_name(name)


def test_build_index(tiny_dataset: Path) -> None:
    index = build_index(tiny_dataset)
    assert list(index.columns) == INDEX_COLUMNS
    assert len(index) == 4
    real = index[index["source"] == "real"].set_index("well_id")
    assert real.loc["WELL-00002", "event_class"] == 1
    assert real.loc["WELL-00002", "n_rows"] == 15
    assert real.loc["WELL-00001", "start_time"] == pd.Timestamp("2017-01-01")
    assert real.loc["WELL-00001", "end_time"] == pd.Timestamp("2017-01-01 00:00:14")
    assert index.loc[index["source"] != "real", "well_id"].isna().all()
    assert (tiny_dataset / index.loc[0, "path"]).exists()  # paths are relative to the root


def test_build_index_empty_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path)


def test_summarize_index(tiny_dataset: Path) -> None:
    summary = summarize_index(build_index(tiny_dataset))
    assert summary.loc[1, "real"] == 1
    assert summary.loc["total", "total"] == 4


def test_load_instance(tiny_dataset: Path) -> None:
    df = load_instance(tiny_dataset / "0/WELL-00001_20170101000000.parquet")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert df["class"].dtype == "Int16"
    assert df["class"].isna().sum() == 5


def test_label_helpers() -> None:
    assert base_class(105) == 5 and base_class(5) == 5 and base_class(0) == 0
    assert is_transient(101) and not is_transient(1)
