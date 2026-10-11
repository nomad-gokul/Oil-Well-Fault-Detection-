import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from owfd.config import PROJECT_ROOT
from owfd.simulator.replay import (
    Replay,
    Scenario,
    demo_catalog,
    describe_source,
    load_demo,
    read_scenario,
    run,
    write_scenario,
)
from tests.bundle import raw_instance


def scenario(name: str, n: int, cls: int = 0) -> Scenario:
    df = raw_instance(n)[["P-PDG", "P-TPT"]]
    return Scenario(name, cls, df, pd.Series(0, index=df.index))


def make_demo_dir(root: Path, sizes: dict[str, int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (name, n) in enumerate(sizes.items()):
        df = raw_instance(n, seed=i)[["P-PDG", "P-TPT", "T-TPT"]].assign(**{"class": 0})
        write_scenario(root / f"{name}.npz", df.drop(columns="class"), df["class"])
        items.append({"well_id": name, "event_class": i, "file": f"{name}.npz", "source": "x"})
    (root / "manifest.json").write_text(json.dumps({"scenarios": items}))
    return root


def test_replay_shifts_time_and_sends_everything_once() -> None:
    start = pd.Timestamp("2030-01-01")
    replay = Replay([scenario("a", 250), scenario("b", 100)], speed=60, start=start)
    got: dict[str, list[pd.DataFrame]] = {"a": [], "b": []}

    async def sink(well: str, chunk: pd.DataFrame) -> None:
        got[well].append(chunk)

    ticks = asyncio.run(run(replay, sink, tick_s=0))
    assert ticks == 5 and replay.done()  # ceil(250 / 60)
    a = pd.concat(got["a"])
    assert len(a) == 250 and a.index[0] == start and a.index.is_unique
    assert (a.index.to_series().diff().dropna() == pd.Timedelta(seconds=1)).all()
    assert len(pd.concat(got["b"])) == 100


def test_loop_keeps_time_moving_forward() -> None:
    replay = Replay([scenario("a", 90)], speed=60, loop=True, start=pd.Timestamp("2030-01-01"))
    chunks = [c for _ in range(4) for _, c in replay.tick()]
    times = pd.concat(chunks).index
    assert times.is_monotonic_increasing and times.is_unique and not replay.done()
    assert len(times) == 60 + 30 + 60 + 30


def test_bundled_demo_set() -> None:
    demo = PROJECT_ROOT / "demo_data"
    if not (demo / "manifest.json").exists():
        pytest.skip("demo set not built")
    scenarios = load_demo(demo)
    assert sorted(s.event_class for s in scenarios) == list(range(10))
    assert sum(p.stat().st_size for p in demo.glob("*.npz")) < 20e6
    assert [s.well_id for s in load_demo(demo, classes=[7, 8])] == [
        "demo-7-pck-scaling", "demo-8-hydrate-prod"]  # fmt: skip
    for s in scenarios:
        assert "class" not in s.data.columns  # labels are never sent to the model
        assert s.data.index.is_monotonic_increasing


def test_scenario_roundtrip_is_exact(tmp_path: Path) -> None:
    df = raw_instance(50)[["P-PDG", "QGL"]]
    df.iloc[3, 0] = 1.180116e42  # real 3W placeholder value; must survive unchanged
    labels = pd.Series(pd.array([None] * 10 + [0] * 30 + [107] * 10, dtype="Int16"), index=df.index)
    write_scenario(tmp_path / "s.npz", df, labels)
    df2, labels2 = read_scenario(tmp_path / "s.npz")
    assert (df2.index == df.index).all()  # same instants, whatever the stored unit
    pd.testing.assert_frame_equal(
        df2, df, check_freq=False, check_names=False, check_index_type=False
    )
    assert labels2.isna().sum() == 10 and labels2.iloc[-1] == 107


def test_describe_source() -> None:
    assert describe_source("4/WELL-00004_20140806010107") == {
        "source_well": "WELL-00004", "recorded_at": "2014-08-06T01:01:07"}  # fmt: skip
    assert describe_source("x") == {"source_well": None, "recorded_at": None}
    assert describe_source("3/SIMULATED_00001") == {"source_well": None, "recorded_at": None}


def test_demo_catalog(tmp_path: Path) -> None:
    assert demo_catalog(tmp_path) == {}  # no manifest
    catalog = demo_catalog(make_demo_dir(tmp_path / "d", {"demo-a": 10, "demo-b": 10}))
    assert catalog["demo-b"]["event_class"] == 1 and catalog["demo-b"]["source_well"] is None
    bundled = PROJECT_ROOT / "demo_data"
    if (bundled / "manifest.json").exists():
        info = demo_catalog(bundled)["demo-4-instability"]
        assert info["class_name"] == "Flow Instability" and info["source_well"] == "WELL-00004"
