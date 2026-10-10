from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.config import PreprocessingConfig, get_settings
from owfd.data.clean import Scaler, clean_instance, mask_column, select_variables, stuck_mask
from owfd.data.load import build_index
from owfd.data.preprocess import preprocess_all
from tests.conftest import make_instance

CFG = PreprocessingConfig(
    variables=["P-PDG", "T-TPT", "QGL", "P-JUS-BS"], ffill_limit_s=2, stuck_after_s=4,
    clip_quantiles=(0.0, 1.0), scaler_sample_every_s=1,
)  # fmt: skip


def series(values: list[float | None]) -> pd.Series:
    return pd.Series(
        values, dtype="float64", index=pd.date_range("2020", periods=len(values), freq="s")
    )


def test_default_config_matches_eda_decisions() -> None:
    cfg = get_settings().preprocessing
    assert cfg.variables == [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "QGL",
    ]
    assert cfg.stuck_after_s == 1800


def test_select_variables_adds_absent_columns_and_keeps_label() -> None:
    df = make_instance([0, 0, 1]).drop(columns=["QGL"])
    out = select_variables(df, ["P-PDG", "QGL"])
    assert list(out.columns) == ["P-PDG", "QGL", "class"]
    assert out["QGL"].isna().all()


def test_stuck_mask_is_causal() -> None:
    s = series([1, 2, 5, 5, 5, 5, 5, 3, None, None, None, None])
    # run of five 5s starts at position 2; flagged from its 4th sample on (stuck_after_s=4)
    expected = [False] * 5 + [True, True] + [False] * 5
    assert stuck_mask(s, 4).tolist() == expected
    # the result at t must not change when future samples are appended
    assert stuck_mask(s.iloc[:5], 4).tolist() == expected[:5]


def test_clean_instance_ffill_limit_masks_and_dtypes() -> None:
    df = make_instance([0] * 8)
    df["T-TPT"] = [1.0, np.nan, np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]  # 3-sample gap, limit 2
    df["P-PDG"] = 7.0  # stuck from the 4th sample
    out = clean_instance(df, CFG)
    assert out["T-TPT"].isna().tolist() == [False, False, False, True, False, False, False, False]
    assert out[mask_column("T-TPT")].tolist() == out["T-TPT"].isna().tolist()
    # stuck values are masked, then forward-fill (limit 2) bridges 2 of them
    assert out[mask_column("P-PDG")].tolist() == [False] * 5 + [True] * 3
    assert out[mask_column("P-JUS-BS")].all()  # absent everywhere
    assert out["P-PDG"].dtype == np.float32 and out[mask_column("QGL")].dtype == bool
    assert out["class"].dtype == "Int16"
    sensor_nan = out[CFG.variables].isna().to_numpy()
    mask = out[[mask_column(v) for v in CFG.variables]].to_numpy()
    assert (sensor_nan == mask).all()  # NaN only where masked


def test_scaler_fit_transform_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(3):
        df = make_instance([0] * 200)
        df["P-PDG"] = rng.normal(100.0, 10.0, 200)
        df["T-TPT"] = 50.0  # constant -> IQR 0 and range 0 -> scale falls back to 1
        frames.append(clean_instance(df, CFG.model_copy(update={"stuck_after_s": 10_000})))
    scaler = Scaler.fit(frames, CFG)
    assert scaler.scale[CFG.variables.index("T-TPT")] == 1.0
    assert scaler.median[CFG.variables.index("P-JUS-BS")] == 0.0  # never observed

    test = frames[0].copy()
    test.loc[test.index[0], "P-PDG"] = 1e9  # outlier -> clipped to the training max
    test.loc[test.index[1], "P-PDG"] = np.nan
    out = scaler.transform(test)
    assert not out[CFG.variables].isna().any().any()
    assert out["P-PDG"].iloc[0] == pytest.approx(
        (scaler.upper[0] - scaler.median[0]) / scaler.scale[0]
    )
    assert out["P-PDG"].iloc[1] == 0.0
    assert out[mask_column("P-PDG")].equals(test[mask_column("P-PDG")])

    scaler.save(tmp_path / "scaler.json")
    assert Scaler.load(tmp_path / "scaler.json") == scaler


def test_scaler_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Scaler.fit([], CFG)


def test_preprocess_all_writes_and_skips(tiny_dataset: Path, tmp_path: Path) -> None:
    index = build_index(tiny_dataset)
    dst = tmp_path / "clean"
    assert preprocess_all(index, tiny_dataset, dst, CFG, workers=2) == 4
    assert preprocess_all(index, tiny_dataset, dst, CFG, workers=2) == 0  # already done
    out = pd.read_parquet(dst / index.loc[0, "path"])
    assert set(CFG.variables) <= set(out.columns)
