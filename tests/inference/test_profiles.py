from pathlib import Path

import numpy as np
import pytest

from owfd.features.context import all_feature_names
from owfd.inference.profiles import WellProfiles
from tests.bundle import VARS, add_profile, build_bundle


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return add_profile(build_bundle(tmp_path_factory.mktemp("b"), context=True), "w1")


def test_profiles_score_and_explain(bundle: Path) -> None:
    import json

    profiles = WellProfiles.from_manifest(
        bundle, json.loads((bundle / "manifest.json").read_text())
    )
    assert profiles and profiles.has("w1") and not profiles.has("w2")
    names = all_feature_names(VARS)
    x = np.zeros((3, len(names)))
    x[:, 0] = [-2.0, 0.0, 2.0]  # the random profile keys on the first feature
    p = profiles.score("w1", x)
    assert p.shape == (3,) and p[2] > p[0]
    d = profiles.drivers("w1", x[2])
    assert len(d) == 5 and d[0]["feature"] == names[0]
    assert [r["contribution"] for r in d] == sorted((r["contribution"] for r in d), reverse=True)
    assert profiles.info("w1")["source_well"] == "WELL-X"


def test_bundle_without_profiles(tmp_path: Path) -> None:
    profiles = WellProfiles.from_manifest(tmp_path, {})
    assert not profiles and not profiles.has("w1") and profiles.info("w1") is None
