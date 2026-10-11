"""Per-well flow-instability profiles at serving time (trained by ``owfd.models.well_profiles``).

A bundle may contain ``profiles/`` with one small XGBoost model per well and a ``chronic``
section in its manifest. For a well with a profile, every prediction's feature vector is also
scored for P(unstable); the API turns that into "chronic instability" alerts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from owfd.inference.predictor import load_booster


class WellProfiles:
    """Profiles of a serving bundle (empty if it has none)."""

    def __init__(self, bundle: Path, section: dict[str, Any] | None) -> None:
        self.section = section or {}
        self.features: list[str] = list(self.section.get("features", []))
        self.models: dict[str, Any] = {}
        for well_id, meta in self.section.get("profiles", {}).items():
            self.models[well_id] = load_booster(bundle / "profiles" / meta["file"])

    @classmethod
    def from_manifest(cls, bundle: Path, manifest: dict[str, Any]) -> WellProfiles:
        """Profiles named in ``manifest['chronic']`` (none for older bundles)."""
        return cls(bundle, manifest.get("chronic"))

    def __bool__(self) -> bool:
        return bool(self.models)

    def has(self, well_id: str) -> bool:
        """Whether this well has a profile."""
        return well_id in self.models

    def info(self, well_id: str) -> dict[str, Any] | None:
        """Training metadata of the well's profile."""
        meta = self.section.get("profiles", {}).get(well_id)
        return dict(meta) if meta else None

    def _matrix(self, features: np.ndarray) -> Any:
        import xgboost as xgb

        return xgb.DMatrix(np.atleast_2d(features), feature_names=self.features)

    def score(self, well_id: str, features: np.ndarray) -> np.ndarray:
        """P(unstable) for feature vectors [n, n_features] (order: ``self.features``)."""
        return np.asarray(self.models[well_id].predict(self._matrix(features)), dtype=float)

    def drivers(self, well_id: str, features: np.ndarray, top: int = 5) -> list[dict[str, Any]]:
        """Largest contributions (log-odds, SHAP) towards "unstable" for one feature vector."""
        contrib = self.models[well_id].predict(self._matrix(features), pred_contribs=True)[0]
        values = np.atleast_1d(np.asarray(features, dtype=float))
        order = np.argsort(-contrib[:-1])[:top]  # last column is the bias
        return [
            {
                "feature": self.features[i],
                "value": float(values[i]),
                "contribution": float(contrib[i]),
            }
            for i in order
        ]
