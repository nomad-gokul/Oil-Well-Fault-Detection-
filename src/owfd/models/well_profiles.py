"""Per-well flow-instability fingerprints ("profiles") learned from operator labels.

Across wells, chronic flow instability is close to undetectable: a detector trained on other
wells reaches a window AUC of only about 0.57, because every well is unstable in its own way.
Within one well the two regimes are almost perfectly separable. So once operators have
labelled some of a well's past recordings as unstable or normal, a small per-well model flags
the well while it is unstable (an alert of class 4, separate from the onset model's alerts).

Usage::

    python -m owfd.models.well_profiles evaluate [--label-frac 0.25]  # time-ordered replay
    python -m owfd.models.well_profiles build                          # profiles for serving

``evaluate`` replays every well's recordings in time order. Each recording is scored by a
profile trained only on that well's recordings that *ended before it started* (labels play the
role of operator feedback; ``--label-frac`` keeps a random share of them), so the first
episodes of every well, before any labels exist, count as misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from owfd.config import PROJECT_ROOT, get_settings
from owfd.features.context import all_feature_names
from owfd.models.evaluate import AlertRule, alert_state

logger = logging.getLogger(__name__)

UNSTABLE = 4  # 3W class "Flow Instability"
PARAMS: dict[str, Any] = {
    "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.5, "tree_method": "hist", "random_state": 0,
}  # fmt: skip
#: Minimum labelled windows of each kind (unstable, normal) before a well gets a profile.
MIN_WINDOWS = 60
#: A profile is refitted after this many newly labelled recordings.
RETRAIN_EVERY = 5
#: Chronic alert rule, fixed in advance (the evaluation shows the results barely depend on it).
RULE = AlertRule(threshold=0.8, k=20)
#: A profile whose newest training data is older than this is not used: the well's normal
#: operation may have changed (replay: same detection, fewer false alarms than no limit).
MAX_AGE_DAYS = 180.0
META = ["instance_id", "well_id", "end_time", "event_class"]


def fit_profile(x: pd.DataFrame, unstable: np.ndarray, n_jobs: int = 4) -> XGBClassifier | None:
    """Binary unstable-vs-normal model for one well; None without enough of both kinds."""
    y = np.asarray(unstable, dtype=bool)
    if y.sum() < MIN_WINDOWS or (~y).sum() < MIN_WINDOWS:
        return None
    model = XGBClassifier(**PARAMS, n_jobs=n_jobs)
    model.fit(x, y, sample_weight=np.where(y, (~y).sum() / y.sum(), 1.0))
    return model


def _is_labelled(instance_id: str, fraction: float, seed: int) -> bool:
    """Deterministic coin flip: does the operator label this recording?"""
    if fraction >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{instance_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def prequential(
    windows: pd.DataFrame,
    instances: pd.DataFrame,
    features: list[str],
    label_frac: float = 1.0,
    seed: int = 0,
    max_age_days: float | None = None,
) -> pd.DataFrame:
    """P(unstable) for every window of every well that ever has unstable labels, each recording
    scored by a profile trained only on that well's (labelled) recordings that ended before it
    started. ``instances`` is indexed by instance id with ``start_time`` and ``end_time``.

    With ``max_age_days``, a profile whose newest training data is older than that when a
    recording starts is not used (the well's normal may have changed); it is refitted as soon
    as any newer labels exist."""
    out = []
    for _, g in windows.groupby("well_id"):
        if not (g["event_class"] == UNSTABLE).any():
            continue
        order = instances.loc[g["instance_id"].unique()].sort_values("start_time").index.tolist()
        model, n_at_fit, fitted_until = None, -1, pd.NaT
        max_age = pd.Timedelta(days=max_age_days) if max_age_days is not None else None
        for iid in order:
            start = instances.at[iid, "start_time"]
            done = [
                o for o in order
                if instances.at[o, "end_time"] <= start and _is_labelled(o, label_frac, seed)
            ]  # fmt: skip
            stale = max_age is not None and model is not None and start - fitted_until > max_age
            if (
                model is None
                or len(done) - n_at_fit >= RETRAIN_EVERY
                or (stale and len(done) > n_at_fit)
            ):
                hist = g[g["instance_id"].isin(done) & g["event_class"].isin([0, UNSTABLE])]
                fitted = fit_profile(hist[features], hist["event_class"].to_numpy() == UNSTABLE)
                if fitted is not None:
                    model, n_at_fit = fitted, len(done)
                    fitted_until = instances.loc[done, "end_time"].max()
            active = model is not None and (max_age is None or start - fitted_until <= max_age)
            cur = g[g["instance_id"] == iid].sort_values("end_time")
            p = (
                model.predict_proba(cur[features])[:, 1]
                if active and model is not None
                else np.zeros(len(cur))
            )
            out.append(cur[META].assign(p_unstable=p, has_profile=active))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=META)


def chronic_report(preds: pd.DataFrame, rule: AlertRule = RULE) -> pd.DataFrame:
    """Per instance: unstable episode caught, false alarms on normal windows, normal hours."""
    rows = []
    for iid, c in preds.groupby("instance_id"):
        c = c.sort_values("end_time")
        state = alert_state(c["p_unstable"].to_numpy(), rule)
        onsets = state & ~np.r_[False, state[:-1]]
        unstable = (c["event_class"] == UNSTABLE).to_numpy()
        normal = (c["event_class"] == 0).to_numpy()
        rows.append({
            "instance_id": iid, "well_id": c["well_id"].iloc[0],
            "unstable": bool(unstable.any()), "caught": bool((state & unstable).any()),
            "had_profile": bool(c["has_profile"].iloc[0]),
            "false_alarms": int((onsets & normal).sum()), "normal_hours": float(normal.sum() / 60),
        })  # fmt: skip
    return pd.DataFrame(rows)


def summarize(report: pd.DataFrame, instances: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers, split into continuations (the well's previous recording was unstable
    too) and new onsets after normal operation."""
    inst = instances.sort_values(["well_id", "start_time"])
    previous = inst.groupby("well_id")["event_class"].shift()
    continuation = (previous == UNSTABLE).reindex(report["instance_id"]).to_numpy()
    u = report["unstable"].to_numpy()
    caught = report["caught"].to_numpy()
    hours = report["normal_hours"].sum()

    def share(mask: np.ndarray) -> list[int]:
        return [int((caught & mask).sum()), int(mask.sum())]

    return {
        "unstable_episodes": share(u),
        "after_first_profile": share(u & report["had_profile"].to_numpy()),
        "continuations": share(u & continuation),
        "new_onsets": share(u & ~continuation),
        "false_alarms_per_day": float(report["false_alarms"].sum() / hours * 24) if hours else 0.0,
        "false_alarms": int(report["false_alarms"].sum()),
        "normal_hours": float(hours),
    }


def _load(settings: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    processed = settings.resolve(settings.processed_data_dir)
    features = all_feature_names(settings.preprocessing.variables)
    windows = pd.concat(
        [pd.read_parquet(processed / "features" / f"{s}.parquet", columns=META + features)
         for s in ("train_full", "val", "test")], ignore_index=True,
    )  # fmt: skip
    instances = pd.read_parquet(processed / "instances.parquet").set_index("instance_id")
    return windows, instances, features


def _demo_sources(demo_dir: Path, instances: pd.DataFrame) -> dict[str, tuple[str, pd.Timestamp]]:
    """Demo well id -> (source well, start of the demo recording)."""
    out = {}
    for f in sorted(demo_dir.glob("*.npz")):
        start = pd.Timestamp(int(np.load(f)["timestamps"][0]))
        match = instances[instances["start_time"] == start]
        if len(match):
            out[f"demo-{f.stem}"] = (str(match["well_id"].iloc[0]), start)
    return out


def build(settings: Any, out_dir: Path) -> dict[str, Any]:
    """Profiles from all labelled history of every well, plus one per demo well trained only on
    its source well's recordings that ended before the demo recording starts. A demo profile
    older than ``MAX_AGE_DAYS`` at the demo recording's start is not built: the API would not
    use it (demo timestamps are shifted to the present, so its age is checked here)."""
    windows, instances, features = _load(settings)
    labelled = windows[windows["event_class"].isin([0, UNSTABLE])]
    targets: dict[str, tuple[str, pd.Timestamp | None]] = {
        str(w): (str(w), None) for w in labelled["well_id"].unique()
    }
    targets.update(_demo_sources(settings.resolve(settings.demo_dir), instances))
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for profile_id, (well, cutoff) in sorted(targets.items()):
        g = labelled[labelled["well_id"] == well]
        if cutoff is not None:
            g = g[g["instance_id"].map(instances["end_time"]) <= cutoff]
        if len(g) and cutoff is not None:
            age = cutoff - g["end_time"].max()
            if age > pd.Timedelta(days=MAX_AGE_DAYS):
                logger.info("%s: newest labels are %s old at the demo start; no profile",
                            profile_id, age)  # fmt: skip
                continue
        model = fit_profile(g[features], g["event_class"].to_numpy() == UNSTABLE)
        if model is None:
            continue
        model.save_model(out_dir / f"{profile_id}.json")
        profiles[profile_id] = {
            "file": f"{profile_id}.json", "source_well": well,
            "trained_until": str(g["end_time"].max()),
            "unstable_windows": int((g["event_class"] == UNSTABLE).sum()),
            "normal_windows": int((g["event_class"] == 0).sum()),
            "age_checked_at_build": cutoff is not None,
        }  # fmt: skip
    section = {
        "threshold": RULE.threshold, "k": RULE.k, "class": UNSTABLE,
        "max_age_days": MAX_AGE_DAYS, "features": features, "profiles": profiles,
    }  # fmt: skip
    (out_dir / "profiles.json").write_text(json.dumps(section, indent=1))
    return section


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Per-well flow-instability profiles.")
    parser.add_argument("command", choices=("evaluate", "build"))
    parser.add_argument("--label-frac", type=float, default=1.0, help="share of labelled past")
    parser.add_argument("--max-age-days", type=float, help="ignore profiles older than this")
    parser.add_argument("--no-max-age", action="store_true", help="evaluate without an age limit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    root = settings.resolve(settings.artifacts_dir) / "well_profiles"
    if args.command == "build":
        version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        section = build(settings, root / version)
        logger.info("%d profiles -> %s", len(section["profiles"]), root / version)
        return
    windows, instances, features = _load(settings)
    if args.max_age_days is None and not args.no_max_age:
        args.max_age_days = MAX_AGE_DAYS
    preds = prequential(
        windows, instances, features, args.label_frac, settings.random_seed, args.max_age_days
    )
    report = chronic_report(preds)
    result = {"label_frac": args.label_frac, "max_age_days": args.max_age_days,
              "rule": RULE.__dict__, "summary": summarize(report, instances)}  # fmt: skip
    tag = f"{args.label_frac:g}" + (f"_age{args.max_age_days:g}" if args.max_age_days else "")
    root.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(root / f"prequential_{tag}.parquet")
    report.to_parquet(root / f"report_{tag}.parquet")
    (root / f"evaluation_{tag}.json").write_text(json.dumps(result, indent=1))
    logger.info("%s", json.dumps(result, indent=1))
    logger.info("-> %s", (root / f"evaluation_{tag}.json").relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
