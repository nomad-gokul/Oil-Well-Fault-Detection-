"""Build the small bundled demo set (``demo_data/``) from held-out *test* instances.

Usage::

    python -m owfd.simulator.make_demo

For each event class, one real test instance (preferring one with a transient, so the alert
can come early) is cut to at most ``max_hours`` around the event: from the instance start
(which includes the unlabelled first hour, used as the well's baseline) to a margin after the
fault reaches its steady state. Only the 7 model sensors and the label are kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from owfd.config import PROJECT_ROOT, get_settings
from owfd.data.load import load_instance
from owfd.data.split import load_splits
from owfd.simulator.replay import DEMO_DIR, SHORT, write_scenario


def pick_instances(profile: pd.DataFrame, test_ids: set[str], max_hours: float) -> pd.DataFrame:
    """One test instance per class: has a transient if possible, short, median-sized."""
    test = profile[profile["instance_id"].isin(test_ids)].copy()
    test["has_transient"] = test["n_transient"] > 0
    test["fits"] = test["duration_s"] <= max_hours * 3600 * 4
    picks = []
    for _, g in test.groupby("event_class"):
        g = g.sort_values(["has_transient", "fits", "duration_s"], ascending=[False, False, True])
        best = g[g["has_transient"] == g["has_transient"].iloc[0]]
        picks.append(best.iloc[len(best) // 4] if len(best) > 3 else best.iloc[0])
    return pd.DataFrame(picks)


def cut(df: pd.DataFrame, max_hours: float, after_steady_s: int = 3600) -> pd.DataFrame:
    """Keep from the start to ``after_steady_s`` after the steady state (or transient) starts."""
    lab = df["class"].fillna(-1).to_numpy()
    fault = np.flatnonzero(lab > 0)
    steady = np.flatnonzero((lab > 0) & (lab < 100))
    anchor = steady[0] if steady.size else (fault[0] if fault.size else len(df) // 2)
    end = min(len(df), anchor + after_steady_s, int(max_hours * 3600))
    start = max(0, end - int(max_hours * 3600))
    return df.iloc[start:end]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build demo_data/ from the test split.")
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=DEMO_DIR)
    args = parser.parse_args(argv)
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    profile = pd.read_parquet(processed / "profile.parquet")
    test_ids = set(load_splits(settings.resolve(settings.splits_file))["instances"]["test"])
    picks = pick_instances(profile, test_ids, args.max_hours)
    variables = settings.preprocessing.variables
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for _, row in picks.iterrows():
        df = load_instance(settings.resolve(settings.dataset_dir) / row["path"])
        part = cut(df, args.max_hours)[[*variables, "class"]]
        # float64 on purpose: some wells report placeholder values (e.g. 1.2e42 Pa) that
        # overflow float32; the model's clipping handles them, so they are kept as-is.
        c = int(row["event_class"])
        name = f"{c}-{SHORT[c]}"
        write_scenario(args.out / f"{name}.npz", part[variables], part["class"])
        scenarios.append(
            {
                "well_id": f"demo-{name}",
                "event_class": c,
                "file": f"{name}.npz",
                "source": row["instance_id"],
                "rows": len(part),
            }
        )
    readme = (
        "Demo scenarios cut from held-out TEST instances of the Petrobras 3W dataset "
        "(CC BY 4.0, https://github.com/petrobras/3W). Built by "
        "`python -m owfd.simulator.make_demo`; see that module for the selection rule."
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "description": readme,
                "split": "test",
                "max_hours": args.max_hours,
                "scenarios": scenarios,
            },
            indent=1,
        )
    )
    for old in args.out.glob("*.parquet"):  # earlier format
        old.unlink()
    total = sum(p.stat().st_size for p in args.out.glob("*.npz"))
    print(
        f"{len(scenarios)} scenarios, {total / 1e6:.1f} MB -> {args.out.relative_to(PROJECT_ROOT)}"
    )
    for s in scenarios:
        print(f"  {s['well_id']:28s} {s['rows'] / 3600:5.1f} h  from {s['source']}")


if __name__ == "__main__":
    main()
