"""Early-warning evaluation: replay held-out instances in time order and score alerts.

An alert fires when P(fault) = 1 - P(normal) exceeds ``threshold`` for ``k`` consecutive
windows (one per minute). Per instance we measure whether and when the fault is caught;
on normal operation we count false alarms per 24 h.

Usage::

    python -m owfd.models.evaluate   # compares XGBoost, the deep model(s) and an ensemble
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Columns the evaluation needs from a per-window prediction table.
PRED_COLUMNS = ["instance_id", "end_time", "event_class", "is_transient", "p_fault"]


@dataclass(frozen=True)
class AlertRule:
    """Alert when P(fault) > ``threshold`` for ``k`` consecutive windows."""

    threshold: float
    k: int


def alert_state(p_fault: np.ndarray, rule: AlertRule) -> np.ndarray:
    """Boolean array: True while the last ``k`` windows are all above the threshold."""
    above = p_fault > rule.threshold
    pos = np.arange(len(above))
    last_below = np.maximum.accumulate(np.where(above, -1, pos))
    return (pos - last_below) >= rule.k  # length of the current run above the threshold


def alert_onsets(p_fault: np.ndarray, rule: AlertRule) -> np.ndarray:
    """Boolean array: True at windows where a *new* alert fires (rising edge of the state)."""
    state = alert_state(p_fault, rule)
    return state & ~np.r_[False, state[:-1]]


def instance_report(df: pd.DataFrame, rule: AlertRule, stride_s: float = 60.0) -> dict[str, Any]:
    """Score one instance (its windows sorted by time).

    - ``normal_hours`` / ``false_alarms``: alerts firing on normal-labelled windows.
    - For instances containing fault windows: ``detected`` (an alert fires on a fault window),
      ``delay_min`` (first such alert minus the first fault window) and ``lead_min`` (steady
      state start minus that alert; positive = warned before the steady state; NaN if the
      instance has no steady-state or transient phase to compare against).
    """
    df = df.sort_values("end_time")
    state = alert_state(df["p_fault"].to_numpy(), rule)
    onsets = state & ~np.r_[False, state[:-1]]
    fault = (df["event_class"] > 0).to_numpy()
    out: dict[str, Any] = {
        "instance_id": df["instance_id"].iloc[0],
        "event_class": int(df["event_class"].max()),
        "normal_hours": float((~fault).sum() * stride_s / 3600),
        "false_alarms": int((onsets & ~fault).sum()),
        "has_fault": bool(fault.any()),
        "detected": False,
        "delay_min": np.nan,
        "lead_min": np.nan,
    }
    if not fault.any():
        return out
    times = df["end_time"].to_numpy()
    # An alert state that is already on when the fault starts also counts as detection.
    caught = np.flatnonzero(state & fault)
    if caught.size:
        first_fault = times[np.flatnonzero(fault)[0]]
        t_alert = times[caught[0]]
        out["detected"] = True
        out["delay_min"] = float((t_alert - first_fault) / np.timedelta64(1, "m"))
        steady = np.flatnonzero(fault & ~df["is_transient"].to_numpy())
        has_transient = bool(df["is_transient"].any())
        if has_transient and steady.size:
            out["lead_min"] = float((times[steady[0]] - t_alert) / np.timedelta64(1, "m"))
    return out


def early_warning_report(preds: pd.DataFrame, rule: AlertRule) -> pd.DataFrame:
    """One row per instance (see :func:`instance_report`)."""
    return pd.DataFrame(
        [instance_report(g, rule) for _, g in preds[PRED_COLUMNS].groupby("instance_id")]
    )


def summarize(report: pd.DataFrame) -> dict[str, float]:
    """Headline numbers: detection rate, lead time, delay, false alarms per 24 h."""
    faults = report[report["has_fault"]]
    hours = report["normal_hours"].sum()
    with_lead = faults["lead_min"].notna().sum()
    return {
        "fault_instances": int(len(faults)),
        "detection_rate": float(faults["detected"].mean()) if len(faults) else np.nan,
        "median_lead_min": float(faults["lead_min"].median()),
        "warned_before_steady_rate": float((faults["lead_min"] > 0).sum() / max(with_lead, 1)),
        "median_delay_min": float(faults["delay_min"].median()),
        "false_alarms_per_day": float(report["false_alarms"].sum() / hours * 24)
        if hours
        else np.nan,
        "normal_hours": float(hours),
    }


def per_class_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Detection rate, median lead and delay per event class."""
    faults = report[report["has_fault"]]
    return faults.groupby("event_class").agg(
        instances=("detected", "size"),
        detection_rate=("detected", "mean"),
        median_lead_min=("lead_min", "median"),
        median_delay_min=("delay_min", "median"),
    )


def tune_rule(
    preds: pd.DataFrame,
    thresholds: np.ndarray,
    ks: tuple[int, ...] = (1, 2, 3, 5),
    max_false_alarms_per_day: float = 1.0,
) -> tuple[AlertRule, pd.DataFrame]:
    """Grid-search the alert rule on *validation* predictions.

    Picks the highest detection rate whose false-alarm rate is within budget (ties: fewer
    false alarms, then shorter delay); if none is within budget, the fewest false alarms.
    """
    rows = []
    for k in ks:
        for t in thresholds:
            rule = AlertRule(float(t), int(k))
            rows.append(
                {"threshold": float(t), "k": int(k), **summarize(early_warning_report(preds, rule))}
            )
    grid = pd.DataFrame(rows)
    ok = grid[grid["false_alarms_per_day"] <= max_false_alarms_per_day]
    if len(ok):
        best = ok.sort_values(["detection_rate", "false_alarms_per_day", "median_delay_min"],
                              ascending=[False, True, True]).iloc[0]  # fmt: skip
    else:
        best = grid.sort_values(["false_alarms_per_day", "detection_rate"],
                                ascending=[True, False]).iloc[0]  # fmt: skip
    return AlertRule(float(best["threshold"]), int(best["k"])), grid


def expected_calibration_error(y_fault: np.ndarray, p_fault: np.ndarray, bins: int = 10) -> float:
    """ECE of P(fault): mean |accuracy - confidence| over equal-width bins, weighted by size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_fault, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_fault[m].mean() - p_fault[m].mean())
    return float(ece)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising validation NLL of softmax(log(proba) / T)."""
    logp = np.log(np.clip(proba, 1e-9, 1))
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 60)):
        z = logp / t
        z -= z.max(axis=1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    """Rescale class probabilities with temperature ``t``."""
    z = np.log(np.clip(proba, 1e-9, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------
# Command line: compare models and pick the production model + alert rule (on validation).
# --------------------------------------------------------------------------------------------


def _latest(root: Path, prefix: str) -> Path | None:
    runs = sorted(
        p for p in root.glob(f"{prefix}-*") if p.is_dir() and (p / "metrics.json").exists()
    )
    return runs[-1] if runs else None


def _with_p_fault(preds: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = preds[["instance_id", "end_time", "event_class", "is_transient"]].copy()
    out["p_fault"] = 1.0 - proba[:, 0]
    return out


def _plot_tradeoff(grids: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, grid in grids.items():
        # Best k for each threshold, then ordered by false-alarm rate.
        best = grid.sort_values("detection_rate").groupby("threshold").tail(1)
        best = best.sort_values(["false_alarms_per_day", "detection_rate"])
        # Markers only: detection is not monotonic in the false-alarm rate, so lines mislead.
        ax.scatter(best["false_alarms_per_day"], best["detection_rate"], s=18, label=name)
    ax.axvline(1.0, color="grey", ls="--", lw=1, label="budget: 1 false alarm / day")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set(
        xlabel="false alarms per 24 h of normal operation (validation)",
        ylabel="share of fault instances detected",
        title="Alert threshold trade-off",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def _plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    edges = np.linspace(0, 1, 11)
    for name, (y, p) in curves.items():
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", ms=3, label=name)
    ax.set(xlabel="predicted P(fault)", ylabel="observed fault frequency",
           title="Reliability (test, after temperature scaling)")  # fmt: skip
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv: list[str] | None = None) -> None:
    """Compare models on held-out wells and write ``configs/serving.yaml``."""
    import argparse
    import json
    import logging

    import yaml

    from owfd.config import PROJECT_ROOT, get_settings
    from owfd.models.metrics import classification_metrics
    from owfd.models.predictions import PROBA, predictions

    parser = argparse.ArgumentParser(description="Early-warning evaluation and model choice.")
    parser.add_argument("--models", nargs="*", type=Path, help="artifact folders (default: latest)")
    parser.add_argument("--max-false-alarms", type=float, default=1.0, help="per 24 h (val)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    artifacts = settings.resolve(settings.artifacts_dir)
    dirs = args.models or [
        d for d in (_latest(artifacts / "xgb", "xgb_baseline"), _latest(artifacts / "dl", "dl_cnn"),
                    _latest(artifacts / "dl", "dl_gru")) if d is not None
    ]  # fmt: skip
    names = {d: d.name.rsplit("-", 2)[0] for d in dirs}
    proba: dict[str, dict[str, np.ndarray]] = {}
    meta: dict[str, pd.DataFrame] = {}
    for d in dirs:
        log.info("Predicting with %s", d.name)
        for split in ("val", "test"):
            p = predictions(d, processed, split)
            proba.setdefault(names[d], {})[split] = p[PROBA].to_numpy()
            meta.setdefault(split, p)
    if len(dirs) > 1:
        proba["ensemble"] = {s: np.mean([proba[names[d]][s] for d in dirs], axis=0)
                             for s in ("val", "test")}  # fmt: skip

    thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)
    rows, grids, curves, reports = [], {}, {}, {}
    for name, pr in proba.items():
        y_val = meta["val"]["event_class"].to_numpy()
        y_test = meta["test"]["event_class"].to_numpy()
        t = fit_temperature(pr["val"], y_val)
        cal = {s: apply_temperature(pr[s], t) for s in ("val", "test")}
        val_preds = _with_p_fault(meta["val"], cal["val"])
        test_preds = _with_p_fault(meta["test"], cal["test"])
        rule, grids[name] = tune_rule(
            val_preds, thresholds, max_false_alarms_per_day=args.max_false_alarms
        )
        val_s = summarize(early_warning_report(val_preds, rule))
        reports[name] = early_warning_report(test_preds, rule)
        test_s = summarize(reports[name])
        y_fault = (y_test > 0).astype(float)
        curves[name] = (y_fault, test_preds["p_fault"].to_numpy())
        rows.append({
            "model": name, "temperature": t,
            "ece_test_raw": expected_calibration_error(y_fault, 1 - pr["test"][:, 0]),
            "ece_test_cal": expected_calibration_error(y_fault, test_preds["p_fault"].to_numpy()),
            "val_macro_f1": classification_metrics(y_val, cal["val"])["macro_f1"],
            "test_macro_f1": classification_metrics(y_test, cal["test"])["macro_f1"],
            "threshold": rule.threshold, "k": rule.k,
            **{f"val_{k}": v for k, v in val_s.items()},
            **{f"test_{k}": v for k, v in test_s.items()},
        })  # fmt: skip
    table = pd.DataFrame(rows).set_index("model")
    # Production choice on validation only: detection at the false-alarm budget, then macro F1.
    chosen = table.sort_values(["val_detection_rate", "val_macro_f1"], ascending=False).index[0]

    figs = PROJECT_ROOT / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _plot_tradeoff(grids, figs / "alert_tradeoff.png")
    _plot_reliability(curves, figs / "reliability.png")
    table.to_csv(PROJECT_ROOT / "docs" / "early_warning_comparison.csv")
    per_class = per_class_summary(reports[chosen])
    per_class.to_csv(PROJECT_ROOT / "docs" / "early_warning_per_class.csv")

    member_dirs = dirs if chosen == "ensemble" else [d for d in dirs if names[d] == chosen]
    serving = {
        "model": chosen,
        "artifacts": [str(d.relative_to(PROJECT_ROOT)) for d in member_dirs],
        "temperature": float(table.loc[chosen, "temperature"]),
        "alert": {
            "threshold": float(table.loc[chosen, "threshold"]),
            "k": int(table.loc[chosen, "k"]),
        },
        "selected_on": "validation: detection rate at <= "
        f"{args.max_false_alarms} false alarms/24 h, then macro F1",
    }
    (PROJECT_ROOT / "configs" / "serving.yaml").write_text(
        "# Written by python -m owfd.models.evaluate\n" + yaml.safe_dump(serving, sort_keys=False)
    )
    pd.set_option("display.width", 200)
    log.info("Chosen: %s\n%s\n%s", chosen, table.T.to_string(), per_class.to_string())
    (artifacts / "early_warning_summary.json").write_text(
        json.dumps(
            {"chosen": chosen, "table": table.reset_index().to_dict("records")},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
