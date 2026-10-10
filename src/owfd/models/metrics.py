"""Shared evaluation metrics for every model (window-level classification)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from owfd.data.schema import CLASS_NAMES

N_CLASSES = len(CLASS_NAMES)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Multiclass + fault/no-fault metrics from class probabilities ``proba`` [n, 10].

    Macro F1 and balanced accuracy average over the classes *present* in ``y_true`` (a split
    may lack a class; scoring absent classes as 0 would be meaningless).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = proba.argmax(axis=1)
    present = np.unique(y_true)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    fault_true = y_true > 0
    fault_score = 1.0 - proba[:, 0]
    both = fault_true.any() and (~fault_true).any()
    return {
        "n": int(len(y_true)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(r[present].mean()),  # mean recall over present classes
        "accuracy": float((y_true == y_pred).mean()),
        "fault_roc_auc": float(roc_auc_score(fault_true, fault_score)) if both else float("nan"),
        "fault_pr_auc": float(average_precision_score(fault_true, fault_score))
        if both
        else float("nan"),
        "fault_f1": float(f1_score(fault_true, y_pred > 0, zero_division=0)),
        "per_class": {
            int(c): {
                "precision": float(p[c]),
                "recall": float(r[c]),
                "f1": float(f[c]),
                "support": int(support[c]),
            }  # fmt: skip
            for c in range(N_CLASSES)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES)).tolist(),
    }


def flat_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    """Scalar metrics (incl. per-class F1) flattened for experiment tracking."""
    out = {f"{prefix}_{k}": v for k, v in metrics.items() if isinstance(v, float)}
    out |= {f"{prefix}_f1_c{c}": m["f1"] for c, m in metrics["per_class"].items() if m["support"]}
    return out


def plot_confusion(metrics: dict[str, Any], title: str) -> plt.Figure:
    """Row-normalised confusion matrix (recall per true class)."""
    cm = np.asarray(metrics["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")  # fmt: skip
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels([f"{c} {CLASS_NAMES[c]}" for c in range(N_CLASSES)], fontsize=7)
    ax.set(xlabel="predicted class", ylabel="true class", title=title)
    fig.tight_layout()
    return fig
