import numpy as np
import pytest

from owfd.models.metrics import N_CLASSES, classification_metrics, flat_metrics, plot_confusion


def one_hot(pred: list[int]) -> np.ndarray:
    proba = np.full((len(pred), N_CLASSES), 0.01)
    proba[np.arange(len(pred)), pred] = 0.91
    return proba


def test_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 2, 2])
    m = classification_metrics(y, one_hot(y.tolist()))
    assert m["macro_f1"] == 1 and m["balanced_accuracy"] == 1 and m["fault_roc_auc"] == 1


def test_macro_f1_ignores_absent_classes() -> None:
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, one_hot([0, 1, 1, 1]))
    # class 0: P=1 R=.5 F1=.667; class 1: P=.667 R=1 F1=.8 -> macro over {0, 1}
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["per_class"][5]["support"] == 0
    assert m["confusion"][0][1] == 1


def test_fault_metrics_and_flattening() -> None:
    y = np.array([0, 0, 3, 3])
    m = classification_metrics(y, one_hot([0, 3, 4, 3]))  # fault vs normal: 3 of 4 right
    assert m["fault_f1"] == pytest.approx(0.8)
    flat = flat_metrics(m, "val")
    assert flat["val_macro_f1"] == m["macro_f1"] and "val_f1_c3" in flat and "val_f1_c5" not in flat
    assert plot_confusion(m, "t") is not None


def test_single_class_gives_nan_auc() -> None:
    m = classification_metrics(np.array([0, 0]), one_hot([0, 0]))
    assert np.isnan(m["fault_roc_auc"])
