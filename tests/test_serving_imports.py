"""The serving image ships without training-only packages: serving code must not import them."""

import subprocess
import sys

TRAINING_ONLY = ["torch", "sklearn", "matplotlib", "pyarrow", "mlflow", "optuna", "shap", "seaborn"]
SERVING_MODULES = [
    "owfd.api.main",
    "owfd.inference.predictor",
    "owfd.inference.alerting",
    "owfd.models.explain",
    "owfd.simulator.replay",
    "owfd.dashboard.client",
]


def test_serving_modules_import_without_training_packages() -> None:
    code = (
        "import sys\n"
        f"for m in {TRAINING_ONLY!r}:\n"
        "    sys.modules[m] = None  # any import of it now raises ImportError\n"
        f"for m in {SERVING_MODULES!r}:\n"
        "    __import__(m)\n"
        "from owfd.models.explain import Explainer\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
