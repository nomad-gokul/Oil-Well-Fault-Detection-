"""Write a small synthetic serving bundle (for CI image builds and smoke tests).

Usage: python scripts/make_test_bundle.py build/bundle
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.bundle import build_bundle  # noqa: E402

if __name__ == "__main__":
    out = build_bundle(Path(sys.argv[1] if len(sys.argv) > 1 else "build/bundle"))
    print(f"Synthetic test bundle -> {out}")
