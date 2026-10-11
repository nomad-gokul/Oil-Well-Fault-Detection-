"""Copy the newest serving bundle to deploy/bundle/ (committed; used by the API image).

Usage: python scripts/publish_bundle.py [artifacts/serving/<version>]

XGBoost models are gzipped (a few hundred KB) and the manifest is updated to match.
"""

import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "bundle"


def main() -> None:
    src = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else sorted((ROOT / "artifacts/serving").iterdir())[-1]
    )
    manifest = json.loads((src / "manifest.json").read_text())
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for member in manifest["members"]:
        data = (src / member["file"]).read_bytes()
        if member["kind"] == "xgb":
            member["file"] += ".gz"
            data = gzip.compress(data, compresslevel=9, mtime=0)  # reproducible bytes
        (OUT / member["file"]).write_bytes(data)
    chronic = manifest.get("chronic")
    if chronic:  # per-well instability profiles, gzipped like the main model
        (OUT / "profiles").mkdir()
        for meta in chronic["profiles"].values():
            data = gzip.compress((src / "profiles" / meta["file"]).read_bytes(), 9, mtime=0)
            meta["file"] += ".gz"
            (OUT / "profiles" / meta["file"]).write_bytes(data)
    for name in ("scaler.json", "reference.json"):
        if (src / name).exists():
            shutil.copy(src / name, OUT / name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    where = OUT.relative_to(ROOT)
    print(f"{manifest['version']} ({manifest['model']}) -> {where} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
