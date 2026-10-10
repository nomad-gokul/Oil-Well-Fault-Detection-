"""Fetch the API's fleet drift check and write a self-contained HTML report.

Usage: python scripts/drift_report.py [--api http://localhost:8000] [--out drift_report.html]
Exit code 1 when drift is flagged (usable as a scheduled CI/cron check).
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

COLORS = {"stable": "#27ae60", "drift": "#c0392b", "insufficient_data": "#7f8c8d"}


def render(report: dict, api: str) -> str:
    """HTML page for a /monitoring/drift response."""
    rows = []
    for sensor, r in report["sensors"].items():
        color = COLORS[r["status"]]
        bar = f"<div style='background:{color};height:10px;width:{r['share_unusual'] * 100:.0f}%'>"
        rows.append(
            f"<tr><td>{html.escape(sensor)}</td><td style='color:{color}'>"
            f"{r['status'].replace('_', ' ')}</td><td>{r['wells_unusual']} / {r['wells_checked']}"
            f"</td><td>{bar}</div></td><td>{html.escape(', '.join(r['unusual_wells']))}</td></tr>"
        )
    flag = "FLEET DRIFT DETECTED" if report["drift"] else "No fleet-wide drift"
    color = COLORS["drift"] if report["drift"] else COLORS["stable"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}}
td,th{{padding:.4rem .8rem;border-bottom:1px solid #ddd;text-align:left}}
td:nth-child(4){{width:30%}}</style></head><body><h1>Fleet data drift report</h1>
<p><b style="color:{color}">{flag}</b></p>
<p>Last {report["window_s"] // 3600} h of readings per well at {html.escape(api)}, deviations from
each well's own baseline vs real training/validation wells &middot;
{datetime.now(UTC):%Y-%m-%d %H:%M} UTC</p>
<table><tr><th>Sensor</th><th>Status</th><th>Unusual wells</th><th>Share</th><th>Wells</th></tr>
{"".join(rows)}</table></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=Path("drift_report.html"))
    args = parser.parse_args()
    url = f"{args.api.rstrip('/')}/monitoring/drift"
    report = httpx.get(url, timeout=60).raise_for_status().json()
    args.out.write_text(render(report, args.api))
    print(f"{'DRIFT' if report['drift'] else 'ok'} -> {args.out}")
    return 1 if report["drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
