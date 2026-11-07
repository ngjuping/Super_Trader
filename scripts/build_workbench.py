"""Build an offline, single-file research workbench; no server/CDN needed.

Run: python scripts/build_workbench.py
Sources live under web/workbench. Existing quantitative research is embedded as
an immutable snapshot, while all simulated interactions run in the browser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_trader.universe import MAINSTREAM, category_for

FIELDS = (
    "category", "code", "name", "horizon", "rules", "weighted_accuracy",
    "plain_accuracy", "predictions", "eligible", "coverage", "recent_accuracy",
    "latest_direction", "base_up_rate",
)


def snapshot_data(source: Path, snapshot: str) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [{key: row.get(key) for key in FIELDS} for row in payload["records"]]
    keys = {(row["category"], row["code"]) for row in records}
    if len(keys) != len(records):
        raise ValueError("Snapshot contains duplicate category/code identities")
    return {
        "snapshot": snapshot,
        "generated_at": payload["generated_at"],
        "totals": payload["totals"],
        "mainstream": [f"{category_for(m)}|{m.code}" for m in MAINSTREAM],
        "records": records,
    }


def build(output: Path | None = None, source: Path | None = None,
          snapshot: str = "2026-11-05") -> Path:
    assets = ROOT / "web" / "workbench"
    output = output or ROOT / "reports" / "chat_main.html"
    source = source or ROOT / "reports" / "fullfit" / "fullfit.json"
    html = (assets / "index.html").read_text(encoding="utf-8")
    css = (assets / "workbench.css").read_text(encoding="utf-8")
    js = (assets / "app.js").read_text(encoding="utf-8")
    data = json.dumps(snapshot_data(source, snapshot), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    for marker, replacement in (
        ("/*__WORKBENCH_CSS__*/", css),
        ("/*__WORKBENCH_DATA__*/", data),
        ("/*__WORKBENCH_JS__*/", js),
    ):
        if html.count(marker) != 1:
            raise ValueError(f"Template must have exactly one {marker}")
        html = html.replace(marker, replacement, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", default="2026-11-05")
    args = parser.parse_args()
    path = build(args.output, args.source, args.snapshot)
    print(f"{path} ({path.stat().st_size:,} bytes)")
