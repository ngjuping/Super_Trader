"""Build-time checks for the standalone offline research workbench."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_workbench", ROOT / "scripts" / "build_workbench.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_snapshot_keeps_category_and_code_as_identity():
    data = BUILDER.snapshot_data(ROOT / "reports" / "fullfit" / "fullfit.json", "2026-11-05")
    assert len(data["records"]) == 6244
    assert len(data["mainstream"]) == 62
    twins = [r for r in data["records"] if r["code"] == "000001"]
    assert {r["category"] for r in twins} == {"指数", "个股"}
    assert all(r["coverage"] >= 0.1 for r in data["records"])


def test_single_file_contains_data_styles_and_script(tmp_path):
    path = BUILDER.build(output=tmp_path / "chat_main.html")
    html = path.read_text(encoding="utf-8")
    assert "/*__WORKBENCH_" not in html
    assert '<style>' in html and '<script>' in html
    assert '<script src=' not in html
    assert 'https://cdn.' not in html
    raw = re.search(r'<script id="workbench-data" type="application/json">(.*?)</script>', html, re.S)[1]
    data = json.loads(raw)
    assert len(data["records"]) == 6244
    assert "无模型调用" in html and "离线演示" in html
    assert 'name="minCoverage"' not in html  # Form renders from app.js, not an extra stale form.


def test_builder_is_reproducible(tmp_path):
    a = BUILDER.build(output=tmp_path / "a.html").read_bytes()
    b = BUILDER.build(output=tmp_path / "b.html").read_bytes()
    assert a == b


def test_embedded_json_cannot_close_script(tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps({
        "generated_at": "2026-11-06", "totals": {"symbols": 1},
        "records": [{"category": "个股", "code": "000001", "name": "</script><b>unsafe</b>"}],
    }), encoding="utf-8")
    html = BUILDER.build(source=source, output=tmp_path / "page.html").read_text(encoding="utf-8")
    raw = re.search(r'<script id="workbench-data" type="application/json">(.*?)</script>', html, re.S)[1]
    assert "</script>" not in raw
    assert json.loads(raw)["records"][0]["name"] == "</script><b>unsafe</b>"


def test_duplicate_snapshot_ids_rejected(tmp_path):
    source = tmp_path / "duplicate.json"
    row = {"category": "个股", "code": "000001"}
    source.write_text(json.dumps({"generated_at":"x", "totals":{}, "records":[row,row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        BUILDER.snapshot_data(source, "2026-11-05")
