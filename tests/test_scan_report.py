"""端到端冒烟：用合成数据跑完整扫描 + 生成报告，确保流程不会在真实数据外就散架。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import super_trader.scan as scan_mod
import super_trader.universe as uni
from super_trader.config import Config
from super_trader.data import SymbolInfo
from super_trader.report import build_reports
from tests.conftest import make_ohlcv


def _make_store(tmp_path, specs):
    root = tmp_path / "日线"
    out = []
    for code, name, cat, seed in specs:
        d = root / cat
        d.mkdir(parents=True, exist_ok=True)
        df = make_ohlcv(900, seed=seed)
        df.to_csv(d / f"{code}_{name}.csv", index=False, encoding="utf-8-sig")
        limit = 0.0 if cat == "指数" else (0.10 if cat == "ETF" else 0.10)
        board = "指数" if cat == "指数" else ("ETF" if cat == "ETF" else "主板")
        out.append((SymbolInfo(code=code, name=name, category=cat,
                               path=str(d / f"{code}_{name}.csv"),
                               board=board, price_limit=limit), "宽基" if cat == "指数" else "ETF"))
    return root, out


@pytest.fixture()
def scan_artifacts(tmp_path, monkeypatch):
    root, members = _make_store(tmp_path, [
        ("000001", "上证指数", "指数", 1),
        ("000300", "沪深300", "指数", 2),
        ("510300", "沪深300ETF", "ETF", 3),
    ])
    monkeypatch.setattr(uni, "resolve_pool", lambda store, name="mainstream": members)
    cfg = Config(
        data_root=str(root),
        cache_dir=str(tmp_path / "cache"),
        report_dir=str(tmp_path / "reports"),
        min_bars=300, warmup_bars=60, horizons=[1, 5],
        min_train_samples=30, min_holdout_samples=15,
        extra={"eval_start": "2000-01-01"},
    )
    outcome = scan_mod.run_scan(cfg, pool="宽基", out_name="test_run", verbose=False)
    return cfg, outcome


def test_scan_writes_all_artifacts(scan_artifacts):
    cfg, outcome = scan_artifacts
    out = outcome.out_dir
    for name in ("results.json", "rule_ranking.csv", "symbol_summary.csv",
                 "adopted_rules.json", "summary.json"):
        assert (out / name).exists(), name
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["totals"]["symbols_evaluated"] == 3
    assert payload["totals"]["tests"] > 0
    assert payload["market_date"]
    assert payload["gate"]["holdout_p_max"] == cfg.holdout_p_max


def test_scan_ranking_has_required_columns(scan_artifacts):
    _, outcome = scan_artifacts
    ranking = pd.DataFrame(outcome.payload["rule_ranking"])
    assert not ranking.empty
    for col in ("rule", "horizon", "symbols", "positive_share", "market_mean_edge",
                "market_p", "market_q", "variant_pass_share", "passes_consistency"):
        assert col in ranking.columns
    assert ranking["passes_consistency"].dtype == bool
    # 只有 3 个标的，不可能满足「跨 20 个标的」的验收线
    assert not ranking["passes_consistency"].any()


def test_report_renders_and_embeds_data(scan_artifacts):
    _, outcome = scan_artifacts
    paths = build_reports(outcome.payload, outcome.out_dir)
    html = paths[0].read_text(encoding="utf-8")
    md = paths[1].read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "window.DATA" in html
    assert "验收线" in html
    assert "标的级筛选是噪声" in md or "标杆" in md


def test_report_survives_empty_results(tmp_path):
    """没有通过规则时，报告也不能崩。"""
    payload = {
        "generated_at": "2026-01-01 00:00:00", "market_date": "2026-01-01",
        "pool": "mainstream",
        "gate": {"train_p_max": 0.1, "holdout_p_max": 0.05, "fdr_q": 0.1, "eval_start": "2005-01-01"},
        "totals": {"tests": 0, "rules": 0, "symbols_evaluated": 0, "promoted": 0},
        "symbol_summary": [], "rule_ranking": [], "adopted": {}, "records": [],
    }
    paths = build_reports(payload, tmp_path / "empty")
    assert paths[0].exists() and paths[1].exists()
    assert "本轮没有任何规则通过验收线 A" in paths[0].read_text(encoding="utf-8")


def test_predict_handles_empty_adoption(scan_artifacts):
    """只有 3 个标的时不会有晋级，predict 要给出提示而不是崩。"""
    from super_trader.predict import predict_today

    cfg, outcome = scan_artifacts
    assert predict_today(cfg, name="test_run", top=5) == 0
