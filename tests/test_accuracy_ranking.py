"""标的准确率排行：分母只有已发出的预测，排序与小样本标记要可复现。"""
from __future__ import annotations

import json
import re

import pytest

from super_trader.accuracy_ranking import aggregate_by_symbol, build_ranking, build_ranking_html


def _rec(code, name, horizon, pred, corr, elig, base=0.5, cat="指数"):
    return {"code": code, "name": name, "category": cat, "horizon": horizon,
            "model": {"rules": ["r"], "threshold": .7, "k": 1},
            "test": {"predictions": pred, "correct": corr, "eligible": elig,
                     "accuracy": (corr / pred) if pred else None,
                     "coverage": pred / elig if elig else None,
                     "majority_baseline": base, "neutral": elig - pred}}


def test_ranks_by_accuracy_desc_then_sample_size():
    rows = aggregate_by_symbol([
        _rec("A", "甲", 1, 100, 60, 400),   # 60%
        _rec("B", "乙", 1, 900, 540, 1000),  # 60% 但样本大
        _rec("C", "丙", 1, 100, 40, 400),   # 40%
    ])
    assert [r["code"] for r in rows] == ["B", "A", "C"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_neutral_excluded_from_denominator_across_horizons():
    rows = aggregate_by_symbol([
        _rec("A", "甲", 1, 100, 60, 400),
        _rec("A", "甲", 5, 100, 50, 400),
    ])
    (row,) = rows
    assert row["predictions"] == 200 and row["correct"] == 110
    assert row["accuracy"] == pytest.approx(.55)
    # 覆盖率只按可评估样本算，中性部分体现在这里
    assert row["coverage"] == pytest.approx(200 / 800)
    assert row["periods_used"] == 2


def test_empty_symbol_is_neutral_not_zero_accuracy():
    rows = aggregate_by_symbol([_rec("A", "甲", 1, 0, 0, 400)])
    (row,) = rows
    assert row["accuracy"] is None and row["predictions"] == 0
    assert row["rank"] == 1


def test_relative_edge_uses_majority_baseline():
    (row,) = aggregate_by_symbol([_rec("A", "甲", 1, 100, 60, 400, base=.55)])
    assert row["edge_pp"] == pytest.approx((.60 - .55) * 100)


def test_ranking_totals_reconcile_with_source_totals():
    data = build_ranking([_rec("A", "甲", 1, 100, 60, 400), _rec("B", "乙", 3, 50, 20, 300)])
    assert data["totals"]["predictions"] == 150
    assert data["totals"]["correct"] == 80
    assert data["totals"]["accuracy"] == pytest.approx(80 / 150)


def test_html_lists_both_tables_and_marks_small_samples(tmp_path):
    payload = {"generated_at": "2026-01-01T00:00:00", "method": "m",
               "records": [_rec("A", "甲", 1, 5, 4, 400), _rec("B", "乙", 1, 800, 400, 1000)]}
    out = build_ranking_html(payload, out_path=tmp_path / "rank.html")
    html = out.read_text(encoding="utf-8")
    assert html.count('id="rank"') == 1 and html.count('id="detail"') == 1
    assert "80.00%" in html          # A: 4/5
    assert "50.00%" in html          # B: 400/800
    assert "小样本" in html           # A 只有 5 次预测
    assert "中性（弃权）不计入分母" in html
    # 排序按准确率：A(80%) 排在 B(50%) 之前
    body = html.split('id="rank"')[1]
    assert body.index("甲") < body.index("乙")


def test_html_escapes_untrusted_names(tmp_path):
    payload = {"records": [_rec("A", "<script>x</script>", 1, 10, 5, 100)]}
    html = build_ranking_html(payload, out_path=tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<script>x</script>" not in html and "&lt;script&gt;" in html
