"""全市场明细表：中性占比、单边剔除与类型汇总。"""
from __future__ import annotations

import pytest

from super_trader.fullfit_table import enrich, summarize_by_type, _pack


def _rec(acc=0.66, base=0.52, preds=400, elig=1000, cat="个股", h=10, direction=1):
    return {"code": "000001", "name": "测试", "category": cat, "horizon": h,
            "weighted_accuracy": acc, "plain_accuracy": acc, "base_up_rate": base,
            "predictions": preds, "eligible": elig, "latest_direction": direction,
            "direction_label": "涨", "probability": acc, "rules": ["a__b"],
            "recent_accuracy": acc, "skill_pp": None}


def test_neutral_share_is_complement_of_coverage():
    (r,) = enrich([_rec(preds=250, elig=1000)])
    assert r["neutral_count"] == 750
    assert r["neutral_share"] == pytest.approx(0.75)
    assert r["coverage"] == pytest.approx(0.25)


def test_zero_eligible_does_not_divide_by_zero():
    (r,) = enrich([_rec(preds=0, elig=0)])
    assert r["neutral_share"] is None and r["coverage"] is None


def test_skill_is_edge_over_best_constant_guess():
    (r,) = enrich([_rec(acc=0.60, base=0.55)])       # 猜涨 55%，模型 60%
    assert r["skill_pp"] == pytest.approx(5.0)
    (r,) = enrich([_rec(acc=0.60, base=0.30)])       # 猜跌 70%，模型 60% → 负
    assert r["skill_pp"] == pytest.approx(-10.0)


def test_one_sided_flagged_by_extreme_base_rate():
    rows = enrich([_rec(base=0.94), _rec(base=0.52), _rec(base=0.10)])
    assert [r["one_sided"] for r in rows] == [True, False, True]


def test_summary_by_type_averages_accuracy_and_neutral():
    rows = enrich([_rec(acc=0.60, preds=500, cat="指数"),
                   _rec(acc=0.70, preds=500, cat="指数"),
                   _rec(acc=0.65, preds=500, cat="个股")])
    stats = {s["type"]: s for s in summarize_by_type(rows)}
    assert stats["指数"]["count"] == 2
    assert stats["指数"]["accuracy"] == pytest.approx(0.65)
    assert stats["指数"]["neutral"] == pytest.approx(0.5)


def test_pack_marks_neutral_rows_with_base_rate():
    row = enrich([_rec(direction=0)])[0]
    row["probability"] = None
    row["rank"] = 1
    packed = _pack(row)
    assert packed[11] == "flat" and "中性" in packed[10]
