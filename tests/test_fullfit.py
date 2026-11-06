"""全历史拟合：权重衰减、有效样本量、覆盖率下限与内样本口径。"""
from __future__ import annotations

import numpy as np
import pytest

from super_trader.data import SymbolInfo
from super_trader.evaluate import EvalConfig, prepare_symbol
from super_trader.fullfit import (
    FullFitConfig, _eff_n, _labels, _vote, _wacc, _weights, fit_symbol,
)
from tests.conftest import make_ohlcv


def _prep(df=None):
    df = make_ohlcv(1200, seed=11) if df is None else df
    info = SymbolInfo(code="T", name="测试", category="指数", path="x", board="指数", price_limit=0.0)
    return prepare_symbol(df, info, EvalConfig())


def test_weights_decay_with_age_but_stay_positive():
    p = _prep()
    idx, _ = _labels(p, 1)
    w = _weights(p, idx, 365.0, floor=0.1)
    assert np.all(w > 0) and np.all(w <= 1.0)
    assert w[-1] > w[0]                      # 越近越高
    assert w[-1] <= 1.0 and w[0] >= 0.1      # 有下限，不会塌到 0
    assert w.max() / w.min() < 20            # 不会跨越数量级


def test_eff_n_penalises_concentrated_weights():
    assert _eff_n(np.ones(100)) == pytest.approx(100.0)
    skewed = np.concatenate([np.ones(1), np.full(99, 1e-4)])
    assert _eff_n(skewed) < 5


def test_weighted_accuracy_uses_only_issued_predictions():
    pred = np.array([1, -1, 0, 1])
    actual = np.array([1, 1, 1, -1])
    w = np.ones(4)
    acc, n = _wacc(pred, actual, w)
    assert n == 3 and acc == pytest.approx(1 / 3)


def test_vote_ties_and_empty_are_neutral():
    stack = np.array([[1, -1, 0, 1], [1, 1, 0, -1]])
    assert _vote(stack).tolist() == [1, 0, 0, 0]


def test_labels_do_not_reach_past_last_bar():
    p = _prep()
    idx, actual = _labels(p, 20)
    assert idx.max() + 20 < p.n
    assert set(np.unique(actual)) <= {-1, 0, 1}


def test_fit_returns_above_half_accuracy_with_floor_respected():
    p = _prep()
    row = fit_symbol(p, FullFitConfig())
    assert row is not None
    assert row["weighted_accuracy"] >= 0.5          # 反向兜底保证过线
    assert row["predictions"] > 0 and 0 < row["coverage"] <= 1
    assert row["eff_n"] > 0
    assert row["rules"] and all(isinstance(r, str) for r in row["rules"])


def test_config_rejects_nonsense():
    for bad in ({"min_signals": 0}, {"target_accuracy": 1.5}, {"max_rules": 0},
                {"weight_floor": 1.0}, {"min_coverage": -0.1}, {"half_life_days": 0}):
        with pytest.raises(ValueError):
            FullFitConfig(**bad)
