"""横截面层：IC 口径、分位价差与新息-West t 的基本正确性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_trader.crosssection import (
    XSConfig, compute_xs, ic_series, quantile_spread, _newey_west_t, evaluate,
)


def _panel(n_days=260, n_sym=40, seed=0, momentum=0.0):
    """构造一个带可控动量/反转的面板。momentum>0 表示前期涨的继续涨。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prev = np.zeros(n_sym)
    rows = []
    price = np.full(n_sym, 100.0)
    for _ in range(n_days):
        shock = rng.standard_normal(n_sym) * 0.01
        drift = momentum * prev            # 用上一段收益驱动下一段
        ret = shock + drift
        price = price * (1 + ret)
        rows.append(price.copy())
        prev = ret
    return pd.DataFrame(rows, index=dates,
                        columns=[f"个股|{i:04d}" for i in range(n_sym)])


def test_ic_detects_planted_momentum_and_reversal():
    cfg = XSConfig(lookbacks=(5,), horizons=(5,), min_symbols_per_day=10)
    for momentum, expect in ((0.8, 1), (-0.8, -1)):
        panel = _panel(momentum=momentum)
        feats = compute_xs(panel, cfg)
        ic = ic_series(feats["rs5_rank"], feats["fwd5_rel"], 10)
        assert ic.notna().sum() > 100
        assert np.sign(ic.mean()) == expect, (momentum, ic.mean())


def test_ic_is_nan_when_market_is_single_sided():
    """横截面只有一只股票时，排名无意义，应返回空。"""
    panel = _panel(n_sym=1)
    cfg = XSConfig(lookbacks=(5,), horizons=(5,), min_symbols_per_day=10)
    feats = compute_xs(panel, cfg)
    ic = ic_series(feats["rs5_rank"], feats["fwd5_rel"], 10)
    assert ic.notna().sum() == 0


def test_quantile_spread_sign_follows_ic():
    cfg = XSConfig(lookbacks=(5,), horizons=(5,), min_symbols_per_day=10, quantiles=5)
    panel = _panel(momentum=0.8)
    feats = compute_xs(panel, cfg)
    spread = quantile_spread(feats["rs5_rank"], feats["fwd5_rel"], 5, 10)
    ic = ic_series(feats["rs5_rank"], feats["fwd5_rel"], 10)
    assert np.sign(spread.mean()) == np.sign(ic.mean())


def test_newey_west_t_rejects_noise_and_accepts_constant_signal():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.standard_normal(500))
    assert abs(_newey_west_t(noise, lag=20)) < 2.5
    strong = pd.Series(rng.standard_normal(500) * 0.1 + 0.3)
    assert _newey_west_t(strong, lag=20) > 2.0


def test_relative_returns_strip_market_move():
    panel = _panel(n_sym=20)
    cfg = XSConfig(lookbacks=(5,), horizons=(5,))
    feats = compute_xs(panel, cfg)
    # 每一天的相对收益中位数应接近 0
    med = feats["fwd5_rel"].median(axis=1).dropna()
    assert np.allclose(med.to_numpy(), 0.0, atol=1e-9)


def test_evaluate_returns_expected_shape():
    cfg = XSConfig(lookbacks=(5, 20), horizons=(5,), min_symbols_per_day=10)
    panel = _panel(momentum=0.5)
    meta = pd.DataFrame({"key": panel.columns, "code": [c.split("|")[1] for c in panel.columns],
                         "name": "x", "category": "个股"})
    out = evaluate(panel, meta, cfg)
    assert out["panel_shape"] == [panel.shape[0], panel.shape[1]]
    # 2 个回看 × 1 周期 + 1 个波动分层
    assert len(out["ic"]) == 3
    assert len(out["absolute"]) == 2
