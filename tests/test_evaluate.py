from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_trader.data import SymbolInfo
from super_trader.evaluate import (
    EvalConfig, MarketPanel, add_selection_flags, evaluate_prepared, evaluate_symbol,
    forward_returns, prepare_symbol,
)
from super_trader.rules import RULES
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------- 前瞻收益
def test_forward_returns_align_correctly():
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    open_ = np.array([9.5, 10.5, 11.5, 12.5, 13.5])
    cc, oc = forward_returns(close, open_, 2, 5)
    assert np.isnan(cc[3]) and np.isnan(cc[4])
    assert cc[0] == pytest.approx(12 / 10 - 1)
    assert oc[0] == pytest.approx(12 / 10.5 - 1)   # 次日开盘进
    assert np.isnan(oc[3])


def test_forward_returns_handles_short_series():
    cc, oc = forward_returns(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 5, 2)
    assert np.all(np.isnan(cc)) and np.all(np.isnan(oc))


# ---------------------------------------------------------------- 样本掩码
def _prep(df, code="600519", cat="个股", limit=0.10, cfg=None):
    info = SymbolInfo(code=code, name="测试", category=cat, path="x",
                      board="主板", price_limit=limit)
    return prepare_symbol(df, info, cfg or EvalConfig())


def test_prepare_rejects_short_history():
    assert _prep(make_ohlcv(200, seed=1)) is None


def test_masks_exclude_forward_windows_with_artifacts():
    df = make_ohlcv(900, seed=2)
    p = _prep(df)
    idx, fwd, fwd_oc, up = p.masks(1)
    # 在中间制造一次除权跳变
    df2 = df.copy()
    i = 700
    for col in ("open", "close", "high", "low"):
        df2.loc[i:, col] *= 0.6
    p2 = _prep(df2)
    idx2, *_ = p2.masks(1)
    # 跳变当天与前一天的前瞻窗口都应被剔除
    assert i not in set(idx2) and (i - 1) not in set(idx2)
    assert len(idx2) < len(idx)


def test_masks_respect_eval_start_and_horizon_tail():
    cfg = EvalConfig(eval_start="2015-01-01")
    p = _prep(make_ohlcv(900, seed=3), cfg=cfg)
    idx, *_ = p.masks(5)
    assert idx.max() <= p.n - 6


# ---------------------------------------------------------------- 评估与晋级
def test_evaluate_produces_expected_columns():
    df = make_ohlcv(900, seed=4)
    recs, meta = evaluate_symbol(df, SymbolInfo(
        code="600519", name="测试", category="个股", path="x",
        board="主板", price_limit=0.10))
    add_selection_flags(recs, EvalConfig())
    assert recs and meta["horizons"]
    keys = {"rule", "horizon", "n", "acc", "base", "edge", "p", "edge_train",
            "edge_hold", "p_hold", "promoted", "eras_total"}
    assert keys <= set(recs[0])
    for r in recs[:50]:
        assert 0.0 <= r["acc"] <= 1.0
        assert 0.0 <= r["p"] <= 1.0


def test_selection_flags_require_both_halves():
    cfg = EvalConfig(min_train_samples=10, min_holdout_samples=5)
    base = dict(n_train=50, edge_train=0.02, p_train=0.01,
                n_hold=50, edge_hold=0.02, p_hold=0.01,
                eras_total=3, eras_positive=3)
    good = add_selection_flags([dict(base)], cfg)[0]
    assert good["pass_train"] and good["pass_holdout"] and good["promoted"]

    bad_train = add_selection_flags([dict(base, p_train=0.5)], cfg)[0]
    assert not bad_train["promoted"]

    bad_hold = add_selection_flags([dict(base, edge_hold=-0.01)], cfg)[0]
    assert not bad_hold["promoted"]

    bad_eras = add_selection_flags([dict(base, eras_total=5, eras_positive=1)], cfg)[0]
    assert not bad_eras["promoted"]


def test_detects_planted_edge():
    """人为制造「连跌 5 日之后必涨」的序列，流程应该把它挑出来。"""
    n = 1500
    rng = np.random.default_rng(9)
    close = np.full(n, 100.0)
    down = 0
    for i in range(1, n):
        if down >= 5:
            step = abs(rng.standard_normal()) * 1.5 + 0.5    # 反弹
            down = 0
        else:
            step = -abs(rng.standard_normal()) * 0.6
            down += 1
        close[i] = close[i - 1] * (1 + step / 100)
    df = pd.DataFrame({
        "date": pd.bdate_range("2012-01-02", periods=n),
        "open": close, "close": close, "high": close * 1.005, "low": close * 0.995,
        "volume": np.full(n, 1e6), "amount": np.full(n, 1e7),
        "amplitude": np.full(n, 1.0), "pct_chg": np.full(n, 0.1),
        "change": np.full(n, 0.1), "turnover": np.full(n, 1.0),
    })
    recs, _ = evaluate_symbol(df, SymbolInfo(
        code="600519", name="测试", category="个股", path="x",
        board="主板", price_limit=0.0))
    hit = [r for r in recs if r["rule"] == "连续__consec_dn5_fade" and r["horizon"] == 1][0]
    assert hit["acc"] > hit["base"] + 0.2
    assert hit["p"] < 1e-6


# ---------------------------------------------------------------- 市场级面板
def _panel_with(n_symbols: int, signal_fn, seed0: int = 100):
    """构造 n 个标的、只留一条规则信号的合成面板。"""
    cfg = EvalConfig(horizons=[1], eval_start="2000-01-01")
    cal = pd.DatetimeIndex(make_ohlcv(600, seed=seed0)["date"])
    panel = MarketPanel(cal, [1])
    for k in range(n_symbols):
        df = make_ohlcv(600, seed=seed0 + k)
        p = _prep(df, cfg=cfg)
        assert p is not None
        for key in panel.rules:
            p.sigs[key] = np.zeros(p.n, dtype=np.int8)
        p.sigs["趋势__s16_05"] = signal_fn(p, k)
        panel.add(p, cfg)
    return panel, cfg


def test_market_panel_detects_planted_perfect_signal():
    """信号恰好等于真实未来方向 → 市场级优势应接近 0.5 且极显著。"""
    def oracle(p, k):
        idx, fwd, _oc, _up = p.masks(1)
        sig = np.zeros(p.n, dtype=np.int8)
        sig[idx] = np.where(fwd >= 0, 1, -1)
        return sig

    panel, cfg = _panel_with(6, oracle)
    res = panel.results(cfg, min_signals=3)
    row = res[res["rule"] == "趋势__s16_05"].iloc[0]
    assert row["market_mean_edge"] > 0.3
    assert row["market_p"] < 1e-6


def test_market_panel_without_skill_is_not_significant():
    rng = np.random.default_rng(42)

    def noise(p, k):
        return rng.choice([-1, 0, 1], size=p.n).astype(np.int8)

    panel, cfg = _panel_with(6, noise, seed0=200)
    res = panel.results(cfg, min_signals=3)
    row = res[res["rule"] == "趋势__s16_05"].iloc[0]
    assert row["market_p"] > 0.01


def test_specification_curve_has_all_variants():
    def oracle(p, k):
        idx, fwd, _oc, _up = p.masks(1)
        sig = np.zeros(p.n, dtype=np.int8)
        sig[idx] = np.where(fwd >= 0, 1, -1)
        return sig

    panel, cfg = _panel_with(6, oracle)
    cfg.min_market_signals = 3
    curve = panel.specification_curve(cfg, base_modes=("per_symbol", "global"),
                                      lags=(5, 21), thresholds=(3,))
    assert not curve.empty
    assert {"variants", "variant_pass_share"} <= set(curve.columns)
    row = curve[curve["rule"] == "趋势__s16_05"].iloc[0]
    assert row["variants"] == 4
    assert row["variant_pass_share"] == pytest.approx(1.0)
