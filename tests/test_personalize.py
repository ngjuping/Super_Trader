"""个性化规则层：四段复现门控 + 安慰剂确实能杀掉假信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_trader.data import SymbolInfo
from super_trader.evaluate import EvalConfig, prepare_symbol
from super_trader.personalize import (
    PersonalConfig, add_gain_columns, best_per_symbol, evaluate_symbol_personal, _fold_blocks,
)


def _planted_df(n: int = 1600, seed: int = 3) -> pd.DataFrame:
    """构造「连跌 5 日后必反弹」的序列，用来验证四段复现门控真的认得出来。"""
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0)
    down = 0
    for i in range(1, n):
        if down >= 5:
            step = abs(rng.standard_normal()) * 1.4 + 0.6
            down = 0
        else:
            step = -abs(rng.standard_normal()) * 0.5
            down += 1
        close[i] = close[i - 1] * (1 + step / 100)
    return pd.DataFrame({
        "date": pd.bdate_range("2012-01-02", periods=n),
        "open": close, "close": close, "high": close * 1.004, "low": close * 0.996,
        "volume": np.full(n, 1e6), "amount": np.full(n, 1e7),
        "amplitude": np.full(n, 1.0), "pct_chg": np.full(n, 0.1),
        "change": np.full(n, 0.1), "turnover": np.full(n, 1.0),
    })


def _prep(df):
    info = SymbolInfo(code="600519", name="测试", category="个股", path="x",
                      board="主板", price_limit=0.0)
    return prepare_symbol(df, info, EvalConfig())


def test_fold_blocks_cover_holdout_without_overlap():
    pos = np.arange(100, 200)
    blocks = _fold_blocks(pos, 4)
    assert len(blocks) == 4
    assert sum(b.size for b in blocks) == 100
    assert all(a[-1] < b[0] for a, b in zip(blocks, blocks[1:]))


def test_planted_signal_passes_four_fold_replication():
    p = _prep(_planted_df())
    recs = evaluate_symbol_personal(p, EvalConfig(), PersonalConfig())
    hit = [r for r in recs if r["rule"] == "连续__consec_dn5_fade" and r["horizon"] == 1]
    assert hit, "规则没被评估到"
    r = hit[0]
    assert r["folds_positive"] == 4
    assert r["edge_hold"] > 0.1
    assert r["validated"], r


def test_placebo_permutation_kills_the_planted_signal():
    p = _prep(_planted_df())
    rng = np.random.default_rng(1)
    recs = evaluate_symbol_personal(p, EvalConfig(), PersonalConfig(), rng=rng)
    hit = [r for r in recs if r["rule"] == "连续__consec_dn5_fade" and r["horizon"] == 1]
    assert hit
    assert not hit[0]["validated"]


def test_random_signal_does_not_validate():
    """把信号换成随机方向，四段复现应该基本挡得住。"""
    p = _prep(_planted_df())
    rng = np.random.default_rng(7)
    for key in list(p.sigs):
        p.sigs[key] = rng.choice([-1, 0, 1], size=p.n).astype(np.int8)
    recs = evaluate_symbol_personal(p, EvalConfig(), PersonalConfig())
    assert sum(r["validated"] for r in recs) <= 3


def test_fold_gate_rejects_when_one_fold_flips():
    """训练段很漂亮，但留出段最后一块掉头 → 不能通过。"""
    p = _prep(_planted_df())
    ecfg = EvalConfig()
    recs = evaluate_symbol_personal(p, ecfg, PersonalConfig())
    target = [r for r in recs if r["rule"] == "连续__consec_dn5_fade" and r["horizon"] == 1][0]
    assert target["validated"]

    # 把最后一块的标签反过来，模拟「规律在这一段失效」
    cfg_strict = PersonalConfig()
    p2 = _prep(_planted_df())
    idx, _f, _fo, up = p2.masks(1)
    hold = np.where(idx >= p2.split_idx)[0]
    last = _fold_blocks(hold, cfg_strict.n_folds)[-1]
    flipped = up.copy()
    flipped[last] = ~flipped[last]
    p2.masks.__self__._mask_cache[1] = (idx, _f, _fo, flipped)  # 注入被打乱的标签
    recs2 = evaluate_symbol_personal(p2, ecfg, cfg_strict)
    t2 = [r for r in recs2 if r["rule"] == "连续__consec_dn5_fade" and r["horizon"] == 1][0]
    assert not t2["validated"]


def test_purge_uses_original_bars_not_valid_row_count():
    bars = np.array([0, 1, 2, 50, 51, 52, 100, 101])
    blocks = _fold_blocks(np.arange(8), 2, gap=3, bar_idx=bars)
    assert blocks[0].tolist() == [0, 1, 2]
    assert bars[blocks[0][-1]] + 3 < bars[blocks[1][0]]


def test_daily_score_handles_long_short_and_horizon_units():
    from super_trader.personalize import _daily_score
    assert _daily_score(np.array([1, 1]), np.array([.06, .04]), .01, 5) == pytest.approx(.008)
    assert _daily_score(np.array([-1, -1]), np.array([-.02, -.04]), .01, 5) == pytest.approx(.008)
    assert _daily_score(np.array([1, -1]), np.array([.05, -.03]), .01, 5) == pytest.approx(.008)


def test_horizon_choice_cannot_see_holdout_labels_or_returns():
    p = _prep(_planted_df())
    first = evaluate_symbol_personal(p, EvalConfig())
    for h in EvalConfig().horizons:
        idx, f, fo, up = p.masks(h)
        f, up = f.copy(), up.copy()
        m = idx >= p.split_idx
        f[m] = -f[m] * 100
        up[m] = ~up[m]
        p._mask_cache[h] = idx, f, fo, up
    second = evaluate_symbol_personal(p, EvalConfig())
    a = {(r['rule'], r['horizon']): (r['horizon_chosen'], r['daily_train'], r['p_train']) for r in first}
    b = {(r['rule'], r['horizon']): (r['horizon_chosen'], r['daily_train'], r['p_train']) for r in second}
    assert a.keys() == b.keys()
    for key in a:
        assert a[key][0] == b[key][0]
        np.testing.assert_allclose(a[key][1:], b[key][1:], equal_nan=True)
    assert all(sum(r['horizon_chosen'] for r in first if r['rule'] == rule) <= 1
               for rule in {r['rule'] for r in first})


def test_long_horizon_can_win_training_choice_despite_lower_power(monkeypatch):
    from types import SimpleNamespace
    import super_trader.personalize as module
    p = _prep(_planted_df())
    sig = (np.arange(p.n) % 2 == 0).astype(np.int8)
    p.sigs = {'test_rule': sig}
    monkeypatch.setattr(module, 'RULES', [SimpleNamespace(key='test_rule', family='test', desc='test')])
    for h, effect in [(1, .001), (20, .1)]:
        idx = np.arange(120, p.n - 20)
        fwd = np.where(sig[idx] > 0, effect, -.001)
        p._mask_cache[h] = idx, fwd, fwd.copy(), fwd > 0
    records = evaluate_symbol_personal(p, EvalConfig(horizons=[1, 20]))
    chosen = [r['horizon'] for r in records if r['horizon_chosen']]
    assert chosen == [20]


def test_primary_rule_ranking_uses_training_not_holdout():
    df = pd.DataFrame([
        {'code': 'A', 'rule': 'train_best', 'daily_train': .02, 'edge_hold': .01},
        {'code': 'A', 'rule': 'holdout_best', 'daily_train': .001, 'edge_hold': .5},
    ])
    assert best_per_symbol(df).iloc[0]['rule'] == 'train_best'


def test_personal_report_exposes_power_and_research_limitations(tmp_path):
    from super_trader.personal_report import build_personal_report
    payload = {'horizon_diagnostics': [
        {'horizon': 20, 'tests': 10, 'chosen_train': 3, 'validated': 0,
         'median_n_hold': 200, 'median_n_eff': 10, 'median_mdes': .26, 'power_limited': 8}],
        'totals': {}, 'gate': {}}
    paths = build_personal_report(payload, tmp_path)
    text = paths[0].read_text(encoding='utf-8')
    assert 'H20' in text and '26.00' in text
    assert '不是严格假发现率' in text
    assert '80%功效' in text


def test_gain_columns_semantics():
    df = pd.DataFrame([
        {"code": "A", "rule": "r1", "horizon": 1, "edge_hold": 0.10, "validated": True},
        {"code": "B", "rule": "r1", "horizon": 1, "edge_hold": 0.00, "validated": True},
        {"code": "A", "rule": "r2", "horizon": 1, "edge_hold": 0.04, "validated": True},
        {"code": "A", "rule": "g1", "horizon": 1, "edge_hold": 0.06, "validated": False},
        {"code": "B", "rule": "g1", "horizon": 1, "edge_hold": 0.02, "validated": False},
    ])
    out = add_gain_columns(df, accepted=[("g1", 1)])
    # r1 的跨标的平均优势 = 0.05
    assert out.loc[out["code"] == "A", "rule_mean_edge"].iloc[0] == pytest.approx(0.05)
    assert out.loc[out["code"] == "A", "specificity"].iloc[0] == pytest.approx(0.05)
    # A 的通用规则最好 = 0.06（g1），所以 r1 的个性化增益 = 0.10 - 0.06
    assert out.loc[(out["code"] == "A") & (out["rule"] == "r1"), "gain"].iloc[0] == pytest.approx(0.04)

    best = best_per_symbol(out[out["validated"]])
    assert set(best["code"]) == {"A", "B"}
    assert best.set_index("code").loc["A", "rule"] == "r1"
