from __future__ import annotations

import math

import numpy as np
import pytest

from super_trader import stats


def test_wilson_ci_brackets_point_estimate():
    lo, hi = stats.wilson_ci(55, 100)
    assert lo < 0.55 < hi
    assert 0.44 < lo and hi < 0.66


def test_wilson_ci_handles_zero_and_full():
    lo, hi = stats.wilson_ci(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = stats.wilson_ci(20, 20)
    assert lo > 0.8 and hi == 1.0


def test_effective_n_shrinks_with_horizon():
    assert stats.effective_n(1000, 1) == 1000
    assert stats.effective_n(1000, 10) == 100
    assert stats.effective_n(1000, 10, enabled=False) == 1000


def test_binom_p_upper_monotonic_in_accuracy():
    p_low = stats.binom_p_upper(0.52, 200, 0.50)
    p_high = stats.binom_p_upper(0.62, 200, 0.50)
    assert p_low > p_high
    assert p_high < 0.01


def test_binom_p_upper_is_large_when_below_base():
    # 命中率低于基准时，单侧「优于基准」的 p 值应该很大（不显著）
    assert stats.binom_p_upper(0.50, 300, 0.52) > 0.5
    assert stats.binom_p_upper(0.30, 300, 0.50) == pytest.approx(1.0)


def test_overlap_adjustment_reduces_significance():
    """同样的命中率，按 n/h 折算后 p 值应变大（更保守）。"""
    raw = stats.binom_p_upper(0.55, 500, 0.50)
    adj = stats.binom_p_upper(0.55, stats.effective_n(500, 10), 0.50)
    assert adj > raw


def test_bh_fdr_controls_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    reject, q = stats.bh_fdr(p, q=0.10)
    assert reject.sum() == 6   # p_(6)=0.06 <= 6*0.10/10 恰好在边上
    assert np.all(q[:5] <= 0.10)
    assert np.all(np.isnan(q) == False)  # noqa: E712


def test_bh_fdr_all_null_rejects_nothing_in_practice():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=100)
    reject, _ = stats.bh_fdr(p, q=0.10)
    assert reject.sum() <= 2


def test_sign_test():
    assert stats.sign_test_p(9, 10) < 0.02
    assert stats.sign_test_p(5, 10) > 0.5
    assert stats.sign_test_p(0, 10) == 1.0


def test_newey_west_se_close_to_iid_formula():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4000)
    se = stats.newey_west_se(x, 21)
    assert se == pytest.approx(x.std() / math.sqrt(x.size), rel=0.15)


def test_newey_west_se_inflates_for_autocorrelated_series():
    """正自相关序列：HAC 标准误应明显大于 sd/√n。"""
    rng = np.random.default_rng(5)
    e = rng.standard_normal(3000)
    x = np.zeros_like(e)
    for i in range(1, x.size):
        x[i] = 0.95 * x[i - 1] + e[i]          # AR(1) 0.95 —— 类似重叠窗口
    naive = x.std() / math.sqrt(x.size)
    hac = stats.newey_west_se(x, 21)
    assert hac > naive * 2


def test_norm_sf():
    assert stats.norm_sf(0.0) == pytest.approx(0.5)
    assert stats.norm_sf(1.96) == pytest.approx(0.025, abs=1e-3)
    assert stats.norm_sf(float("nan")) == pytest.approx(1.0)


def test_min_detectable_edge_decreases_with_sample():
    big = stats.min_detectable_edge(10_000)
    small = stats.min_detectable_edge(100)
    assert 0 < big < small
