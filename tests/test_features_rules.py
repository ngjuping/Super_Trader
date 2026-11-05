from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_trader.features import build_features, calendar_flags
from super_trader.rules import RULES, rule_signals, signal_frame
from tests.conftest import make_ohlcv

CUT = 600


# ---------------------------------------------------------------- 无未来函数
def test_features_are_prefix_stable():
    """截断未来数据后，前面每一根的特征必须完全不变——这是「无未来函数」的硬约束。"""
    df = make_ohlcv(900, seed=21)
    full = build_features(df)
    part = build_features(df.iloc[:CUT].reset_index(drop=True))
    numeric = [k for k, v in full.data.items()
               if isinstance(v, np.ndarray) and v.dtype != bool and v.ndim == 1]
    bad = []
    for k in numeric:
        a, b = full[k][:CUT], part[k]
        ok = np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        if not ok:
            bad.append(k)
    assert not bad, f"这些特征用到了未来数据: {bad}"


def test_rules_are_prefix_stable():
    df = make_ohlcv(900, seed=22)
    full = rule_signals(build_features(df))
    part = rule_signals(build_features(df.iloc[:CUT].reset_index(drop=True)))
    bad = [k for k in full if not np.array_equal(full[k][:CUT], part[k])]
    assert not bad, f"这些规则用到了未来数据: {bad}"


# ---------------------------------------------------------------- 基本形态
def test_signal_shape_and_domain():
    df = make_ohlcv(700, seed=23)
    fs = build_features(df)
    sigs = rule_signals(fs)
    assert len(sigs) == len(RULES)
    for k, v in sigs.items():
        assert v.shape == (len(df),), k
        assert v.dtype == np.int8, k
        assert set(np.unique(v)) <= {-1, 0, 1}, k


def test_signal_frame_matches_dict():
    df = make_ohlcv(400, seed=24)
    fs = build_features(df)
    mat = signal_frame(fs)
    sigs = rule_signals(fs)
    assert mat.shape == (len(df), len(RULES))
    for j, r in enumerate(RULES):
        assert np.array_equal(mat[:, j], sigs[r.key])


def test_every_rule_has_description():
    for r in RULES:
        assert r.desc and len(r.desc) > 6
        assert "__" in r.key and r.family
        assert r.direction in ("long", "short", "both")


def test_rising_series_triggers_trend_rules():
    n = 400
    close = np.linspace(10, 25, n)
    df = pd.DataFrame({
        "date": pd.bdate_range("2018-01-02", periods=n),
        "open": close * 1.001, "close": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full(n, 1e6), "amount": np.full(n, 1e7),
        "amplitude": np.full(n, 1.0), "pct_chg": np.full(n, 0.5),
        "change": np.full(n, 0.05), "turnover": np.full(n, 1.0),
    })
    sigs = rule_signals(build_features(df, price_limit=0.10))
    assert sigs["趋势__s16_05"][-1] == 1
    assert sigs["趋势__ma_stack"][-1] == 1
    assert sigs["趋势__above_ema144"][-1] == 1


def test_limit_rules_need_price_limit():
    df = make_ohlcv(300, seed=25)
    df.loc[250, "close"] = df.loc[249, "close"] * 1.0995   # 主板涨停
    df.loc[250, "high"] = df.loc[250, "close"]
    df.loc[250, "open"] = df.loc[250, "close"]
    df.loc[250, "low"] = df.loc[250, "close"]
    stock = rule_signals(build_features(df, price_limit=0.10))
    index = rule_signals(build_features(df, price_limit=0.0))
    assert stock["涨跌停__limit_up_follow"][250] == 1
    assert index["涨跌停__limit_up_follow"][250] == 0


# ---------------------------------------------------------------- 日历
def test_month_end_flags_last_two_trading_days():
    d = pd.DatetimeIndex(pd.to_datetime([
        "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30",
        "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05",
    ]))
    f = calendar_flags(d)
    assert f["month_end"].tolist() == [False, False, True, True, False, False, False, False]
    assert f["month_start"].tolist() == [False, False, False, False, True, True, False, False]


def test_last_bar_of_incomplete_month_is_not_month_end():
    """最新一根（月中）绝不能被判成月末——否则实盘信号会每天喊「月末买」。"""
    d = pd.DatetimeIndex(pd.bdate_range("2026-11-02", "2026-11-05"))
    f = calendar_flags(d)
    assert f["month_end"].tolist() == [False, False, False, False]


def test_holiday_flags():
    d = pd.DatetimeIndex(pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"]))
    f = calendar_flags(d)
    assert f["pre_holiday"].tolist() == [False, True, False, False]
    assert f["post_holiday"].tolist() == [False, False, True, False]


def test_features_tolerate_short_series():
    df = make_ohlcv(130, seed=26)
    fs = build_features(df)
    assert len(fs) == 130
    assert fs["ema144"][0] == pytest.approx(df["close"].iloc[0])
