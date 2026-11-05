from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_trader.data import (
    KlineStore, SymbolInfo, artifact_mask, classify_board, quality_report, suspension_mask,
)
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------- 板块/涨跌停
@pytest.mark.parametrize("code,category,board,limit", [
    ("600519", "个股", "主板", 0.10),
    ("000001", "个股", "主板", 0.10),
    ("300750", "个股", "创业板", 0.20),
    ("688981", "个股", "科创板", 0.20),
    ("830799", "个股", "北交所", 0.30),
    ("510300", "ETF", "ETF", 0.10),
    ("000300", "指数", "指数", 0.0),
])
def test_classify_board(code, category, board, limit):
    b, l = classify_board(code, category)
    assert b == board and l == pytest.approx(limit)


# ---------------------------------------------------------------- 除权识别
def test_artifact_mask_flags_ex_dividend_gap():
    df = make_ohlcv(60, seed=1)
    # 人为制造一个 -35% 的除权缺口（远超主板 ±10%）
    i = 40
    factor = 0.65
    for col in ("open", "close", "high", "low"):
        df.loc[i:, col] = df.loc[i:, col] * factor
    mask = artifact_mask(df, 0.10)
    assert mask[i]
    assert mask.sum() == 1


def test_artifact_mask_ignores_normal_limit_moves():
    df = make_ohlcv(80, seed=2)
    for i in (30, 50):
        prev = df.loc[i - 1, "close"]
        for col in ("open", "close", "high", "low"):
            df.loc[i:, col] = df.loc[i:, col] * (prev * 1.099 / df.loc[i, "col"] if col == "col" else 1)
    # 构造 ±9.9%（未超涨停）的普通大波动
    close = df["close"].to_numpy(dtype=float).copy()
    close[31] = close[30] * 1.099
    df["close"] = close
    mask = artifact_mask(df, 0.10)
    assert mask.sum() == 0


def test_artifact_mask_disabled_for_index():
    df = make_ohlcv(50, seed=3)
    df.loc[20:, "close"] *= 0.5
    assert artifact_mask(df, 0.0).sum() == 0


def test_suspension_mask_detects_long_gap():
    df = make_ohlcv(40, seed=4)
    dates = list(df["date"].iloc[:20]) + list(pd.bdate_range(df["date"].iloc[19] + pd.Timedelta(days=40),
                                                            periods=20))
    df["date"] = dates
    m = suspension_mask(df)
    assert m[20] and m.sum() == 1


# ---------------------------------------------------------------- 读取与缓存
def _write_csv(path, df):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_store_catalog_and_load(tmp_path):
    root = tmp_path / "日线"
    (root / "个股").mkdir(parents=True)
    _write_csv(root / "个股" / "600519_贵州茅台.csv", make_ohlcv(50, seed=6))
    store = KlineStore(root, tmp_path / "cache")
    cat = store.catalog()
    assert len(cat) == 1 and cat.iloc[0]["code"] == "600519"
    info = store.find("600519", "个股")
    assert info is not None and info.price_limit == pytest.approx(0.10)
    df = store.load(info)
    assert len(df) == 50 and df["date"].is_monotonic_increasing


def test_store_cache_roundtrip_and_invalidation(tmp_path):
    root = tmp_path / "日线"
    (root / "个股").mkdir(parents=True)
    src = root / "个股" / "000001_平安银行.csv"
    _write_csv(src, make_ohlcv(60, seed=8))
    store = KlineStore(root, tmp_path / "cache")
    a = store.load("000001")
    b = store.load("000001")            # 走缓存
    assert a.equals(b)
    _write_csv(src, make_ohlcv(70, seed=9))   # 改文件 → 缓存失效
    c = store.load("000001")
    assert len(c) == 70


def test_store_drops_bad_rows(tmp_path):
    root = tmp_path / "日线"
    (root / "个股").mkdir(parents=True)
    df = make_ohlcv(30, seed=10)
    df.loc[5, "close"] = np.nan
    df.loc[6, "high"] = df.loc[6, "low"] - 1      # high < low
    df.loc[7, "close"] = -3                        # 负价
    _write_csv(root / "个股" / "600000_浦发银行.csv", df)
    store = KlineStore(root, None)
    out = store.load("600000")
    assert len(out) == 27


def test_quality_report_fields(sample_df, symbol_info):
    rep = quality_report(sample_df, symbol_info)
    assert rep["bars"] == len(sample_df)
    assert rep["artifact_bars"] == 0
    assert rep["start"] < rep["end"]
