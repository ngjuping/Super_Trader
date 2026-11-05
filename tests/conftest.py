"""测试用的合成行情与通用夹具（不依赖本机真实数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 800, seed: int = 7, drift: float = 0.0002,
               vol: float = 0.015, start: str = "2015-01-05") -> pd.DataFrame:
    """生成一段有自相关和波动聚集的合成日线，字段与东财 CSV 一致。"""
    rng = np.random.default_rng(seed)
    # 波动率聚集：GARCH(1,1)-ish
    sigma = np.zeros(n)
    sigma[0] = vol
    eps = rng.standard_normal(n)
    for i in range(1, n):
        sigma[i] = np.sqrt(0.85 * sigma[i - 1] ** 2 + 0.1 * (vol * 0.5) ** 2
                           + 0.05 * (sigma[i - 1] * eps[i - 1]) ** 2)
    ret = drift + sigma * eps
    close = 10.0 * np.exp(np.cumsum(ret))
    # 开盘 = 昨收 + 隔夜跳空
    gap = rng.standard_normal(n) * sigma * 0.5
    open_ = np.r_[close[0] * (1 + gap[0]), close[:-1] * (1 + gap[1:])]
    noise = np.abs(rng.standard_normal((2, n))) * sigma * 0.6
    high = np.maximum(open_, close) * (1 + noise[0])
    low = np.minimum(open_, close) * (1 - noise[1])
    volume = np.abs(rng.lognormal(12, 0.4, n))
    amount = volume * 100 * close
    prev = np.r_[np.nan, close[:-1]]
    pct = (close / prev - 1) * 100

    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": open_, "close": close, "high": high, "low": low,
        "volume": volume, "amount": amount,
        "amplitude": (high - low) / prev * 100,
        "pct_chg": pct, "change": close - prev,
        "turnover": volume / 1e6,
    })


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return make_ohlcv(900, seed=11)


@pytest.fixture()
def symbol_info(tmp_path):
    from super_trader.data import SymbolInfo

    p = tmp_path / "000001_测试.csv"
    p.write_text("date,open,close,high,low,volume,amount,amplitude,pct_chg,change,turnover\n",
                 encoding="utf-8")
    return SymbolInfo(code="000001", name="测试", category="个股", path=str(p),
                      board="主板", price_limit=0.10)
