"""特征层：把标准化 OHLCV 变成一个「按 bar 对齐的特征矩阵」。

所有特征在第 i 根 bar 上只使用 <= i 的信息（rolling/ewm 默认含当根，比较类特征
一律对「过去」做 shift(1)），因此第 i 根的信号可以用第 i 根收盘价成交后在
第 i+1 根开盘执行，不存在未来函数。日历类特征（月末、长假前）由交易日历推出，
不包含任何价格信息。

金额/成交量单位沿用东财原始口径（手、元），因此跨标的比较前需先做比值标准化，
本模块输出的量能特征全部是「相对自身近期均值」的比值，可直接横截面使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class FeatureSet:
    """特征字典 + 原始行情，便于评估层取用。"""

    data: dict[str, np.ndarray]
    df: pd.DataFrame
    dates: pd.DatetimeIndex

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    def __len__(self) -> int:
        return len(self.df)


# ---------------------------------------------------------------- 小工具
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 全涨（loss=0）时 RSI=100；全跌时为 0
    out = out.where(loss > 0, 100.0)
    out = out.where(gain > 0, 0.0)
    return out


def _streak(flag: pd.Series) -> pd.Series:
    """连续为 True 的长度（当根计入）。"""
    f = flag.fillna(False).astype(int)
    block = (f != f.shift()).cumsum()
    return f.groupby(block).cumsum()


def _rolling_rank(s: pd.Series, n: int) -> pd.Series:
    """当前值在过去 n 根中的分位（0~1），只用历史。"""
    return s.rolling(n).rank(pct=True)


def calendar_flags(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """交易日历特征（不含价格信息）。

    - month_end: 当月最后 2 个交易日
    - month_start: 当月前 2 个交易日
    - pre_holiday: 下一根 K 线距当根 >= 4 个自然日（长假前）
    - post_holiday: 当根距上一根 >= 4 个自然日（长假后首日）

    月末/月初用「交易所工作日历」判定，而不是「数据里的下一根 K 线」——后者在最新
    一根 K 线上会因为没有「下一根」而误判成月末，把实盘信号算错。交易所日历是事先
    公布的，不含未来价格信息。
    """
    n = len(dates)
    if n == 0:
        return {"month_end": np.zeros(0, bool), "month_start": np.zeros(0, bool),
                "pre_holiday": np.zeros(0, bool), "post_holiday": np.zeros(0, bool),
                "dow": np.zeros(0, float), "dom": np.zeros(0, float)}

    # 已走完的月份：直接按数据里该月的交易日序列取前/后 2 根（最贴合真实节假日）。
    # 只有「当前这个还没走完的月」才需要用交易所日历推断——否则最后一根会被误判成月末。
    s = pd.Series(np.arange(n, dtype=float))
    period = pd.Series(dates).dt.to_period("M")
    rank_end = s.groupby(period).rank(ascending=False, method="first").to_numpy()
    rank_start = s.groupby(period).rank(ascending=True, method="first").to_numpy()
    month_end = rank_end <= 2
    month_start = rank_start <= 2

    first_period = period.iloc[0]
    last_period = period.iloc[-1]
    # 首月可能从月中开始、末月可能还没走完，这两段的「月初/月末」必须用交易所日历判定，
    # 否则最后一根 K 线会被当成月末（实盘信号错），首月会被当成月初。
    override = ((period == last_period) | (period == first_period)).to_numpy()
    if override.any():
        start = (dates.min() - pd.offsets.MonthBegin(1)).normalize()
        end = (dates.max() + pd.offsets.MonthEnd(1)).normalize()
        bds = pd.Series(pd.bdate_range(start, end))
        bpos = pd.Series(np.arange(len(bds), dtype=float))
        gb = bpos.groupby(bds.dt.to_period("M"))
        rank_e = gb.rank(ascending=False, method="first").to_numpy()
        rank_s = gb.rank(ascending=True, method="first").to_numpy()
        lookup_e = dict(zip(bds.values, rank_e))
        lookup_s = dict(zip(bds.values, rank_s))
        vals_e = np.array([lookup_e.get(v, 99) for v in dates.values], dtype=float)
        vals_s = np.array([lookup_s.get(v, 99) for v in dates.values], dtype=float)
        month_end = np.where(override, vals_e <= 2, month_end)
        month_start = np.where(override, vals_s <= 2, month_start)

    gaps = np.zeros(n, dtype=float)
    gaps[1:] = (dates[1:] - dates[:-1]).days.values.astype(float)
    pre_holiday = np.zeros(n, dtype=bool)
    post_holiday = np.zeros(n, dtype=bool)
    pre_holiday[:-1] = gaps[1:] >= 4
    post_holiday[1:] = gaps[1:] >= 4

    return {
        "month_end": month_end,
        "month_start": month_start,
        "pre_holiday": pre_holiday,
        "post_holiday": post_holiday,
        "dow": dates.dayofweek.to_numpy().astype(float),
        "dom": dates.day.to_numpy().astype(float),
    }


# ---------------------------------------------------------------- 主函数
def build_features(df: pd.DataFrame, price_limit: float = 0.0) -> FeatureSet:
    """计算全部特征。df 需按时间升序且列齐全。"""
    df = df.reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    turn = df["turnover"].astype(float)
    prev_c = c.shift(1)

    # ---- 波动 ----
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr120 = atr14.rolling(120).mean()
    atr_pct = atr14 / atr120.replace(0, np.nan)
    atr_rank = _rolling_rank(atr_pct, 250)

    # ---- 均线 / 斜率 ----
    ema8, ema20, ema50, ema144 = _ema(c, 8), _ema(c, 20), _ema(c, 50), _ema(c, 144)
    atr_safe = atr14.replace(0, np.nan)
    s16 = (ema20 - ema20.shift(16)) / atr_safe
    s8 = (ema8 - ema8.shift(8)) / atr_safe
    s50 = (ema50 - ema50.shift(50)) / atr_safe
    dist_ema20 = (c - ema20) / atr_safe
    dist_ema50 = (c - ema50) / atr_safe

    # ---- 动量 / 超买超卖 ----
    r14 = _rsi(c, 14)
    r2 = _rsi(c, 2)
    mid20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std(ddof=0)
    pctb = (c - (mid20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    bb_width = (4 * sd20) / mid20.replace(0, np.nan)
    bb_width_rank = _rolling_rank(bb_width, 250)

    # ---- 区间位置 / 突破 ----
    lo20, hi20 = c.rolling(20).min(), c.rolling(20).max()
    lo60, hi60 = c.rolling(60).min(), c.rolling(60).max()
    lo120, hi120 = c.rolling(120).min(), c.rolling(120).max()
    pos20 = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    pos60 = (c - lo60) / (hi60 - lo60).replace(0, np.nan)
    pos120 = (c - lo120) / (hi120 - lo120).replace(0, np.nan)
    hh20 = h.rolling(20).max().shift(1)
    ll20 = l.rolling(20).min().shift(1)
    hh55 = h.rolling(55).max().shift(1)
    ll55 = l.rolling(55).min().shift(1)
    kelt_up = ema20 + 2 * atr14
    kelt_dn = ema20 - 2 * atr14

    # ---- 量能 ----
    vma20 = v.rolling(20).mean()
    vma5 = v.rolling(5).mean()
    vol_r = v / vma20.replace(0, np.nan)
    vol_r5 = v / vma5.replace(0, np.nan)
    amt_r = amt / amt.rolling(20).mean().replace(0, np.nan)
    turn_mean = turn.rolling(60).mean()
    turn_std = turn.rolling(60).std(ddof=0)
    turn_z = (turn - turn_mean) / turn_std.replace(0, np.nan)

    # ---- 形态 ----
    rng = h - l
    body = (c - o) / atr_safe
    upper = (h - pd.concat([o, c], axis=1).max(axis=1)) / atr_safe
    lower = (pd.concat([o, c], axis=1).min(axis=1) - l) / atr_safe
    range_atr = rng / atr_safe
    gap = o / prev_c.replace(0, np.nan) - 1
    gap_atr = (o - prev_c) / atr_safe
    nr7 = rng <= rng.rolling(7).min() * 1.0000001

    # ---- 连涨连跌 / 效率 ----
    up = c.diff() > 0
    dn = c.diff() < 0
    consec_up = _streak(up)
    consec_dn = _streak(dn)
    net = (c - c.shift(100)).abs()
    path = c.diff().abs().rolling(100).sum()
    er100 = net / path.replace(0, np.nan)
    z5 = (c - c.rolling(5).mean()) / c.rolling(5).std(ddof=0).replace(0, np.nan)

    ret1 = c / prev_c.replace(0, np.nan) - 1
    ret5 = c / c.shift(5).replace(0, np.nan) - 1
    ret20 = c / c.shift(20).replace(0, np.nan) - 1

    # ---- A 股专属：涨跌停 / 一字板 ----
    if price_limit > 0:
        limit_up = ret1 >= price_limit - 0.005
        limit_dn = ret1 <= -(price_limit - 0.005)
        one_word_up = limit_up & (o == c) & (h == l)
        broke_limit_up = limit_up & (c < h)  # 盘中触板未封住（用 high 近似）
    else:
        zeros = pd.Series(False, index=c.index)
        limit_up = limit_dn = one_word_up = broke_limit_up = zeros

    # ---- 日历 ----
    cal = calendar_flags(dates)

    data: dict[str, np.ndarray] = {
        # 原始
        "open": o.to_numpy(float), "high": h.to_numpy(float),
        "low": l.to_numpy(float), "close": c.to_numpy(float),
        "volume": v.to_numpy(float), "amount": amt.to_numpy(float),
        "turnover": turn.to_numpy(float),
        "ret1": ret1.to_numpy(float),
        # 波动
        "atr14": atr14.to_numpy(float), "atr_pct": atr_pct.to_numpy(float),
        "atr_rank": atr_rank.to_numpy(float),
        # 趋势
        "ema8": ema8.to_numpy(float), "ema20": ema20.to_numpy(float),
        "ema50": ema50.to_numpy(float), "ema144": ema144.to_numpy(float),
        "s16": s16.to_numpy(float), "s8": s8.to_numpy(float), "s50": s50.to_numpy(float),
        "ma_stack": ((ema8 > ema20) & (ema20 > ema50)).astype(float).to_numpy() - ((ema8 < ema20) & (ema20 < ema50)).astype(float).to_numpy(),
        "dist_ema20": dist_ema20.to_numpy(float),
        "dist_ema50": dist_ema50.to_numpy(float),
        # 动量
        "rsi14": r14.to_numpy(float), "rsi2": r2.to_numpy(float),
        "pctb": pctb.to_numpy(float), "bb_width": bb_width.to_numpy(float),
        "bb_width_rank": bb_width_rank.to_numpy(float),
        "z5": z5.to_numpy(float), "er100": er100.to_numpy(float),
        # 位置 / 突破
        "pos20": pos20.to_numpy(float), "pos60": pos60.to_numpy(float),
        "pos120": pos120.to_numpy(float),
        "hh20": hh20.to_numpy(float), "ll20": ll20.to_numpy(float),
        "hh55": hh55.to_numpy(float), "ll55": ll55.to_numpy(float),
        "kelt_up": kelt_up.to_numpy(float), "kelt_dn": kelt_dn.to_numpy(float),
        # 量能
        "vol_r": vol_r.to_numpy(float), "vol_r5": vol_r5.to_numpy(float),
        "amt_r": amt_r.to_numpy(float), "turn_z": turn_z.to_numpy(float),
        # 形态
        "body": body.to_numpy(float), "upper": upper.to_numpy(float),
        "lower": lower.to_numpy(float), "range_atr": range_atr.to_numpy(float),
        "gap": gap.to_numpy(float), "gap_atr": gap_atr.to_numpy(float),
        "nr7": nr7.to_numpy(bool),
        # 连续 / 收益
        "consec_up": consec_up.to_numpy(float), "consec_dn": consec_dn.to_numpy(float),
        "ret5": ret5.to_numpy(float), "ret20": ret20.to_numpy(float),
        # A 股约束
        "limit_up": limit_up.to_numpy(bool), "limit_dn": limit_dn.to_numpy(bool),
        "one_word_up": one_word_up.to_numpy(bool),
        "broke_limit_up": broke_limit_up.to_numpy(bool),
        "price_limit": np.full(len(df), float(price_limit)),
    }
    data.update(cal)
    return FeatureSet(data=data, df=df, dates=dates)
