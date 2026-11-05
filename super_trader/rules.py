"""规则库：每个规则把特征映射成方向信号 +1 / 0 / -1。

信号语义（第 i 根 bar 收盘后）：
    +1 = 规则认为未来 h 根「上涨」
    -1 = 规则认为未来 h 根「下跌」
     0 = 规则不表态（中性，不计入统计）
信号只用 <= i 的信息，因此可以在第 i+1 根开盘执行（A 股 T+1，当日买入隔日才能卖）。

命名约定：`family__rule`（家族__规则名），便于跨标的汇总。
设计目标不是「好看」，而是「可解释、可复现、可证伪」——每条规则都能用一句中文说清，
都能被后续数据推翻。规则数量刻意控制在 80 条以内，避免纯数据挖掘式的过拟合。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .features import FeatureSet


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    desc: str          # 一句中文说明（会写进报告和提示词）
    direction: str     # long / short / both
    fn: Callable[[FeatureSet], np.ndarray]

    @property
    def key(self) -> str:
        return f"{self.family}__{self.name}"


def _sig(n: int, long_cond, short_cond) -> np.ndarray:
    """长/空条件 → int8 信号。条件必须是 bool 数组（NaN 比较结果为 False）。"""
    s = np.zeros(n, dtype=np.int8)
    lc = np.asarray(long_cond, dtype=bool)
    sc = np.asarray(short_cond, dtype=bool)
    s[lc] = 1
    s[sc & ~lc] = -1
    return s


RULES: list[Rule] = []


def rule(name: str, family: str, desc: str, direction: str = "both"):
    def deco(fn: Callable[[FeatureSet], np.ndarray]):
        RULES.append(Rule(name, family, desc, direction, fn))
        return fn

    return deco


# ============================================================ 趋势（顺势）
@rule("s16_03", "趋势", "EMA20 的 16 根斜率 ≥0.3ATR 看涨，≤-0.3ATR 看跌（弱趋势）")
def _(f: FeatureSet):
    s = f["s16"]
    return _sig(len(f), s >= 0.3, s <= -0.3)


@rule("s16_05", "趋势", "EMA20 的 16 根斜率 ≥0.5ATR 看涨，≤-0.5ATR 看跌（标准趋势）")
def _(f: FeatureSet):
    s = f["s16"]
    return _sig(len(f), s >= 0.5, s <= -0.5)


@rule("s16_10", "趋势", "EMA20 的 16 根斜率 ≥1.0ATR 看涨，≤-1.0ATR 看跌（强趋势）")
def _(f: FeatureSet):
    s = f["s16"]
    return _sig(len(f), s >= 1.0, s <= -1.0)


@rule("s8_05", "趋势", "EMA8 的 8 根斜率 ≥0.5ATR 顺势（更快的趋势，噪声更大）")
def _(f: FeatureSet):
    s = f["s8"]
    return _sig(len(f), s >= 0.5, s <= -0.5)


@rule("s50_05", "趋势", "EMA50 的 50 根斜率 ≥0.5ATR 顺势（慢趋势，代表中期方向）")
def _(f: FeatureSet):
    s = f["s50"]
    return _sig(len(f), s >= 0.5, s <= -0.5)


@rule("ma_stack", "趋势", "EMA8>EMA20>EMA50 多头排列看涨；反向排列看跌")
def _(f: FeatureSet):
    st = f["ma_stack"]
    return _sig(len(f), st > 0, st < 0)


@rule("above_ema144", "趋势", "收盘在 EMA144 之上看涨、之下看跌（牛熊分界，慢线过滤）")
def _(f: FeatureSet):
    atr = np.where(f["atr14"] > 0, f["atr14"], np.nan)
    d = (f["close"] - f["ema144"]) / atr
    return _sig(len(f), d > 0, d < 0)


@rule("dist_ema20_follow", "趋势", "收盘高出 EMA20 ≥1ATR 顺势看涨，低 1ATR 顺势看跌")
def _(f: FeatureSet):
    d = f["dist_ema20"]
    return _sig(len(f), d >= 1.0, d <= -1.0)


# ============================================================ 突破
@rule("donchian20", "突破", "收盘创 20 日新高看涨、新低看跌（不区分量能）")
def _(f: FeatureSet):
    c = f["close"]
    return _sig(len(f), c > f["hh20"], c < f["ll20"])


@rule("donchian55", "突破", "收盘突破 55 日高/低点（唐奇安通道，趋势起点）")
def _(f: FeatureSet):
    c = f["close"]
    return _sig(len(f), c > f["hh55"], c < f["ll55"])


@rule("keltner_break", "突破", "收盘突破 EMA20±2ATR（肯特纳通道）")
def _(f: FeatureSet):
    c = f["close"]
    return _sig(len(f), c > f["kelt_up"], c < f["kelt_dn"])


@rule("nr7_break", "突破", "近 7 根振幅最窄（NR7）后按当根实体方向跟随")
def _(f: FeatureSet):
    n7 = f["nr7"]
    return _sig(len(f), n7 & (f["close"] > f["open"]), n7 & (f["close"] < f["open"]))


@rule("new_high5", "突破", "收盘创 5 日新高顺势看涨，新低看跌（短周期动量）")
def _(f: FeatureSet):
    c = f["close"]
    hh5 = np.maximum.reduce([np.roll(f["high"], k) for k in (1, 2, 3, 4, 5)])
    ll5 = np.minimum.reduce([np.roll(f["low"], k) for k in (1, 2, 3, 4, 5)])
    s = _sig(len(f), c > hh5, c < ll5)
    s[:6] = 0
    return s


@rule("nr7_close_break", "突破", "NR7 且收盘突破前 7 日高点顺势（窄幅整理后的放量突破前身）")
def _(f: FeatureSet):
    n7 = f["nr7"]
    hh7 = np.maximum.reduce([np.roll(f["high"], k) for k in (1, 2, 3, 4, 5, 6, 7)])
    ll7 = np.minimum.reduce([np.roll(f["low"], k) for k in (1, 2, 3, 4, 5, 6, 7)])
    s = _sig(len(f), n7 & (f["close"] > hh7), n7 & (f["close"] < ll7))
    s[:8] = 0
    return s


# ============================================================ 均值回归（逆势）
@rule("rsi2_extreme", "均值回归", "RSI(2)<10 看涨、>90 看跌（短线超跌反弹/超涨回落）")
def _(f: FeatureSet):
    r = f["rsi2"]
    return _sig(len(f), r < 10, r > 90)


@rule("rsi14_extreme", "均值回归", "RSI(14)<30 看涨、>70 看跌")
def _(f: FeatureSet):
    r = f["rsi14"]
    return _sig(len(f), r < 30, r > 70)


@rule("bb_pctb_extreme", "均值回归", "布林 %B<0.05 看涨、>0.95 看跌（贴近或突破带外反向）")
def _(f: FeatureSet):
    b = f["pctb"]
    return _sig(len(f), b < 0.05, b > 0.95)


@rule("bb_pctb_soft", "均值回归", "布林 %B<0.15 看涨、>0.85 看跌（更宽的超买超卖定义）")
def _(f: FeatureSet):
    b = f["pctb"]
    return _sig(len(f), b < 0.15, b > 0.85)


@rule("far_from_ema20", "均值回归", "收盘低于 EMA20 达 2ATR 看涨、高于 2ATR 看跌（偏离过大反向）")
def _(f: FeatureSet):
    d = f["dist_ema20"]
    return _sig(len(f), d <= -2.0, d >= 2.0)


@rule("far_from_ema20_15", "均值回归", "收盘偏离 EMA20 达 1.5ATR 反向（比 2ATR 更早进场）")
def _(f: FeatureSet):
    d = f["dist_ema20"]
    return _sig(len(f), d <= -1.5, d >= 1.5)


@rule("pos60_extreme", "均值回归", "60 日区间位置<0.10 看涨、>0.90 看跌")
def _(f: FeatureSet):
    p = f["pos60"]
    return _sig(len(f), p < 0.10, p > 0.90)


@rule("pos20_extreme", "均值回归", "20 日区间位置<0.15 看涨、>0.85 看跌")
def _(f: FeatureSet):
    p = f["pos20"]
    return _sig(len(f), p < 0.15, p > 0.85)


@rule("z5_extreme", "均值回归", "收盘对 5 日均线 z 分数 ≤-2 看涨、≥2 看跌")
def _(f: FeatureSet):
    z = f["z5"]
    return _sig(len(f), z <= -2.0, z >= 2.0)


@rule("rsi2_oversold_uptrend", "均值回归", "RSI(2)<15 且收盘在 EMA50 上（上升趋势中的急跌买点）")
def _(f: FeatureSet):
    return _sig(len(f), (f["rsi2"] < 15) & (f["close"] > f["ema50"]), (f["rsi2"] > 85) & (f["close"] < f["ema50"]))


@rule("ret5_extreme", "均值回归", "5 日涨跌超 ±8% 反向（短期过度反应）")
def _(f: FeatureSet):
    r = f["ret5"]
    return _sig(len(f), r <= -0.08, r >= 0.08)


@rule("ret20_extreme", "均值回归", "20 日涨跌超 ±20% 反向（中期过度延伸）")
def _(f: FeatureSet):
    r = f["ret20"]
    return _sig(len(f), r <= -0.20, r >= 0.20)


# ============================================================ 波动率状态
@rule("trend_highvol", "波动率", "仅在 ATR 高于 120 日均值 1.2 倍（高波动）时按 EMA20 斜率顺势")
def _(f: FeatureSet):
    hi = f["atr_pct"] > 1.2
    s = f["s16"]
    return _sig(len(f), hi & (s >= 0.5), hi & (s <= -0.5))


@rule("trend_lowvol", "波动率", "仅在低波动（ATR<0.8 倍均值）时按 EMA20 斜率顺势")
def _(f: FeatureSet):
    lo = f["atr_pct"] < 0.8
    s = f["s16"]
    return _sig(len(f), lo & (s >= 0.5), lo & (s <= -0.5))


@rule("mr_lowvol", "波动率", "仅低波动时做布林 %B 反向（震荡市才做均值回归）")
def _(f: FeatureSet):
    lo = f["atr_pct"] < 0.8
    b = f["pctb"]
    return _sig(len(f), lo & (b < 0.05), lo & (b > 0.95))


@rule("squeeze_break", "波动率", "低波动 + 突破 20 日高/低点（挤压后方向选择）")
def _(f: FeatureSet):
    lo = f["atr_pct"] < 0.8
    c = f["close"]
    return _sig(len(f), lo & (c > f["hh20"]), lo & (c < f["ll20"]))


@rule("vol_expansion_follow", "波动率", "ATR 放大到 1.3 倍且当根实体 ≥0.3ATR 顺势（波动扩张跟随）")
def _(f: FeatureSet):
    hi = f["atr_pct"] > 1.3
    b = f["body"]
    return _sig(len(f), hi & (b >= 0.3), hi & (b <= -0.3))


@rule("narrow_range_break", "波动率", "布林带宽处于 250 日最低 20% 分位 + 突破 20 日极值")
def _(f: FeatureSet):
    nar = f["bb_width_rank"] < 0.2
    c = f["close"]
    return _sig(len(f), nar & (c > f["hh20"]), nar & (c < f["ll20"]))


@rule("atr_rank_extreme", "波动率", "ATR 分位<20% 看涨（低波动后往往向上）、>80% 看跌")
def _(f: FeatureSet):
    a = f["atr_rank"]
    return _sig(len(f), a < 0.2, a > 0.8)


# ============================================================ 量能
@rule("vol_spike_follow", "量能", "成交量>20 日均量 1.5 倍且实体 ≥0.2ATR，按实体方向顺势")
def _(f: FeatureSet):
    v = f["vol_r"] > 1.5
    b = f["body"]
    return _sig(len(f), v & (b >= 0.2), v & (b <= -0.2))


@rule("vol_spike_fade", "量能", "放量（>2 倍均量）大阳线后看跌，大阴线后看涨（情绪极值反向）")
def _(f: FeatureSet):
    v = f["vol_r"] > 2.0
    b = f["body"]
    return _sig(len(f), v & (b <= -0.8), v & (b >= 0.8))


@rule("lowvol_uptrend", "量能", "缩量（<20 日均量 0.7 倍）上涨 且 EMA20 斜率>0.5ATR（无量上涨延续）")
def _(f: FeatureSet):
    q = f["vol_r"] < 0.7
    s = f["s16"]
    return _sig(len(f), q & (s >= 0.5), q & (s <= -0.5))


@rule("vol_contraction_pullback", "量能", "缩量回踩：量<0.8 倍均量、收盘在 EMA50 上但低于 EMA20 看涨")
def _(f: FeatureSet):
    q = f["vol_r"] < 0.8
    c = f["close"]
    return _sig(len(f), q & (c > f["ema50"]) & (c < f["ema20"]), np.zeros(len(f), dtype=bool))


@rule("turnover_spike_fade", "量能", "换手率 z 分数>2 时反向（过热换手后的短期均值回复）")
def _(f: FeatureSet):
    z = f["turn_z"] > 2.0
    c, o = f["close"], f["open"]
    return _sig(len(f), z & (c < o), z & (c > o))


@rule("vol_price_div", "量能", "价创新高但缩量看跌、价创新低但缩量看涨（量价背离）")
def _(f: FeatureSet):
    c = f["close"]
    return _sig(len(f), (c < f["ll20"]) & (f["vol_r"] < 0.7), (c > f["hh20"]) & (f["vol_r"] < 0.7))


@rule("vol_price_confirm", "量能", "价创新高且放量看涨、价创新低且放量看跌（量价共振）")
def _(f: FeatureSet):
    c = f["close"]
    return _sig(len(f), (c > f["hh20"]) & (f["vol_r"] > 1.5), (c < f["ll20"]) & (f["vol_r"] > 1.5))


@rule("amount_ratio_follow", "量能", "成交额>20 日均额 2 倍且收阳看涨、收阴看跌")
def _(f: FeatureSet):
    a = f["amt_r"] > 2.0
    c, o = f["close"], f["open"]
    return _sig(len(f), a & (c > o), a & (c < o))


@rule("er_low_trend", "量能", "趋势明确（|斜率|≥0.5ATR）且效率比 ER≥0.15（走势干净）才顺势")
def _(f: FeatureSet):
    s = f["s16"]
    er = f["er100"]
    return _sig(len(f), (s >= 0.5) & (er >= 0.15), (s <= -0.5) & (er >= 0.15))


# ============================================================ 形态
@rule("bar_reversal", "形态", "阴转阳看涨、阳转阴看跌（当根实体方向反转）")
def _(f: FeatureSet):
    c, o = f["close"], f["open"]
    up = c > o
    prev_up = np.roll(up, 1)
    s = _sig(len(f), up & ~prev_up, (~up) & prev_up)
    s[0] = 0
    return s


@rule("outside_bar_follow", "形态", "外包线（高更高、低更低）按实体方向跟随")
def _(f: FeatureSet):
    h, l = f["high"], f["low"]
    ph, pl = np.roll(h, 1), np.roll(l, 1)
    out = (h > ph) & (l < pl)
    c, o = f["close"], f["open"]
    s = _sig(len(f), out & (c > o), out & (c < o))
    s[0] = 0
    return s


@rule("pin_bar_lower", "形态", "长下影（下影≥1.5ATR 且≥2 倍实体）看涨；长上影看跌")
def _(f: FeatureSet):
    lo, up, b = f["lower"], f["upper"], np.abs(f["body"])
    return _sig(len(f), (lo >= 1.5) & (lo >= 2 * b), (up >= 1.5) & (up >= 2 * b))


@rule("wide_range_follow", "形态", "大实体（振幅≥2ATR）按实体方向跟随")
def _(f: FeatureSet):
    r = f["range_atr"] >= 2.0
    return _sig(len(f), r & (f["body"] >= 0.5), r & (f["body"] <= -0.5))


@rule("doji_after_trend", "形态", "强趋势中出现十字星（实体<0.15ATR）反向（趋势停顿）")
def _(f: FeatureSet):
    d = np.abs(f["body"]) < 0.15
    s = f["s16"]
    return _sig(len(f), d & (s <= -1.0), d & (s >= 1.0))


@rule("close_high_low_position", "形态", "收盘位于当根振幅上 20%（收在高位）看涨、下 20% 看跌")
def _(f: FeatureSet):
    h, l, c = f["high"], f["low"], f["close"]
    rng = np.where((h - l) > 0, h - l, np.nan)
    p = (c - l) / rng
    return _sig(len(f), p >= 0.8, p <= 0.2)


# ============================================================ 缺口
@rule("gap_follow", "缺口", "跳空高开>0.5ATR 看涨、低开>0.5ATR 看跌（缺口不回补）")
def _(f: FeatureSet):
    g = f["gap_atr"]
    return _sig(len(f), g > 0.5, g < -0.5)


@rule("gap_fill", "缺口", "跳空高开>0.5ATR 看跌、低开>0.5ATR 看涨（缺口回补）")
def _(f: FeatureSet):
    g = f["gap_atr"]
    return _sig(len(f), g < -0.5, g > 0.5)


@rule("gap_down_reversal", "缺口", "低开>1ATR 但当日收阳（低开高走）看涨")
def _(f: FeatureSet):
    g = f["gap_atr"] < -1.0
    return _sig(len(f), g & (f["close"] > f["open"]), np.zeros(len(f), dtype=bool))


@rule("gap_up_fade", "缺口", "高开>1ATR 且当日收阴（高开低走）看跌")
def _(f: FeatureSet):
    g = f["gap_atr"] > 1.0
    return _sig(len(f), np.zeros(len(f), dtype=bool), g & (f["close"] < f["open"]))


# ============================================================ 连续
@rule("consec_up3_fade", "连续", "连涨≥3 根看跌（短线过热）")
def _(f: FeatureSet):
    return _sig(len(f), np.zeros(len(f), dtype=bool), f["consec_up"] >= 3)


@rule("consec_dn3_fade", "连续", "连跌≥3 根看涨（短线超跌）")
def _(f: FeatureSet):
    return _sig(len(f), f["consec_dn"] >= 3, np.zeros(len(f), dtype=bool))


@rule("consec_up3_follow", "连续", "连涨≥3 根顺势看涨（强势延续）")
def _(f: FeatureSet):
    return _sig(len(f), f["consec_up"] >= 3, np.zeros(len(f), dtype=bool))


@rule("consec_dn3_follow", "连续", "连跌≥3 根顺势看跌（下跌延续）")
def _(f: FeatureSet):
    return _sig(len(f), np.zeros(len(f), dtype=bool), f["consec_dn"] >= 3)


@rule("consec_up5_fade", "连续", "连涨≥5 根看跌（极端过热）")
def _(f: FeatureSet):
    return _sig(len(f), np.zeros(len(f), dtype=bool), f["consec_up"] >= 5)


@rule("consec_dn5_fade", "连续", "连跌≥5 根看涨（极端超跌）")
def _(f: FeatureSet):
    return _sig(len(f), f["consec_dn"] >= 5, np.zeros(len(f), dtype=bool))


@rule("consec_dn_in_uptrend", "连续", "上升趋势（收盘>EMA50 且斜率>0）中连跌 2 根看涨（趋势内回调买点）")
def _(f: FeatureSet):
    ok = (f["close"] > f["ema50"]) & (f["s16"] > 0)
    return _sig(len(f), ok & (f["consec_dn"] >= 2), np.zeros(len(f), dtype=bool))


# ============================================================ 日历
@rule("monday_trend", "日历", "周一按 EMA20 斜率顺势（周末信息消化后开盘方向）")
def _(f: FeatureSet):
    mon = f["dow"] == 0
    s = f["s16"]
    return _sig(len(f), mon & (s >= 0.5), mon & (s <= -0.5))


@rule("friday_trend", "日历", "周五按 EMA20 斜率顺势（周线收官效应）")
def _(f: FeatureSet):
    fri = f["dow"] == 4
    s = f["s16"]
    return _sig(len(f), fri & (s >= 0.5), fri & (s <= -0.5))


@rule("month_start_long", "日历", "月初前 2 个交易日看涨（月初资金面效应）")
def _(f: FeatureSet):
    m = f["month_start"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("month_end_long", "日历", "月末最后 2 个交易日看涨（月末做净值效应）")
def _(f: FeatureSet):
    m = f["month_end"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("pre_holiday_long", "日历", "长假前的最后一个交易日看涨（节前缩量上涨）")
def _(f: FeatureSet):
    m = f["pre_holiday"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("post_holiday_follow", "日历", "长假后首日按 EMA20 斜率顺势")
def _(f: FeatureSet):
    m = f["post_holiday"]
    s = f["s16"]
    return _sig(len(f), m & (s >= 0.5), m & (s <= -0.5))


# ============================================================ 涨跌停（A 股专属）
@rule("limit_up_follow", "涨跌停", "当日涨停（收盘涨幅≈涨停板）看涨（次日惯性）")
def _(f: FeatureSet):
    m = f["limit_up"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("limit_up_fade", "涨跌停", "当日涨停看跌（次日冲高回落）")
def _(f: FeatureSet):
    m = f["limit_up"]
    return _sig(len(f), np.zeros(len(f), dtype=bool), m)


@rule("limit_dn_fade", "涨跌停", "当日跌停看涨（跌停后的超跌反弹）")
def _(f: FeatureSet):
    m = f["limit_dn"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("limit_dn_follow", "涨跌停", "当日跌停看跌（跌停后继续下跌）")
def _(f: FeatureSet):
    m = f["limit_dn"]
    return _sig(len(f), np.zeros(len(f), dtype=bool), m)


@rule("one_word_up_follow", "涨跌停", "一字涨停（开=收=高=低）看涨（封板最强形态）")
def _(f: FeatureSet):
    m = f["one_word_up"]
    return _sig(len(f), m, np.zeros(len(f), dtype=bool))


@rule("broke_limit_up_fade", "涨跌停", "盘中触及涨停但收盘回落（炸板）看跌")
def _(f: FeatureSet):
    m = f["broke_limit_up"]
    return _sig(len(f), np.zeros(len(f), dtype=bool), m)


# ============================================================ 组合
@rule("trend_plus_vol", "组合", "EMA20 斜率≥0.5ATR 且放量>1.2 倍顺势（趋势+量能确认）")
def _(f: FeatureSet):
    s = f["s16"]
    v = f["vol_r"] > 1.2
    return _sig(len(f), v & (s >= 0.5), v & (s <= -0.5))


@rule("trend_plus_pos", "组合", "EMA20 斜率≥0.5ATR 且位于 20 日区间上半部看涨（趋势+位置同向）")
def _(f: FeatureSet):
    s = f["s16"]
    return _sig(len(f), (s >= 0.5) & (f["pos20"] > 0.5), (s <= -0.5) & (f["pos20"] < 0.5))


@rule("trend_pullback_ema20", "组合", "上升趋势中回踩 EMA20（|偏离|<0.5ATR）且当根收阳看涨")
def _(f: FeatureSet):
    ok = (f["s16"] >= 0.5) & (np.abs(f["dist_ema20"]) < 0.5) & (f["close"] > f["open"])
    return _sig(len(f), ok, np.zeros(len(f), dtype=bool))


@rule("trend_pullback_ema20_dn", "组合", "下降趋势中反抽 EMA20 且当根收阴看跌")
def _(f: FeatureSet):
    ok = (f["s16"] <= -0.5) & (np.abs(f["dist_ema20"]) < 0.5) & (f["close"] < f["open"])
    return _sig(len(f), np.zeros(len(f), dtype=bool), ok)


@rule("breakout_near_level", "组合", "突破 20 日高点但仅高于高点 0.5ATR 内（刚突破，追高风险小）")
def _(f: FeatureSet):
    c = f["close"]
    atr = np.where(f["atr14"] > 0, f["atr14"], np.nan)
    near = c > f["hh20"]
    near = near & (c <= f["hh20"] + 0.5 * atr)
    dn = c < f["ll20"]
    dn = dn & (c >= f["ll20"] - 0.5 * atr)
    return _sig(len(f), near, dn)


@rule("mr_in_uptrend", "组合", "收盘在 EMA50 上但布林 %B<0.1 看涨（上升趋势中的超卖）")
def _(f: FeatureSet):
    ok = (f["close"] > f["ema50"]) & (f["pctb"] < 0.10)
    return _sig(len(f), ok, np.zeros(len(f), dtype=bool))


@rule("trend_weak_highvol_break", "组合", "高波动 + 突破 55 日高点看涨（趋势启动的经典组合）")
def _(f: FeatureSet):
    hi = f["atr_pct"] > 1.0
    c = f["close"]
    return _sig(len(f), hi & (c > f["hh55"]), hi & (c < f["ll55"]))


# ============================================================ 对外接口
RULE_BY_KEY: dict[str, Rule] = {r.key: r for r in RULES}


def rule_signals(f: FeatureSet) -> dict[str, np.ndarray]:
    """跑全部规则，返回 {规则key: int8 信号数组}。"""
    out: dict[str, np.ndarray] = {}
    n = len(f)
    for r in RULES:
        sig = np.asarray(r.fn(f), dtype=np.int8)
        if sig.shape[0] != n:
            raise ValueError(f"规则 {r.key} 输出长度 {sig.shape[0]} != {n}")
        sig[~np.isfinite(f["close"])] = 0
        out[r.key] = sig
    return out


def signal_frame(f: FeatureSet) -> "np.ndarray":
    """(n_bars, n_rules) 的信号矩阵，列顺序与 RULES 一致。"""
    sigs = rule_signals(f)
    return np.column_stack([sigs[r.key] for r in RULES]).astype(np.int8)
