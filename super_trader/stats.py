"""统计层：只做「可解释、可复核」的小工具，不引入黑箱。

关键约定
--------
1) **重叠窗口修正**：h 日前瞻收益在相邻信号之间重叠 h-1 天，n 个信号的有效独立
   样本量约等于 n/h。直接把 1000 个重叠样本当独立样本算 p 值，会把显著性放大
   数倍——这是「回测里到处都是神规则」的最常见原因。本模块默认按 n_eff=n/h 检验，
   `n_eff` 参数可关闭该修正（仅在测试对照时使用）。
2) **基准涨率**：A 股长期上涨概率并非 50%（指数约 52~54%），因此检验的零假设是
   「规则命中率 = 该标的该周期的无条件涨率」，而不是「= 50%」。
3) **FDR**：同时检验几百条规则，用 Benjamini-Hochberg 控制假发现率。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm

HAC_MIN_LAG = 21  # HAC 最小滞后阶数（一个月），覆盖波动聚集


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（小样本/极端比例比正态近似稳）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_n(n: int, horizon: int, enabled: bool = True) -> float:
    """重叠窗口下的有效独立样本量。"""
    if not enabled or horizon <= 1:
        return float(n)
    return max(1.0, n / float(horizon))


def binom_p_upper(acc: float, n_eff: float, p0: float) -> float:
    """单侧检验 H1: acc > p0，样本量取 n_eff（可为小数，内部取整并同步换算命中数）。"""
    if not np.isfinite(acc) or n_eff <= 0 or not (0 < p0 < 1):
        return 1.0
    ne = int(max(1, round(n_eff)))
    ke = int(round(acc * ne))
    if ke <= 0:
        return 1.0
    if ke > ne:
        ke = ne
    return float(binom.sf(ke - 1, ne, p0))


def z_edge(acc: float, n_eff: float, p0: float) -> float:
    """相对基准涨率的标准化优势（单位：标准误）。"""
    if n_eff <= 0 or not (0 < p0 < 1):
        return float("nan")
    se = math.sqrt(p0 * (1 - p0) / n_eff)
    return (acc - p0) / se if se > 0 else float("nan")


def bh_fdr(pvals, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg：返回 (是否拒绝, q 值)。NaN 视为不拒绝。"""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    qv = np.full(n, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return reject, qv
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    m = ps.size
    ranked = ps[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    qv[idx[order]] = adj
    reject[idx[order]] = adj <= q
    return reject, qv


def norm_sf(z: float) -> float:
    """标准正态右尾概率 P(Z > z)。"""
    if not np.isfinite(z):
        return 1.0
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def sign_test_p(pos: int, n: int) -> float:
    """符号检验：n 个独立单位中 pos 个为正，检验是否好于随机（单侧）。"""
    if n <= 0 or pos <= 0:
        return 1.0
    return float(binom.sf(pos - 1, n, 0.5))


def newey_west_se(x, lag: int) -> float:
    """Newey-West(Bartlett) HAC 标准误：同时处理重叠窗口与波动率聚集。

    普通 t 检验假设序列独立；前瞻收益重叠 h 天、A 股波动又成簇聚集，会把标准误低估
    一个档次（于是「到处都是神规则」）。这里用 Bartlett 核加权的 HAC 估计，滞后阶数
    取 max(h, 21)，即至少覆盖一个月度尺度上的相依性。
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    T = a.size
    if T < 3:
        return float("nan")
    xc = a - a.mean()
    var = float(xc @ xc) / T
    L = int(max(0, min(lag, T - 1)))
    for k in range(1, L + 1):
        gk = float(xc[k:] @ xc[:-k]) / T
        var += 2.0 * (1.0 - k / (L + 1.0)) * gk
    if var <= 0:
        return float("nan")
    return float(math.sqrt(var / T))


def min_detectable_edge(n_eff: float, p0: float = 0.5, alpha: float = 0.05) -> float:
    """在给定有效样本量下，双侧 5% 水平、80% 功效能检出的最小优势。

    用来回答「样本够不够」：如果可检出优势是 6 个百分点，而真实优势只有 2 个
    百分点，那么「没检出」不代表「没规律」，只代表数据不够。
    """
    if n_eff <= 0:
        return float("nan")
    se = math.sqrt(p0 * (1 - p0) / n_eff)
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(0.80)) * se)


def mean_ci(values, z: float = 1.96) -> tuple[float, float, float]:
    """均值与正态近似置信区间。"""
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(a.mean())
    if a.size == 1:
        return (m, float("nan"), float("nan"))
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    return (m, m - z * se, m + z * se)


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7):
    """对跨标的结果做 bootstrap 均值区间（不依赖正态假设）。"""
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size < 3:
        return mean_ci(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(a.mean()), float(lo), float(hi))
