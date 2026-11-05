"""评估层：把「规则信号 + 未来收益」变成一张可检验的证据表。

两层证据
--------
1) **标的级**：每个 (标的 × 规则 × 周期) 的命中率、优势、p 值、训练/留出/分阶段。
2) **市场级**（`MarketPanel`）：把同一天所有标的的信号横向聚合，得到「当天的市场级
   方向优势」时间序列，再按 n_eff=T/h 做检验。A 股各标的日收益相关性很高（同涨同跌），
   把 62 个标的当 62 次独立实验会严重夸大显著性；市场级检验把「同日横截面」先平均掉，
   只剩时间维度的独立性，这才是「这是不是一条市场规律」的正确问法。

口径（写进报告，接受质疑）
--------------------------
* 信号在第 i 根收盘后产生，前瞻收益 = close[i+h]/close[i]-1（纯预测口径）。
* 另给「可执行口径」= close[i+h]/open[i+1]-1：A 股 T+1，最早只能次日开盘成交。
* 命中判定只看方向，收平（收益恰好为 0）不计入。
* 基准不是 50%，而是同一窗口的无条件涨率。
* 除权/复牌等价格重塑 bar 落在前瞻窗口内的样本一律剔除。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from . import stats
from .data import SymbolInfo, artifact_mask
from .features import FeatureSet, build_features
from .rules import RULES, rule_signals


@dataclass
class EvalConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    warmup_bars: int = 120
    train_fraction: float = 0.6
    min_train_samples: int = 60
    min_holdout_samples: int = 30
    train_p_max: float = 0.10
    holdout_p_max: float = 0.05
    fdr_q: float = 0.10
    overlap_adjust: bool = True
    eval_start: str = "2005-01-01"
    era_starts: tuple[str, ...] = (
        "2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01", "2025-01-01",
    )
    min_era_samples: int = 20
    min_market_signals: int = 5     # 当天至少几个标的触发才算一个市场观测


# ------------------------------------------------------------------ 前瞻收益
def forward_returns(close: np.ndarray, open_: np.ndarray, horizon: int, n: int):
    """返回 (close→close 收益, next_open→close 收益)，尾部不足 h 的位置为 NaN。"""
    fwd_cc = np.full(n, np.nan)
    fwd_oc = np.full(n, np.nan)
    if n <= horizon:
        return fwd_cc, fwd_oc
    c0 = close[: n - horizon]
    c1 = close[horizon:]
    fwd_cc[: n - horizon] = c1 / c0 - 1.0
    nxt_open = open_[1 : n - horizon + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd_oc[: n - horizon] = c1 / nxt_open - 1.0
    fwd_oc[~np.isfinite(fwd_oc)] = np.nan
    return fwd_cc, fwd_oc


def _era_labels(dates: pd.DatetimeIndex, era_starts: tuple[str, ...]) -> np.ndarray:
    starts = pd.to_datetime(list(era_starts))
    lab = np.full(len(dates), -1, dtype=int)
    for i, s in enumerate(starts):
        lab[dates >= s] = i
    return lab


# ------------------------------------------------------------------ 预处理
@dataclass
class PreparedSymbol:
    info: SymbolInfo
    df: pd.DataFrame
    fs: FeatureSet
    sigs: dict[str, np.ndarray]
    dates: pd.DatetimeIndex
    art_cum: np.ndarray
    art_strict: np.ndarray
    era_lab: np.ndarray
    start_idx: int
    split_idx: int
    n: int
    _mask_cache: dict[int, tuple] = field(default_factory=dict, repr=False)

    def masks(self, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (有效样本下标, close→close 收益, next_open→close 收益, 是否上涨)。"""
        if h in self._mask_cache:
            return self._mask_cache[h]
        n = self.n
        fwd_cc, fwd_oc = forward_returns(self.fs["close"], self.fs["open"], h, n)
        valid = np.zeros(n, dtype=bool)
        valid[self.start_idx: max(self.start_idx, n - h)] = True
        hi_idx = np.minimum(np.arange(n) + h, n - 1)
        art_in_window = (self.art_cum[hi_idx + 1] - self.art_cum[np.arange(n) + 1]) > 0
        valid &= ~art_in_window
        valid &= ~self.art_strict          # 信号 bar 本身是除权/复牌 bar 时特征已失真，剔除
        valid &= np.isfinite(fwd_cc) & np.isfinite(fwd_oc)
        valid &= fwd_cc != 0.0
        idx = np.where(valid)[0]
        out = (idx, fwd_cc[idx], fwd_oc[idx], fwd_cc[idx] > 0)
        self._mask_cache[h] = out
        return out


def prepare_symbol(df: pd.DataFrame, info: SymbolInfo, cfg: EvalConfig) -> PreparedSymbol | None:
    n = len(df)
    if n < cfg.warmup_bars + 200:
        return None
    fs = build_features(df, price_limit=info.price_limit)
    sigs = rule_signals(fs)
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    art = artifact_mask(df, info.price_limit)
    art_cum = np.concatenate([[0], np.cumsum(art.astype(np.int64))])
    start_ts = pd.Timestamp(cfg.eval_start).to_datetime64()
    start_idx = max(cfg.warmup_bars, int(np.searchsorted(dates.values, start_ts)))
    total = n - start_idx
    if total < 300:
        return None
    split_idx = start_idx + int(total * cfg.train_fraction)
    return PreparedSymbol(
        info=info, df=df, fs=fs, sigs=sigs, dates=dates, art_cum=art_cum,
        art_strict=art, era_lab=_era_labels(dates, cfg.era_starts), start_idx=start_idx,
        split_idx=split_idx, n=n,
    )


# ------------------------------------------------------------------ 单标的
def _sub_stats(sig: np.ndarray, fwd: np.ndarray, up: np.ndarray, base_up: float,
               horizon: int, overlap: bool) -> dict:
    n = int(sig.size)
    if n == 0:
        return {"n": 0, "hit": 0, "acc": float("nan"), "base": float("nan"),
                "edge": float("nan"), "p": 1.0, "n_eff": 0.0,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_long": 0, "n_short": 0, "acc_long": float("nan"),
                "acc_short": float("nan")}
    long_m = sig > 0
    short_m = sig < 0
    n_l, n_s = int(long_m.sum()), int(short_m.sum())
    hit = int(up[long_m].sum() + (~up[short_m]).sum())
    acc = hit / n
    p0 = (n_l * base_up + n_s * (1.0 - base_up)) / n
    ne = stats.effective_n(n, horizon, overlap)
    lo, hi = stats.wilson_ci(hit, n)
    return {
        "n": n, "hit": hit, "acc": acc, "base": p0, "edge": acc - p0,
        "p": stats.binom_p_upper(acc, ne, p0), "n_eff": ne,
        "ci_lo": lo, "ci_hi": hi,
        "n_long": n_l, "n_short": n_s,
        "acc_long": float(up[long_m].mean()) if n_l else float("nan"),
        "acc_short": float((~up[short_m]).mean()) if n_s else float("nan"),
    }


def evaluate_prepared(p: PreparedSymbol, cfg: EvalConfig,
                      label_transform=None) -> tuple[list[dict], dict]:
    """评估单标的。

    label_transform(up: np.ndarray) -> np.ndarray 是安慰剂检验用的钩子：把「是否上涨」
    标签整体置换后重跑同一套流程，用来看这套筛选在纯噪声上能造出多少「规律」。
    """
    info, n, sigs = p.info, p.n, p.sigs
    records: list[dict] = []
    meta = {
        "code": info.code, "name": info.name, "category": info.category,
        "board": info.board, "bars": int(n),
        "eval_bars": int(n - p.start_idx), "eval_start": str(p.dates[p.start_idx].date()),
        "split_date": str(p.dates[min(p.split_idx, n - 1)].date()),
        "end": str(p.dates[-1].date()),
        "artifact_bars": int(p.art_cum[-1]),
        "horizons": {},
    }

    for h in cfg.horizons:
        idx, f_cc, f_oc, up = p.masks(h)
        if idx.size < 100:
            continue
        if label_transform is not None:
            up = np.asarray(label_transform(up.copy()), dtype=bool)
        base_up = float(up.mean())
        oc_up = f_oc > 0
        base_up_oc = float(oc_up.mean())
        is_train = idx < p.split_idx
        era_vals = p.era_lab[idx]
        meta["horizons"][str(h)] = {
            "samples": int(idx.size),
            "train_samples": int(is_train.sum()),
            "holdout_samples": int((~is_train).sum()),
            "base_up": base_up,
            "base_up_exec": base_up_oc,
            "mean_ret": float(f_cc.mean()),
            "mean_ret_exec": float(f_oc.mean()),
            "min_detectable_edge": stats.min_detectable_edge(
                stats.effective_n(int(idx.size), h, cfg.overlap_adjust), base_up),
        }

        for r in RULES:
            s_all = sigs[r.key][idx]
            trig = s_all != 0
            if not trig.any():
                continue
            st = _sub_stats(s_all[trig], f_cc[trig], up[trig], base_up, h, cfg.overlap_adjust)

            tr_m = trig & is_train
            ho_m = trig & (~is_train)
            tr = _sub_stats(s_all[tr_m], f_cc[tr_m], up[tr_m], base_up, h, cfg.overlap_adjust)
            ho = _sub_stats(s_all[ho_m], f_cc[ho_m], up[ho_m], base_up, h, cfg.overlap_adjust)

            eras_pos = eras_tot = 0
            for e in np.unique(era_vals):
                em = trig & (era_vals == e)
                ne = int(em.sum())
                if ne < cfg.min_era_samples:
                    continue
                eras_tot += 1
                long_e = s_all[em] > 0
                hit_e = (int((long_e & up[em]).sum()) + int((~long_e & ~up[em]).sum())) / ne
                p0_e = (int(long_e.sum()) * base_up + int((~long_e).sum()) * (1 - base_up)) / ne
                if hit_e > p0_e:
                    eras_pos += 1

            ex = _sub_stats(s_all[trig], f_oc[trig], oc_up[trig], base_up_oc, h, cfg.overlap_adjust)

            long_m = trig & (s_all > 0)
            short_m = trig & (s_all < 0)
            records.append({
                "code": info.code, "name": info.name, "category": info.category,
                "board": info.board, "rule": r.key, "family": r.family,
                "desc": r.desc, "horizon": h,
                "n": st["n"], "hit": st["hit"], "acc": st["acc"],
                "base": st["base"], "edge": st["edge"], "p": st["p"], "n_eff": st["n_eff"],
                "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                "n_long": st["n_long"], "n_short": st["n_short"],
                "acc_long": st["acc_long"], "acc_short": st["acc_short"],
                "base_up": base_up, "trigger_rate": st["n"] / float(idx.size),
                "n_train": tr["n"], "acc_train": tr["acc"], "edge_train": tr["edge"],
                "p_train": tr["p"], "train_base": tr["base"],
                "n_hold": ho["n"], "acc_hold": ho["acc"], "edge_hold": ho["edge"],
                "p_hold": ho["p"], "hold_base": ho["base"],
                "eras_total": eras_tot, "eras_positive": eras_pos,
                "acc_exec": ex["acc"], "edge_exec": ex["edge"], "p_exec": ex["p"],
                "mean_ret_long": float(f_cc[long_m].mean()) if long_m.any() else float("nan"),
                "mean_ret_short": float(f_cc[short_m].mean()) if short_m.any() else float("nan"),
                "uncond_mean_ret": float(f_cc.mean()),
            })

    if records:
        rej, qv = stats.bh_fdr([r["p"] for r in records], cfg.fdr_q)
        for r, ok, q in zip(records, rej, qv):
            r["q_value"] = float(q) if np.isfinite(q) else float("nan")
            r["fdr_pass"] = bool(ok)
    return records, meta


def evaluate_symbol(df: pd.DataFrame, info: SymbolInfo,
                    cfg: EvalConfig | None = None) -> tuple[list[dict], dict]:
    cfg = cfg or EvalConfig()
    p = prepare_symbol(df, info, cfg)
    if p is None:
        return [], {"code": info.code, "name": info.name, "category": info.category,
                    "skipped": "too_few_bars", "bars": len(df)}
    return evaluate_prepared(p, cfg)


# ------------------------------------------------------------------ 市场级检验
class MarketPanel:
    """把同日横截面聚合成「市场级方向优势」的时间序列，一个交易日一个观测。

    对每个 (规则, 周期) 与每个交易日 t（仅留出段）：

        acc(t)   = 当日全部触发信号的方向命中率
        base(t)  = Σ_触发标的 各自的无条件命中率 / 触发数（多空分别取 p / 1-p）
        edge(t)  = acc(t) − base(t)

    然后对 edge(t) 的均值做检验，标准误用 Newey-West HAC（滞后 max(h,21)）。

    为什么这样算：
    * 如果把 n 个标的 × T 天当成 n·T 个独立样本，A 股同涨同跌会把显著性放大一个数量级。
      这里先把同一天压成一个观测，横截面相关性自然消失，独立性的单位回到「时间」。
    * 基准用每个标的自己的无条件命中率，而不是全局 50%——否则「规则偏爱创业板」这种
      标的构成差异会被误读成预测能力。
    * HAC 同时处理前瞻窗口重叠与波动率聚集，比 sd/√n 更接近真实抽样误差。
    """

    def __init__(self, calendar: pd.DatetimeIndex, horizons: Iterable[int]):
        self.calendar = pd.DatetimeIndex(calendar)
        self.D = len(self.calendar)
        self.horizons = list(horizons)
        self.rules = [r.key for r in RULES]
        R = len(self.rules)
        self.nL: dict[int, np.ndarray] = {}
        self.hitL: dict[int, np.ndarray] = {}
        self.nS: dict[int, np.ndarray] = {}
        self.hitS: dict[int, np.ndarray] = {}
        self.expL: dict[int, np.ndarray] = {}   # 触发标的各自无条件命中率之和（多头侧）
        self.expS: dict[int, np.ndarray] = {}
        self.base_up: dict[int, np.ndarray] = {}
        self.base_n: dict[int, np.ndarray] = {}
        for h in self.horizons:
            self.nL[h] = np.zeros((R, self.D), dtype=np.int32)
            self.hitL[h] = np.zeros((R, self.D), dtype=np.int32)
            self.nS[h] = np.zeros((R, self.D), dtype=np.int32)
            self.hitS[h] = np.zeros((R, self.D), dtype=np.int32)
            self.expL[h] = np.zeros((R, self.D), dtype=np.float32)
            self.expS[h] = np.zeros((R, self.D), dtype=np.float32)
            self.base_up[h] = np.zeros(self.D, dtype=np.int32)
            self.base_n[h] = np.zeros(self.D, dtype=np.int32)

    def add(self, p: PreparedSymbol, cfg: EvalConfig) -> None:
        """把单标的的留出段信号/命中聚合进面板（每标的只累加一次）。"""
        pos_all = np.searchsorted(self.calendar.values, p.dates.values)
        for h in cfg.horizons:
            if h not in self.nL:
                continue
            idx, _f_cc, _f_oc, up = p.masks(h)
            if idx.size == 0:
                continue
            keep = idx >= p.split_idx          # 只用留出段，避免与挑选段重叠
            sel = idx[keep]
            if sel.size == 0:
                continue
            pos = pos_all[sel]
            up_sel = up[keep]
            up_i = up_sel.astype(np.int32)
            b_s = float(up_sel.mean())         # 该标的自有的无条件命中率
            self.base_n[h] += np.bincount(pos, minlength=self.D)[: self.D]
            self.base_up[h] += np.bincount(pos, weights=up_i, minlength=self.D)[: self.D].astype(np.int32)
            for ri, key in enumerate(self.rules):
                s = p.sigs[key][sel]
                if not s.any():
                    continue
                L = s > 0
                S = s < 0
                if L.any():
                    nL_loc = np.bincount(pos[L], minlength=self.D)[: self.D]
                    self.nL[h][ri] += nL_loc
                    self.hitL[h][ri] += np.bincount(
                        pos[L], weights=up_i[L].astype(np.float64), minlength=self.D
                    )[: self.D].astype(np.int32)
                    self.expL[h][ri] += (nL_loc * b_s).astype(np.float32)
                if S.any():
                    nS_loc = np.bincount(pos[S], minlength=self.D)[: self.D]
                    self.nS[h][ri] += nS_loc
                    self.hitS[h][ri] += np.bincount(
                        pos[S], weights=(~up_sel[S]).astype(np.float64), minlength=self.D
                    )[: self.D].astype(np.int32)
                    self.expS[h][ri] += (nS_loc * (1.0 - b_s)).astype(np.float32)

    def results(self, cfg: EvalConfig, base_mode: str = "per_symbol",
                lag: int | None = None, min_signals: int | None = None) -> pd.DataFrame:
        """市场级检验。base_mode: per_symbol（默认，按触发标的自身命中率）或 global（全局常数）。"""
        rows = []
        lag_default = stats.HAC_MIN_LAG
        thr = cfg.min_market_signals if min_signals is None else int(min_signals)
        for h in self.horizons:
            b_n = float(self.base_n[h].sum())
            b_up = float(self.base_up[h].sum()) / b_n if b_n > 0 else 0.5
            lag_h = int(max(h, lag if lag is not None else lag_default))
            for ri, key in enumerate(self.rules):
                nL = self.nL[h][ri].astype(np.float64)
                nS = self.nS[h][ri].astype(np.float64)
                n = nL + nS
                act = n >= thr
                if int(act.sum()) < 30:
                    continue
                hitL = self.hitL[h][ri].astype(np.float64)
                hitS = self.hitS[h][ri].astype(np.float64)
                exp = self.expL[h][ri].astype(np.float64) + self.expS[h][ri].astype(np.float64)
                acc = np.where(act, (hitL + hitS) / np.maximum(n, 1), np.nan)
                if base_mode == "global":
                    base = np.where(act, (nL * b_up + nS * (1 - b_up)) / np.maximum(n, 1), np.nan)
                else:
                    # 基准 = 当日触发标的各自无条件命中率的加权和 / 触发数。
                    # 不用当日横截面均值（会把「择时」信息减成 0），也不用全局常数
                    # （会把「规则偏爱高波动板块」这种标的构成差异误读成预测能力）。
                    base = np.where(act & (exp > 0), exp / np.maximum(n, 1), np.nan)
                edge_t = acc - base
                e = edge_t[np.isfinite(edge_t)]
                if e.size < 30:
                    continue
                mean_e = float(e.mean())
                sd = float(e.std(ddof=1)) if e.size > 1 else float("nan")
                ne = stats.effective_n(e.size, h, cfg.overlap_adjust)
                se = stats.newey_west_se(e, lag_h)
                z = mean_e / se if np.isfinite(se) and se > 0 else float("nan")
                pv = stats.norm_sf(z)
                rows.append({
                    "rule": key, "horizon": h,
                    "market_days": int(e.size),
                    "market_mean_edge": mean_e,
                    "market_sd": sd,
                    "market_se_hac": se,
                    "market_n_eff": ne,
                    "market_z": z,
                    "market_p": max(pv, 0.0),
                    "market_share_positive": float((e > 0).mean()),
                    "market_mean_acc": float(np.nanmean(acc)),
                    "market_mean_base": float(np.nanmean(base)),
                    "market_signal_days_avg": float(np.nanmean(n[act])),
                    "market_min_detectable_edge": stats.min_detectable_edge(ne, b_up),
                })
        out = pd.DataFrame(rows)
        if not out.empty:
            out["market_q"] = stats.bh_fdr(out["market_p"].to_numpy(dtype=float), cfg.fdr_q)[1]
        return out

    # ------------------------------------------------------------------ 规格曲线
    def specification_curve(self, cfg: EvalConfig,
                            base_modes: tuple[str, ...] = ("per_symbol", "global"),
                            lags: tuple[int, ...] = (5, 21, 63),
                            thresholds: tuple[int, ...] = (5, 20)) -> pd.DataFrame:
        """同一份数据、多种合理设定下重复市场级检验（规格曲线）。

        结论不应该依赖「基准怎么取、HAC 滞后取几、每天至少几个信号」这些研究者自由度。
        规则只有在一整片合理设定下都站得住，才值得写进结论。
        """
        frames = []
        for bm in base_modes:
            for lag in lags:
                for thr in thresholds:
                    df = self.results(cfg, base_mode=bm, lag=lag, min_signals=thr)
                    if df.empty:
                        continue
                    df = df[["rule", "horizon", "market_mean_edge", "market_p", "market_q", "market_days"]].copy()
                    df["variant"] = f"{bm}|lag{lag}|n{thr}"
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        long = pd.concat(frames, ignore_index=True)
        g = long.groupby(["rule", "horizon"])
        agg = pd.DataFrame({
            "variants": g.size(),
            "variant_pass": g.apply(lambda d: int((d["market_q"] <= cfg.fdr_q).sum()), include_groups=False),
            "variant_p01": g.apply(lambda d: int((d["market_p"] <= 0.01).sum()), include_groups=False),
            "variant_min_edge": g["market_mean_edge"].min(),
            "variant_max_edge": g["market_mean_edge"].max(),
            "variant_median_p": g["market_p"].median(),
        }).reset_index()
        agg["variant_pass_share"] = agg["variant_pass"] / agg["variants"].clip(lower=1)
        agg["variant_edge_min_positive"] = agg["variant_min_edge"] > 0
        return agg


# ------------------------------------------------------------------ 晋级判定
def add_selection_flags(records: list[dict], cfg: EvalConfig) -> list[dict]:
    """训练段入选 → 留出段确认 → 阶段稳定 → 晋级。"""
    for r in records:
        tr_ok = (
            r["n_train"] >= cfg.min_train_samples
            and np.isfinite(r["edge_train"]) and r["edge_train"] > 0
            and r["p_train"] <= cfg.train_p_max
        )
        ho_ok = (
            r["n_hold"] >= cfg.min_holdout_samples
            and np.isfinite(r["edge_hold"]) and r["edge_hold"] > 0
            and r["p_hold"] <= cfg.holdout_p_max
        )
        et = int(r.get("eras_total", 0))
        ep = int(r.get("eras_positive", 0))
        era_ok = True if et == 0 else ep >= max(2, int(np.ceil(et * 0.6)))
        r["pass_train"] = bool(tr_ok)
        r["pass_holdout"] = bool(ho_ok)
        r["pass_eras"] = bool(era_ok)
        r["stable_eras"] = bool(et >= 2 and ep >= max(2, int(np.ceil(et * 0.6))))
        r["promoted"] = bool(tr_ok and ho_ok and era_ok)
    return records
