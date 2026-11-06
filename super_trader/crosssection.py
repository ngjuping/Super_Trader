"""横截面层：相对强弱 + 行业轮动。

和前面的「单标的时间序列」不同，这一层问的是**同一时点、跨标的**的问题：

1. 相对强弱（RS）：某标的近 N 日涨幅减去全市场同一时点的中位数涨幅。
   排名靠前的下一段是不是继续跑赢？这是标准的横截面动量检验。
2. 行业轮动：把行业/主题指数当成「板块资产」，看板块动量的排名能否预测下一段排名。
3. 规模/波动分层：小市值、高波动是不是系统性地表现不同（A 股长期存在小市值效应，
   但 2017 年后反转明显，正好可以检验）。

统计口径：
* 同一天所有标的构成一个横截面，IC 用当天横截面 Spearman 秩相关，再看时间维度的均值与 t 值。
  这样「同一天 8000 只股票一起涨」不会被当成 8000 个独立样本。
* 收益一律用「相对中位数的超额」，剥离市场整体涨跌，避免把牛熊当成选股能力。
* 不做任何筛选，全部标的（满足最小样本的）都进面板。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import KlineStore, SymbolInfo


@dataclass(frozen=True)
class XSConfig:
    start: str = "2015-01-01"
    min_bars: int = 500
    horizons: tuple[int, ...] = (1, 5, 20)
    lookbacks: tuple[int, ...] = (5, 20, 60)
    min_symbols_per_day: int = 200   # 当天至少这么多标的才算一个有效横截面
    quantiles: int = 5


def load_panel(cfg: Config, config: XSConfig, cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读全部标的收盘价，拼成 date × code 面板，并返回标的信息表。"""
    if cache and cache.exists():
        data = pd.read_parquet(cache)
        meta_path = cache.with_suffix(".meta.parquet")
        meta = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()
        return data, meta

    store = KlineStore(cfg.data_path, cfg.cache_path)
    cat = store.catalog()
    frames, metas = [], []
    step = 0
    for _, row in cat.iterrows():
        step += 1
        if step % 500 == 0:
            print(f"  读取 {step}/{len(cat)}", flush=True)
        info = SymbolInfo(code=str(row["code"]), name=str(row["name"]),
                          category=str(row["category"]), path=str(row["path"]),
                          board=str(row["board"]), price_limit=float(row["price_limit"]))
        try:
            df = store.load(info)
        except Exception:
            continue
        if len(df) < config.min_bars:
            continue
        s = pd.Series(df["close"].to_numpy(float),
                      index=pd.DatetimeIndex(pd.to_datetime(df["date"])),
                      name=f"{row['category']}|{row['code']}")
        s = s[~s.index.duplicated(keep="last")]
        frames.append(s)
        metas.append({"key": s.name, "code": str(row["code"]), "name": str(row["name"]),
                      "category": str(row["category"])})
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.loc[panel.index >= pd.Timestamp(config.start)]
    meta = pd.DataFrame(metas)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache)
        meta.to_parquet(cache.with_suffix(".meta.parquet"))
    return panel, meta


def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面百分位排名（0~1），缺失保持 NaN。"""
    return df.rank(axis=1, pct=True)


def compute_xs(panel: pd.DataFrame, config: XSConfig) -> dict[str, pd.DataFrame]:
    """构造横截面特征与前瞻相对收益。"""
    rets = panel.pct_change(fill_method=None)
    out: dict[str, pd.DataFrame] = {}

    for lb in config.lookbacks:
        past = panel / panel.shift(lb) - 1.0
        # 相对强弱 = 本标的 N 日涨幅 − 当日全市场中位数
        out[f"rs{lb}"] = past.sub(past.median(axis=1), axis=0)
        out[f"rs{lb}_rank"] = _xs_rank(out[f"rs{lb}"])

    vol20 = rets.rolling(20).std(ddof=0)
    out["vol_rank"] = _xs_rank(vol20)

    for h in config.horizons:
        fwd = panel.shift(-h) / panel - 1.0
        out[f"fwd{h}"] = fwd
        out[f"fwd{h}_rel"] = fwd.sub(fwd.median(axis=1), axis=0)

    out["_rets"] = rets
    return out


def _newey_west_t(series: pd.Series, lag: int) -> float:
    x = series.dropna().to_numpy(float)
    if x.size < 30:
        return float("nan")
    mean = x.mean()
    dev = x - mean
    var = float(np.dot(dev, dev)) / x.size
    for k in range(1, min(lag, x.size - 1) + 1):
        cov = float(np.dot(dev[k:], dev[:-k])) / x.size
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    if var <= 0:
        return float("nan")
    return float(mean / np.sqrt(var / x.size))


def ic_series(feature: pd.DataFrame, target: pd.DataFrame,
              min_symbols: int) -> pd.Series:
    """逐日横截面 Spearman IC。"""
    f = feature.to_numpy(float)
    t = target.to_numpy(float)
    mask = np.isfinite(f) & np.isfinite(t)
    counts = mask.sum(axis=1)
    keep = counts >= min_symbols
    values = np.full(f.shape[0], np.nan)
    if not keep.any():
        return pd.Series(values, index=feature.index)
    fr = pd.DataFrame(np.where(mask, f, np.nan)).rank(axis=1)
    tr = pd.DataFrame(np.where(mask, t, np.nan)).rank(axis=1)
    fr = fr.to_numpy(float)
    tr = tr.to_numpy(float)
    for i in np.where(keep)[0]:
        a, b = fr[i][mask[i]], tr[i][mask[i]]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom > 0:
            values[i] = float((a * b).sum() / denom)
    return pd.Series(values, index=feature.index)


def quantile_spread(feature: pd.DataFrame, fwd: pd.DataFrame,
                    q: int, min_symbols: int) -> pd.Series:
    """最高分位减最低分位的未来相对收益（逐日）。"""
    fr = feature.rank(axis=1, pct=True)
    hi = fr >= (1 - 1.0 / q)
    lo = fr <= (1.0 / q)
    hi_ret = fwd.where(hi).mean(axis=1)
    lo_ret = fwd.where(lo).mean(axis=1)
    spread = hi_ret - lo_ret
    counts = feature.notna().sum(axis=1)
    return spread.where(counts >= min_symbols)


def evaluate(panel: pd.DataFrame, meta: pd.DataFrame, config: XSConfig) -> dict:
    feats = compute_xs(panel, config)
    results = []
    for lb in config.lookbacks:
        for h in config.horizons:
            ic = ic_series(feats[f"rs{lb}_rank"], feats[f"fwd{h}_rel"],
                           config.min_symbols_per_day)
            spread = quantile_spread(feats[f"rs{lb}_rank"], feats[f"fwd{h}_rel"],
                                     config.quantiles, config.min_symbols_per_day)
            results.append({
                "kind": "relative_strength",
                "feature": f"rs{lb}",
                "horizon": h,
                "days": int(ic.notna().sum()),
                "ic_mean": float(ic.mean()) if ic.notna().any() else None,
                "ic_t": _newey_west_t(ic, lag=max(h, 20)),
                "ic_positive_share": float((ic > 0).mean()) if ic.notna().any() else None,
                "spread_mean": float(spread.mean()) if spread.notna().any() else None,
                "spread_t": _newey_west_t(spread, lag=max(h, 20)),
            })

    # 波动分层的横截面检验：高波动是否系统性跑输
    for h in config.horizons:
        ic = ic_series(feats["vol_rank"], feats[f"fwd{h}_rel"], config.min_symbols_per_day)
        results.append({
            "kind": "volatility", "feature": "vol20", "horizon": h,
            "days": int(ic.notna().sum()),
            "ic_mean": float(ic.mean()) if ic.notna().any() else None,
            "ic_t": _newey_west_t(ic, lag=max(h, 20)),
            "ic_positive_share": float((ic > 0).mean()) if ic.notna().any() else None,
            "spread_mean": None, "spread_t": None,
        })

    # 绝对方向：RS 最高分位未来上涨的比例，对比全样本上涨比例
    abs_rows = []
    for lb in config.lookbacks:
        for h in config.horizons:
            fwd = feats[f"fwd{h}"]
            fr = feats[f"rs{lb}_rank"]
            top = fr >= 0.8
            bottom = fr <= 0.2
            up_all = float((fwd > 0).stack().mean())
            abs_rows.append({
                "lookback": lb, "horizon": h,
                "top_up_rate": float((fwd.where(top) > 0).stack().mean()),
                "bottom_up_rate": float((fwd.where(bottom) > 0).stack().mean()),
                "all_up_rate": up_all,
            })

    return {"ic": results, "absolute": abs_rows,
            "panel_shape": [int(panel.shape[0]), int(panel.shape[1])],
            "date_range": [str(panel.index.min().date()), str(panel.index.max().date())],
            "meta": meta}


def sectors(meta: pd.DataFrame) -> pd.DataFrame:
    """行业/主题指数单独一层：它们的横截面就是板块轮动。"""
    return meta[meta["category"] == "指数"].copy()
