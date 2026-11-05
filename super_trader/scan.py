"""扫描编排：跑池子 → 汇总 → 选规则 → 落盘。

输出：
  results.json        全量证据表（体积大，默认不入库）
  rule_ranking.csv    规则×周期汇总：跨标的一致性 + 市场级检验 + 是否通过验收线 A
  symbol_summary.csv  每个标的一行：样本区间、基准涨率、晋级规则数
  adopted_rules.json  每个标的当前可用的规则（含最新一根 bar 的信号）
  summary.json        运行元信息（数据截止日、样本量、门槛、耗时）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .config import Config
from .data import KlineStore
from .evaluate import (
    EvalConfig, MarketPanel, add_selection_flags, evaluate_prepared, prepare_symbol,
)
from .rules import RULES


@dataclass
class ScanOutcome:
    payload: dict
    out_dir: Path


def eval_config(cfg: Config) -> EvalConfig:
    return EvalConfig(
        horizons=list(cfg.horizons),
        warmup_bars=cfg.warmup_bars,
        train_fraction=cfg.train_fraction,
        min_train_samples=cfg.min_train_samples,
        min_holdout_samples=cfg.min_holdout_samples,
        train_p_max=cfg.train_p_max,
        holdout_p_max=cfg.holdout_p_max,
        fdr_q=cfg.fdr_q,
        overlap_adjust=cfg.overlap_adjust,
        eval_start=cfg.extra.get("eval_start", "2005-01-01"),
        era_starts=tuple(cfg.extra.get("era_starts", (
            "2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01", "2025-01-01",
        ))),
    )


# ------------------------------------------------------------------ 汇总
def _rule_ranking(rec: pd.DataFrame, market: pd.DataFrame,
                  curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """规则×周期的三层证据汇总。

    判据刻意不是「平均命中率高」，而是：
      * 跨标的一致性：多少个不同标的留出段优势为正（防止靠一两个标的撑起来）
      * 市场级检验：把同日横截面压成一个观测、按时间做 HAC 检验（防止 62 个相关标的重复计数）
      * 规格曲线：换基准口径 / HAC 滞后 / 每日最少信号数，结论是否依旧
      * 阶段稳定：不是只在某一轮行情有效
    """
    has_hold = rec["n_hold"] > 0
    g = rec[has_hold]
    rows = []
    for (rule, h), sub in g.groupby(["rule", "horizon"]):
        n_sym = int(len(sub))
        edge = sub["edge_hold"].to_numpy(dtype=float)
        acc = sub["acc_hold"].to_numpy(dtype=float)
        pos = int(np.nansum(edge > 0))
        rows.append({
            "rule": rule, "family": sub["family"].iloc[0], "horizon": int(h),
            "desc": sub["desc"].iloc[0],
            "symbols": n_sym, "symbols_positive": pos,
            "positive_share": pos / n_sym if n_sym else float("nan"),
            "sign_p": stats.sign_test_p(pos, n_sym),
            "mean_edge_hold": float(np.nanmean(edge)),
            "median_edge_hold": float(np.nanmedian(edge)),
            "mean_acc_hold": float(np.nanmean(acc)),
            "mean_base_hold": float(np.nanmean(sub["hold_base"].to_numpy(dtype=float))),
            "mean_trigger": float(np.nanmean(sub["trigger_rate"].to_numpy(dtype=float))),
            "mean_acc_train": float(np.nanmean(sub["acc_train"].to_numpy(dtype=float))),
            "mean_edge_train": float(np.nanmean(sub["edge_train"].to_numpy(dtype=float))),
            "promoted_symbols": int(sub["promoted"].sum()),
            "mean_eras_ratio": float(np.nanmean(
                sub["eras_positive"].to_numpy(dtype=float)
                / np.maximum(1, sub["eras_total"].to_numpy(dtype=float)))),
            "mean_n": float(np.nanmean(sub["n"].to_numpy(dtype=float))),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ci = [stats.bootstrap_mean_ci(v, n_boot=1200) for v in _edges_by_row(g, out)]
    out["mean_edge_ci_lo"] = [c[1] for c in ci]
    out["mean_edge_ci_hi"] = [c[2] for c in ci]

    if not market.empty:
        out = out.merge(market, on=["rule", "horizon"], how="left")
    else:
        for c in ("market_p", "market_q", "market_mean_edge", "market_z", "market_days",
                  "market_share_positive", "market_mean_acc", "market_mean_base"):
            out[c] = np.nan
    if curve is not None and not curve.empty:
        out = out.merge(curve, on=["rule", "horizon"], how="left")
    else:
        for c in ("variants", "variant_pass", "variant_pass_share", "variant_min_edge"):
            out[c] = np.nan

    out["passes_consistency"] = (
        (out["symbols"] >= 20)
        & (out["positive_share"] >= 0.65)
        & (out["market_days"] >= 100)
        & (out["market_p"] <= 0.01)
        & (out["market_q"] <= 0.10)
        & (out["variant_pass_share"] >= 0.75)
        & (out["variant_min_edge"] > 0)
        & (out["mean_edge_hold"] > 0.005)
        & (out["mean_edge_ci_lo"] > 0)
        & (out["mean_eras_ratio"] >= 0.60)
    )
    # 只看「留出段还为正」的样本，避免训练段表现主导排序
    out["rank_score"] = out["market_mean_edge"].fillna(0) * np.sqrt(out["symbols"])
    return out.sort_values(["passes_consistency", "rank_score"], ascending=[False, False])


def _edges_by_row(g: pd.DataFrame, out: pd.DataFrame):
    for _, row in out.iterrows():
        sub = g[(g["rule"] == row["rule"]) & (g["horizon"] == row["horizon"])]
        yield sub["edge_hold"].to_numpy(dtype=float)


def _symbol_summary(rec: pd.DataFrame, metas: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metas:
        if m.get("skipped"):
            rows.append({"code": m["code"], "name": m.get("name", ""),
                         "category": m.get("category", ""), "skipped": m["skipped"]})
            continue
        sub = rec[rec["code"] == m["code"]]
        h1 = m["horizons"].get("1", {})
        best = sub[sub["promoted"]].sort_values("p_hold").head(1)
        rows.append({
            "code": m["code"], "name": m["name"], "category": m["category"],
            "board": m["board"], "bars": m["bars"], "eval_bars": m["eval_bars"],
            "eval_start": m["eval_start"], "split_date": m["split_date"], "end": m["end"],
            "artifact_bars": m["artifact_bars"],
            "base_up_1d": h1.get("base_up", float("nan")),
            "base_up_exec_1d": h1.get("base_up_exec", float("nan")),
            "mean_ret_1d": h1.get("mean_ret", float("nan")),
            "promoted_count": int(sub["promoted"].sum()),
            "best_rule": best["rule"].iloc[0] if len(best) else "",
            "best_horizon": int(best["horizon"].iloc[0]) if len(best) else 0,
            "best_edge_hold": float(best["edge_hold"].iloc[0]) if len(best) else float("nan"),
            "best_acc_hold": float(best["acc_hold"].iloc[0]) if len(best) else float("nan"),
        })
    return pd.DataFrame(rows)


def _adopted(rec: pd.DataFrame, latest: dict[str, dict[str, int]],
             max_per_symbol: int = 5) -> dict:
    out: dict[str, dict] = {}
    for code, sub in rec[rec["promoted"]].groupby("code"):
        sub = sub.sort_values(["p_hold", "edge_hold"], ascending=[True, False])
        sub = sub.drop_duplicates(subset=["rule"], keep="first").head(max_per_symbol)
        items = []
        for _, r in sub.iterrows():
            items.append({
                "rule": r["rule"], "family": r["family"], "desc": r["desc"],
                "horizon": int(r["horizon"]),
                "acc_hold": float(r["acc_hold"]), "base_hold": float(r["hold_base"]),
                "edge_hold": float(r["edge_hold"]), "p_hold": float(r["p_hold"]),
                "n_hold": int(r["n_hold"]),
                "acc_train": float(r["acc_train"]), "edge_train": float(r["edge_train"]),
                "trigger_rate": float(r["trigger_rate"]),
                "eras_positive": int(r["eras_positive"]), "eras_total": int(r["eras_total"]),
                "acc_exec": float(r["acc_exec"]),
                "today_signal": int(latest.get(code, {}).get(r["rule"], 0)),
            })
        out[code] = {"name": sub["name"].iloc[0],
                     "category": sub["category"].iloc[0], "rules": items}
    return out


# ------------------------------------------------------------------ 主流程
def run_scan(cfg: Config | None = None, pool: str = "mainstream", top: int | None = None,
             out_name: str | None = None, limit: int | None = None,
             verbose: bool = True) -> ScanOutcome:
    cfg = cfg or Config.load()
    cfg.ensure_dirs()
    store = KlineStore(cfg.data_path, cfg.cache_path)
    ecfg = eval_config(cfg)

    if pool == "auto":
        from .universe import auto_liquid
        members = auto_liquid(store, top=top or 40)
    else:
        from .universe import resolve_pool
        members = resolve_pool(store, pool)
    if limit:
        members = members[:limit]

    # ---- 1) 先把数据读进内存，并建立统一交易日历（市场级检验需要对齐日期）
    loaded: list[tuple] = []
    t0 = time.time()
    for info, tag in members:
        try:
            df = store.load(info)
        except Exception as exc:
            if verbose:
                print(f"  ! 读取失败 {info.code} {info.name}: {exc}", flush=True)
            continue
        if len(df) < cfg.min_bars:
            if verbose:
                print(f"  - 跳过 {info.name}（{len(df)} 根 < {cfg.min_bars}）", flush=True)
            continue
        loaded.append((info, tag, df))

    all_dates = set()
    for _, _, df in loaded:
        d = pd.to_datetime(df["date"])
        all_dates.update(d[d >= pd.Timestamp(ecfg.eval_start)].values)
    calendar = pd.DatetimeIndex(sorted(all_dates))
    panel = MarketPanel(calendar, ecfg.horizons)

    # ---- 2) 逐标的评估
    all_records: list[dict] = []
    metas: list[dict] = []
    latest: dict[str, dict[str, int]] = {}
    for i, (info, tag, df) in enumerate(loaded, 1):
        prep = prepare_symbol(df, info, ecfg)
        if prep is None:
            metas.append({"code": info.code, "name": info.name,
                          "category": info.category, "skipped": "insufficient_eval_window"})
            continue
        recs, meta = evaluate_prepared(prep, ecfg)
        if not recs:
            metas.append({"code": info.code, "name": info.name,
                          "category": info.category, "skipped": "no_signals"})
            continue
        add_selection_flags(recs, ecfg)
        for r in recs:
            r["tag"] = tag
        all_records.extend(recs)
        meta["tag"] = tag
        metas.append(meta)
        latest[info.code] = {k: int(v[-1]) for k, v in prep.sigs.items()}
        panel.add(prep, ecfg)
        if verbose:
            n_pro = sum(r["promoted"] for r in recs)
            print(f"  [{i:>2}/{len(loaded)}] {info.name:<10s} bars={len(df):>5d} "
                  f"tests={len(recs):>4d} promoted={n_pro:>2d} ({time.time() - t0:.1f}s)",
                  flush=True)

    rec = pd.DataFrame(all_records)
    market = panel.results(ecfg)
    curve = panel.specification_curve(ecfg)
    ranking = _rule_ranking(rec, market, curve) if not rec.empty else pd.DataFrame()
    symbol_summary = _symbol_summary(rec, metas)
    adopted = _adopted(rec, latest) if not rec.empty else {}

    train_pass = int(rec["pass_train"].sum()) if not rec.empty else 0
    promoted = int(rec["promoted"].sum()) if not rec.empty else 0
    expected_false = train_pass * cfg.holdout_p_max

    market_date = max((m.get("end") for m in metas if m.get("end")), default="")
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_date": market_date,
        "pool": pool,
        "config": cfg.to_dict(),
        "gate": {
            "eval_start": ecfg.eval_start,
            "train_fraction": cfg.train_fraction,
            "train_p_max": cfg.train_p_max,
            "holdout_p_max": cfg.holdout_p_max,
            "min_train_samples": cfg.min_train_samples,
            "min_holdout_samples": cfg.min_holdout_samples,
            "fdr_q": cfg.fdr_q,
            "overlap_adjust": cfg.overlap_adjust,
        },
        "totals": {
            "symbols_requested": len(members),
            "symbols_evaluated": len(metas),
            "tests": int(len(rec)),
            "rules": len(RULES),
            "train_pass": train_pass,
            "expected_false_holdout": expected_false,
            "promoted": promoted,
            "promoted_over_expected": (promoted / expected_false) if expected_false > 0 else float("nan"),
            "fdr_pass": int(rec["fdr_pass"].sum()) if not rec.empty else 0,
            "tier_a_rules": int(ranking["passes_consistency"].sum()) if not ranking.empty else 0,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "symbol_meta": metas,
        "symbol_summary": symbol_summary.to_dict("records"),
        "rule_ranking": ranking.to_dict("records") if not ranking.empty else [],
        "adopted": adopted,
        "rule_catalog": [{"key": r.key, "family": r.family, "desc": r.desc} for r in RULES],
        "records": all_records,
    }

    name = out_name or cfg.scan_name
    out_dir = cfg.report_path / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if not ranking.empty:
        ranking.to_csv(out_dir / "rule_ranking.csv", index=False, encoding="utf-8-sig")
    symbol_summary.to_csv(out_dir / "symbol_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "adopted_rules.json").write_text(
        json.dumps(adopted, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({k: payload[k] for k in ("generated_at", "market_date", "pool", "gate", "totals")},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n完成：{payload['totals']}", flush=True)
        print(f"输出目录：{out_dir}", flush=True)
    return ScanOutcome(payload=payload, out_dir=out_dir)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)
