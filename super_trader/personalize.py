"""按标的研究规则与持有周期，结果属于探索性候选。

周期在训练段按方向化超额收益/持有天数选择，不能按留出显著性挑选。
边界净化只消除标签窗口交叉，不保证各段统计独立。
日线 H1/H3 等是持有期，不是分钟线/日线这样的 K 线频率。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .config import Config
from .data import KlineStore
from .evaluate import EvalConfig, PreparedSymbol, prepare_symbol
from .rules import RULES


@dataclass
class PersonalConfig:
    n_folds: int = 4              # 留出段切成几块做连续复现
    min_train: int = 60           # 训练段最少触发
    min_fold: int = 20            # 每一块最少触发
    min_holdout: int = 80         # 留出段合计最少触发
    train_p_max: float = 0.10
    holdout_p_max: float = 0.05
    min_edge: float = 0.005       # 留出段优势 > 0.5pp
    min_folds_positive: int = 4   # 必须是 4/4 全同向
    # ---- 周期公平性相关 ----
    purge_folds: bool = True      # 折与折之间留出 h 根的空隙，避免前瞻窗口跨界
    horizon_pick: str = "train_daily"  # train_daily=只用训练段按「每日毛超额收益」定周期；all=每个周期单独验
    max_mdes: float = 0.20        # 仅报告提示，不作为新增通过门槛
    min_power_eff: float = 8.0    # n/h近似低于此值时提示证据薄弱，不新增淘汰门槛


def _fold_blocks(idx_hold: np.ndarray, n_folds: int, gap: int = 0, bar_idx=None) -> list[np.ndarray]:
    """把留出段切成 n_folds 个连续块。

    gap>0 时从每块末尾切掉 gap 个样本：这些样本的 h 日前瞻窗口会伸进下一块，
    若保留，四块就不再是四段时间稳定性检查，长周期会因为跨界重叠而虚高通过率。
    """
    if gap <= 0:
        return [b for b in np.array_split(idx_hold, n_folds) if b.size > 0]
    groups = np.array_split(idx_hold, n_folds)
    out = []
    for i, g in enumerate(groups):
        if i < len(groups) - 1:
            if bar_idx is None:
                g = g[: max(0, g.size - gap)]
            else:
                boundary = bar_idx[groups[i + 1][0]] if groups[i + 1].size else np.inf
                g = g[bar_idx[g] + gap < boundary]
        if g.size:
            out.append(g)
    return out


def _daily_score(sig, fwd, baseline: float, horizon: int) -> float:
    """方向化毛超额收益/持有天数，不代表可执行净利润。"""
    if len(sig) == 0:
        return float("nan")
    return float(np.mean(sig * (fwd - baseline))) / horizon


def _sub(sig, up, fwd, base_up, horizon, overlap: bool) -> dict:
    n = int(sig.size)
    if n == 0:
        return {"n": 0, "edge": float("nan"), "acc": float("nan"), "p": 1.0,
                "n_long": 0, "n_short": 0, "ret": float("nan")}
    long_m = sig > 0
    short_m = sig < 0
    n_l, n_s = int(long_m.sum()), int(short_m.sum())
    hit = int(up[long_m].sum()) + int((~up[short_m]).sum())
    acc = hit / n
    p0 = (n_l * base_up + n_s * (1 - base_up)) / n
    p = stats.binom_p_upper(acc, stats.effective_n(n, horizon, overlap), p0)
    return {"n": n, "edge": acc - p0, "acc": acc, "p": p, "p0": p0,
            "n_long": n_l, "n_short": n_s, "ret": float(np.mean(fwd))}


def evaluate_symbol_personal(
    p: PreparedSymbol,
    ecfg: EvalConfig,
    pcfg: PersonalConfig | None = None,
    rng=None,
) -> list[dict]:
    """单标的个性化筛选：训练段选规则与周期，留出段用分块+空隙做连续复现。

    三处针对「长周期吃亏」的修正：
    1. **折间空隙（purge）**：每块末尾切掉 h 根，避免前瞻窗口跨块，四块才是不共享前瞻标签。
    2. **周期只在训练段定**：按「每日毛超额收益」选周期（不是按留出段显著性选），
       这样周期不是看了答案才挑的，也就不会自动倒向样本最多的 H1。
    3. **功效记录**：每格都记下 n_eff=n/h 与最小可检出优势 mdes；
       mdes 太大说明这格根本没能力分辨，标成「功效不足」而不是「没有规律」。

    rng 不为 None 时进入安慰剂模式：在每一段内部独立打乱标签（训练段、每个留出块各一次）。
    """
    pcfg = pcfg or PersonalConfig()
    info, n, sigs = p.info, p.n, p.sigs

    # ---------------- 先把每个周期的样本空间准备好 ----------------
    ctx: dict[int, dict] = {}
    for h in ecfg.horizons:
        idx, f_cc, f_oc, up = p.masks(h)
        if idx.size < 200:
            continue
        is_train = idx < p.split_idx
        ntr = int(is_train.sum())
        hold_pos = np.where(~is_train)[0]
        if hold_pos.size < pcfg.min_holdout:
            continue
        gap = h if pcfg.purge_folds else 0
        blocks = _fold_blocks(hold_pos, pcfg.n_folds, gap=gap, bar_idx=idx)
        if len(blocks) < pcfg.n_folds:
            continue
        ntr_keep = int(np.sum(idx + h < p.split_idx)) if gap else ntr   # 训练段同样切掉与留出段重叠的尾部
        if rng is not None:
            # 同步置换涨跌标签和收益，安慰剂也必须重新做训练段周期选择。
            # 段内置换会破坏序列相关，仅作敏感性诊断，不等于严格FDR估计。
            up, f_cc = up.copy(), f_cc.copy()
            for b in [np.arange(ntr_keep), *blocks]:
                order = rng.permutation(b)
                up[b], f_cc[b] = up[order], f_cc[order]
        ctx[h] = {
            "idx": idx, "fwd": f_cc, "up": up, "is_train": is_train,
            "ntr": ntr, "ntr_keep": ntr_keep, "blocks": blocks,
            "base_up": float(up[:ntr_keep].mean()) if ntr_keep else 0.5,
            "hold_mask": np.isin(np.arange(idx.size), np.concatenate(blocks)) if blocks else np.zeros(idx.size, bool),
        }

    out: list[dict] = []
    for r in RULES:
        profiles: list[dict] = []
        for h, c in ctx.items():
            idx, up, fwd = c["idx"], c["up"], c["fwd"]
            s_all = sigs[r.key][idx]
            trig = s_all != 0
            if not trig.any():
                continue
            base_up = c["base_up"]
            ntr_keep = c["ntr_keep"]

            # ---- 训练段（已切掉与留出段重叠的尾部）----
            tsel = np.zeros(idx.size, dtype=bool)
            tsel[:ntr_keep] = True
            tm = trig & tsel
            if int(tm.sum()) < pcfg.min_train:
                continue
            tr = _sub(s_all[tm], up[tm], fwd[tm], base_up, h, ecfg.overlap_adjust)

            # ---- 留出段分块 ----
            fold_edges, fold_ns = [], []
            for b in c["blocks"]:
                m = trig[b]
                nb = int(m.sum())
                fold_ns.append(nb)
                fold_edges.append(
                    _sub(s_all[b][m], up[b][m], fwd[b][m], float(up[b].mean()), h, ecfg.overlap_adjust)["edge"]
                    if nb else float("nan"))

            # ---- 留出段整体（只取四块覆盖的样本，不含空隙）----
            hm = trig & c["hold_mask"]
            ho = _sub(s_all[hm], up[hm], fwd[hm], float(up[c["hold_mask"]].mean()), h, ecfg.overlap_adjust)
            n_ho = ho["n"]

            fe = np.array(fold_edges, dtype=float)
            fn = np.array(fold_ns, dtype=float)
            okf = np.isfinite(fe)
            folds_pos = int(np.sum(fe[okf] > 0))
            folds_enough = bool(np.all(fn[okf] >= pcfg.min_fold))

            n_eff = stats.effective_n(n_ho, h, ecfg.overlap_adjust) if n_ho else 0.0
            mdes = stats.min_detectable_edge(n_eff, ho.get("p0", base_up)) if n_eff else float("nan")
            power_ok = bool(np.isfinite(mdes) and mdes <= pcfg.max_mdes and n_eff >= pcfg.min_power_eff)

            # 每日毛超额收益（用于在训练段选周期；经济口径，不是显著性口径）
            # 候选周期使用相同训练截止；有效日期可因artifact等掩码不同。正负信号均计入。
            score_mask = trig & (idx + max(ecfg.horizons) < p.split_idx)
            score_base_mask = idx + max(ecfg.horizons) < p.split_idx
            train_base_return = float(np.mean(fwd[score_base_mask])) if score_base_mask.any() else 0.0
            daily_train = (_daily_score(s_all[score_mask], fwd[score_mask], train_base_return, h)
                           if score_mask.sum() >= pcfg.min_train else float("nan"))

            tr_ok = (tr["n"] >= pcfg.min_train and tr["edge"] > 0 and tr["p"] <= pcfg.train_p_max)
            fold_ok = (okf.sum() == pcfg.n_folds and folds_enough
                       and folds_pos >= pcfg.min_folds_positive)
            ho_ok = (n_ho >= pcfg.min_holdout and ho["edge"] >= pcfg.min_edge
                     and ho["p"] <= pcfg.holdout_p_max)

            profiles.append({
                "code": info.code, "name": info.name, "category": info.category,
                "board": info.board, "rule": r.key, "family": r.family,
                "desc": r.desc, "horizon": h,
                "n_train": tr["n"], "acc_train": tr["acc"], "base_train": tr.get("p0", float("nan")),
                "edge_train": tr["edge"], "p_train": tr["p"],
                "n_hold": n_ho, "acc_hold": ho["acc"], "base_hold": ho.get("p0", float("nan")),
                "edge_hold": ho["edge"], "p_hold": ho["p"],
                "fold_edges": [None if not np.isfinite(x) else float(x) for x in fe],
                "fold_ns": [int(x) for x in fn],
                "folds_positive": folds_pos, "folds_enough": folds_enough,
                "min_fold_edge": float(np.nanmin(fe)) if okf.any() else float("nan"),
                "trigger_rate": n_ho / float(idx.size) if idx.size else float("nan"),
                "n_long": ho["n_long"], "n_short": ho["n_short"],
                "acc_exec": float("nan"),
                "n_eff_hold": float(n_eff), "mdes_hold": float(mdes), "power_ok": power_ok,
                "daily_train": float(daily_train),
                "daily_hold": _daily_score(s_all[hm], fwd[hm], float(np.mean(fwd[c["hold_mask"]])), h),
                "pass_train": bool(tr_ok), "pass_folds": bool(fold_ok),
                "pass_holdout": bool(ho_ok),
                "validated": bool(tr_ok and fold_ok and ho_ok),
                "watchlist": bool(tr_ok and okf.sum() == pcfg.n_folds and folds_enough
                                  and folds_pos >= pcfg.n_folds - 1 and ho["edge"] > 0
                                  and not (tr_ok and fold_ok and ho_ok)),
                "horizon_chosen": False,
            })

        if not profiles:
            continue

        if pcfg.horizon_pick == "train_daily":
            # 周期只在训练段定：取「每日毛超额收益」最高的那个，其余周期一律不晋级
            cands = [x for x in profiles if x["n_train"] >= pcfg.min_train and np.isfinite(x["daily_train"])]
            pick = max(cands, key=lambda x: x["daily_train"]) if cands else None
            for x in profiles:
                if pick is not None and x is pick:
                    x["horizon_chosen"] = True
                else:
                    x["validated"] = False
                    x["watchlist"] = False
        else:
            for x in profiles:
                if x["validated"]:
                    x["horizon_chosen"] = True
        out.extend(profiles)
    return out


# ------------------------------------------------------------------ 编排
def _personal_cfg(cfg: Config) -> tuple[EvalConfig, PersonalConfig]:
    from .scan import eval_config

    ecfg = eval_config(cfg)
    pcfg = PersonalConfig()
    return ecfg, pcfg


def pick_best(records: pd.DataFrame) -> pd.DataFrame:
    """每个标的取一条最好的已验证规则（同标的同规则只保留最好的周期）。"""
    v = records[records["validated"]].copy()
    if v.empty:
        return v
    v = v.sort_values(["p_hold", "edge_hold"], ascending=[True, False])
    v = v.drop_duplicates(subset=["code", "rule"], keep="first")
    return v.sort_values("edge_hold", ascending=False)


def _accepted_pairs(cfg: Config) -> list[tuple[str, int]]:
    """读取市场级扫描里通过验收线 A 的 (规则, 周期)。"""
    p = cfg.report_path / (cfg.scan_name or "mainstream_daily") / "results.json"
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [(r["rule"], int(r["horizon"]))
            for r in payload.get("rule_ranking", []) if r.get("passes_consistency")]


def add_gain_columns(rec_df: pd.DataFrame, accepted: list[tuple[str, int]]) -> pd.DataFrame:
    """给每条已验证规则补三列，回答「个性化到底多赚了多少」：

    * rule_mean_edge：同一规则在全部标的上的留出段平均优势（这条规则的市场级性格）
    * specificity：本标的的优势 − 该规则的平均优势（>0 才算「这个标的格外适合它」）
    * gain：本标的的个性化最优优势 − 通用规则在本标的上能拿到的最好优势
    """
    if rec_df.empty:
        return rec_df
    df = rec_df.copy()
    mean_edge = df.groupby(["rule", "horizon"])["edge_hold"].mean()
    df["rule_mean_edge"] = [
        float(mean_edge.get((r, h), np.nan)) for r, h in zip(df["rule"], df["horizon"])
    ]
    df["specificity"] = df["edge_hold"] - df["rule_mean_edge"]

    # 通用规则在每个标的上能达到的最好优势
    market_best: dict[str, float] = {}
    market_best_rule: dict[str, str] = {}
    if accepted:
        acc = set(accepted)
        sub = df[[(r, h) in acc for r, h in zip(df["rule"], df["horizon"])]]
        for code, g in sub.groupby("code"):
            i = g["edge_hold"].idxmax()
            market_best[code] = float(g.loc[i, "edge_hold"])
            market_best_rule[code] = f"{g.loc[i, 'rule']} H{int(g.loc[i, 'horizon'])}"
    df["market_best_edge"] = df["code"].map(market_best)
    df["market_best_rule"] = df["code"].map(market_best_rule)
    df["gain"] = df["edge_hold"] - df["market_best_edge"]
    return df


def best_per_symbol(validated: pd.DataFrame) -> pd.DataFrame:
    """每个标的一条主规则（按留出段优势取最好，同分时 p 值小的优先）。"""
    if validated.empty:
        return validated
    # 修正版主规则按训练分数排序，不再对留出优势取最大。
    cols = ["daily_train"] if "daily_train" in validated.columns else ["edge_hold"]
    v = validated.sort_values(cols, ascending=False)
    return v.drop_duplicates(subset=["code"], keep="first")


def run_personalize(cfg: Config | None = None, pool: str = "mainstream",
                    reps: int = 2, seed: int = 17, verbose: bool = True) -> dict:
    cfg = cfg or Config.load()
    cfg.ensure_dirs()
    ecfg, pcfg = _personal_cfg(cfg)
    store = KlineStore(cfg.data_path, cfg.cache_path)

    from .universe import resolve_pool

    members = resolve_pool(store, pool)
    rng = np.random.default_rng(seed)
    preps: list[PreparedSymbol] = []
    for info, tag in members:
        df = store.load(info)
        if len(df) < cfg.min_bars:
            continue
        p = prepare_symbol(df, info, ecfg)
        if p is not None:
            preps.append(p)

    t0 = time.time()
    if verbose:
        print(f"个性化筛选：{len(preps)} 个标的 × {len(RULES)} 条规则 × {len(ecfg.horizons)} 个周期")
        print(f"  训练段 p≤{pcfg.train_p_max}，留出段切成 {pcfg.n_folds} 块且需 "
              f"{pcfg.min_folds_positive}/{pcfg.n_folds} 块同向，留出 p≤{pcfg.holdout_p_max}\n")

    real: list[dict] = []
    for i, p in enumerate(preps, 1):
        recs = evaluate_symbol_personal(p, ecfg, pcfg)
        real.extend(recs)
        if verbose:
            nv = sum(r["validated"] for r in recs)
            nw = sum(r["watchlist"] for r in recs)
            print(f"  [{i:>2}/{len(preps)}] {p.info.name:<10s} 验证 {nv:>2d}  候选 {nw:>3d} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    rec_df = pd.DataFrame(real)

    # 安慰剂：同一套流程，在每一段内部独立打乱标签（正确的零假设）
    placebo_counts = []
    placebo_symbols = []
    placebo_gains: list[float] = []
    accepted = _accepted_pairs(cfg)
    for rep in range(reps):
        cnt = 0
        codes: set[str] = set()
        pool_recs: list[dict] = []
        for p in preps:
            recs = evaluate_symbol_personal(p, ecfg, pcfg, rng=rng)
            pool_recs.extend(recs)
            ok = [r for r in recs if r["validated"]]
            cnt += len(ok)
            codes.update(r["code"] for r in ok)
        # 安慰剂版本的「个性化增益」：同样取最优，看随机标签下能「多赚」多少
        if pool_recs:
            pdf = add_gain_columns(pd.DataFrame(pool_recs), accepted)
            pb = best_per_symbol(pdf[pdf["validated"]])
            if not pb.empty:
                g = pb["gain"].dropna()
                if g.size:
                    placebo_gains.append(float(g.mean()))
        placebo_counts.append(cnt)
        placebo_symbols.append(len(codes))
        if verbose:
            print(f"  安慰剂第 {rep + 1} 轮：验证 {cnt} 条 / {len(codes)} 个标的", flush=True)
    placebo_mean = float(np.mean(placebo_counts)) if placebo_counts else float("nan")
    placebo_sym_mean = float(np.mean(placebo_symbols)) if placebo_symbols else float("nan")
    placebo_gain_mean = float(np.mean(placebo_gains)) if placebo_gains else float("nan")

    rec_df = add_gain_columns(rec_df, accepted)
    best = pick_best(rec_df)
    per_symbol = best_per_symbol(rec_df[rec_df["validated"]]) if not rec_df.empty else pd.DataFrame()
    watch = rec_df[rec_df["watchlist"]].copy()
    if not watch.empty:
        watch = watch.sort_values("edge_hold", ascending=False).drop_duplicates(
            subset=["code", "rule"], keep="first")

    distinct_rules = int(best["rule"].nunique()) if not best.empty else 0
    gains = per_symbol["gain"].dropna() if not per_symbol.empty else pd.Series(dtype=float)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pool": pool,
        "symbols": len(preps),
        "rules": len(RULES),
        "horizons": list(ecfg.horizons),
        "tests": int(len(rec_df)),
        "gate": asdict(pcfg),
        "method_revision": "purged_train_only_horizon_v2",
        "research_status": "exploratory_same_history_not_fresh_holdout",
        "horizon_diagnostics": [
            {"horizon": int(h), "tests": len(g),
             "chosen_train": int(g["horizon_chosen"].sum()),
             "validated": int(g["validated"].sum()),
             "median_n_hold": float(g["n_hold"].median()),
             "median_n_eff": float(g["n_eff_hold"].median()),
             "median_mdes": float(g["mdes_hold"].median()),
             "power_limited": int((~g["power_ok"]).sum())}
            for h, g in rec_df.groupby("horizon")],
        "accepted_market_pairs": [f"{r} H{h}" for r, h in accepted],
        "totals": {
            "validated_pairs": int(rec_df["validated"].sum()),
            "validated_symbols": int(per_symbol["code"].nunique()) if not per_symbol.empty else 0,
            "distinct_rules": distinct_rules,
            "watchlist": int(rec_df["watchlist"].sum()),
            "placebo_pairs": placebo_mean,
            "placebo_symbols": placebo_sym_mean,
            "placebo_runs": placebo_counts,
            "ratio": (len(best) / placebo_mean) if placebo_mean else None,
            "gain_mean": float(gains.mean()) if gains.size else float("nan"),
            "gain_positive": int((gains > 0).sum()) if gains.size else 0,
            "gain_median": float(gains.median()) if gains.size else float("nan"),
            "gain_placebo": placebo_gain_mean,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "by_symbol": best.to_dict("records") if not best.empty else [],
        "per_symbol_best": per_symbol.to_dict("records") if not per_symbol.empty else [],
        "watchlist": watch.to_dict("records") if not watch.empty else [],
        "records": rec_df.to_dict("records") if not rec_df.empty else [],
    }
    out_dir = cfg.report_path / "personalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "personalized.json").write_text(
        json.dumps(payload, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if verbose:
        t = payload["totals"]
        print(f"\n完成：验证 {t['validated_pairs']} 条规则-标的对 / {t['validated_symbols']} 个标的 "
              f"（{t['distinct_rules']} 条不同规则）")
        print(f"      安慰剂平均 {placebo_mean:.1f} 条 / {placebo_sym_mean:.1f} 个标的")
        print(f"      个性化增益：均值 {t['gain_mean']*100:+.2f}pp（安慰剂 "
              f"{placebo_gain_mean*100:+.2f}pp），{t['gain_positive']}/{len(gains)} 个标的优于通用规则")
    return payload


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
    return str(o)
