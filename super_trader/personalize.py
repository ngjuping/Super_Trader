"""个性化规则：每个标的、每个周期，各自选自己的规律。

为什么需要单独一层
------------------
市场级检验问的是「这条规律是不是全市场通用」。但不同标的的性格确实不同：
银行股多是低波动均值回归，资源股是趋势跟随，小盘成长股是情绪脉冲。
把它们硬塞进一条统一规则，等于先假设了个性化不存在——这正是第一轮的缺陷所在。

问题不在于「个性化有没有意义」，而在于「怎么证明它不是拟合出来的」。
78 条规则 × 5 个周期 = 390 个假设 / 标的，如果只做一次 60/40 切分，
光靠运气就会有一批规则「晋级」。第一轮的安慰剂已经证明了这一点。

所以这一层的核心不是模型，而是**验证方式**：

1. 训练段选规则（前 60%，门槛 p≤0.10，方向为正）。
2. 留出段切成 K=4 个**互不重叠**的时间块，规则必须在**每一块**都同向为正，
   且每块样本量 ≥20。任何一块翻向就淘汰——这 4 块是 4 次独立复现。
3. 留出段整体再算一次重叠修正后的 p 值（≤0.05）与优势（>0.5pp）。
4. 用同样的流程跑随机标签，得到经验假阳性率；只有真实数远高于安慰剂才认账。

第 2 条是本层的关键：一次留出可以被一次运气骗过，四次连续复现不行。
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


def _fold_blocks(idx_hold: np.ndarray, n_folds: int) -> list[np.ndarray]:
    return [b for b in np.array_split(idx_hold, n_folds) if b.size > 0]


def evaluate_symbol_personal(
    p: PreparedSymbol,
    ecfg: EvalConfig,
    pcfg: PersonalConfig | None = None,
    rng=None,
) -> list[dict]:
    """单标的个性化筛选：训练段选、留出段分块连续复现。

    rng 不为 None 时进入安慰剂模式：在**每一段内部**独立打乱「是否上涨」标签
    （训练段一次、每个留出块各一次）。这是正确的零假设——标签与信号无关。
    不能用「整段平移」：平移保留了段与段之间的同向性，会让四段复现这道关失效。
    """
    pcfg = pcfg or PersonalConfig()
    info, n, sigs = p.info, p.n, p.sigs
    out: list[dict] = []

    for h in ecfg.horizons:
        idx, f_cc, f_oc, up = p.masks(h)
        if idx.size < 200:
            continue

        is_train = idx < p.split_idx
        hold_pos = np.where(~is_train)[0]
        if hold_pos.size < pcfg.min_holdout:
            continue
        blocks = _fold_blocks(hold_pos, pcfg.n_folds)

        if rng is not None:
            up = up.copy()
            ntr = int(is_train.sum())
            up[:ntr] = rng.permutation(up[:ntr])
            for b in blocks:
                up[b] = rng.permutation(up[b])

        base_up = float(up.mean())

        for r in RULES:
            s_all = sigs[r.key][idx]
            trig = s_all != 0
            if not trig.any():
                continue

            # ---- 训练段 ----
            tm = trig & is_train
            n_tr = int(tm.sum())
            if n_tr < pcfg.min_train:
                continue
            long_t = s_all[tm] > 0
            hit_t = int((long_t & up[tm]).sum()) + int((~long_t & ~up[tm]).sum())
            acc_t = hit_t / n_tr
            p0_t = (int(long_t.sum()) * base_up + int((~long_t).sum()) * (1 - base_up)) / n_tr
            edge_t = acc_t - p0_t
            p_t = stats.binom_p_upper(acc_t, stats.effective_n(n_tr, h, ecfg.overlap_adjust), p0_t)

            # ---- 留出段分块复现 ----
            fold_edges, fold_ns, fold_hits = [], [], []
            for b in blocks:
                m = trig[b]
                nb = int(m.sum())
                fold_ns.append(nb)
                if nb == 0:
                    fold_edges.append(float("nan"))
                    fold_hits.append(float("nan"))
                    continue
                long_b = s_all[b][m] > 0
                hit_b = int((long_b & up[b][m]).sum()) + int((~long_b & ~up[b][m]).sum())
                acc_b = hit_b / nb
                p0_b = (int(long_b.sum()) * base_up + int((~long_b).sum()) * (1 - base_up)) / nb
                fold_edges.append(acc_b - p0_b)
                fold_hits.append(acc_b)

            # ---- 留出段整体 ----
            hm = trig & (~is_train)
            n_ho = int(hm.sum())
            if n_ho == 0:
                continue
            long_h = s_all[hm] > 0
            hit_h = int((long_h & up[hm]).sum()) + int((~long_h & ~up[hm]).sum())
            acc_h = hit_h / n_ho
            p0_h = (int(long_h.sum()) * base_up + int((~long_h).sum()) * (1 - base_up)) / n_ho
            edge_h = acc_h - p0_h
            p_h = stats.binom_p_upper(acc_h, stats.effective_n(n_ho, h, ecfg.overlap_adjust), p0_h)

            fe = np.array(fold_edges, dtype=float)
            fn = np.array(fold_ns, dtype=float)
            valid_folds = np.isfinite(fe)
            folds_pos = int(np.sum(fe[valid_folds] > 0))
            folds_enough = bool(np.all(fn[valid_folds] >= pcfg.min_fold))

            tr_ok = (n_tr >= pcfg.min_train and edge_t > 0 and p_t <= pcfg.train_p_max)
            fold_ok = (valid_folds.sum() == pcfg.n_folds and folds_enough
                       and folds_pos >= pcfg.min_folds_positive)
            ho_ok = (n_ho >= pcfg.min_holdout and edge_h >= pcfg.min_edge
                     and p_h <= pcfg.holdout_p_max)

            # 候选：3/4 块同向 + 训练段显著 + 留出方向为正（未达全部门槛）
            watch = (tr_ok and valid_folds.sum() == pcfg.n_folds and folds_enough
                     and folds_pos >= pcfg.n_folds - 1 and edge_h > 0)

            out.append({
                "code": info.code, "name": info.name, "category": info.category,
                "board": info.board, "rule": r.key, "family": r.family,
                "desc": r.desc, "horizon": h,
                "n_train": n_tr, "acc_train": acc_t, "base_train": p0_t,
                "edge_train": edge_t, "p_train": p_t,
                "n_hold": n_ho, "acc_hold": acc_h, "base_hold": p0_h,
                "edge_hold": edge_h, "p_hold": p_h,
                "fold_edges": [None if not np.isfinite(x) else float(x) for x in fe],
                "fold_ns": [int(x) for x in fn],
                "folds_positive": folds_pos,
                "folds_enough": folds_enough,
                "min_fold_edge": float(np.nanmin(fe)) if np.isfinite(fe).any() else float("nan"),
                "trigger_rate": n_ho / float(idx.size),
                "n_long": int(long_h.sum()), "n_short": int((~long_h).sum()),
                "acc_exec": float("nan"),
                "pass_train": bool(tr_ok),
                "pass_folds": bool(fold_ok),
                "pass_holdout": bool(ho_ok),
                "validated": bool(tr_ok and fold_ok and ho_ok),
                "watchlist": bool(watch and not (tr_ok and fold_ok and ho_ok)),
            })
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
    cols = ["edge_hold"] + (["p_hold"] if "p_hold" in validated.columns else [])
    v = validated.sort_values(cols, ascending=[False] * len(cols))
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
