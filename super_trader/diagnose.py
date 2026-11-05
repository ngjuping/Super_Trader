"""诊断工具：数据体检 + 安慰剂检验。

安慰剂检验（placebo）是本项目最重要的一道防线：
把「未来是否上涨」的标签在时间轴上随机平移，信号、筛选流程、门槛全部不变，再看能
「发现」多少规律。如果真实数据的晋级率和随机标签的晋级率差不多，所谓规律就只是
多重检验的噪声。

用「平移」而不是「完全打乱」：h 日前瞻收益天然高度自相关，完全打乱会抹掉相依结构，
反而让检验更难通过、低估随机标签的通过率。整段循环平移保留原有相依结构，只切断
「标签」与「信号」的对应关系，才是正确的零假设。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import stats
from .config import Config
from .data import KlineStore, quality_report
from .evaluate import _sub_stats, add_selection_flags, prepare_symbol
from .rules import RULES
from .scan import eval_config
from .universe import resolve_pool


def data_check(cfg: Config, pool: str = "mainstream") -> int:
    store = KlineStore(cfg.data_path, cfg.cache_path)
    members = resolve_pool(store, pool)
    rows = []
    for info, tag in members:
        df = store.load(info)
        rep = quality_report(df, info)
        rep["tag"] = tag
        rows.append(rep)
    d = pd.DataFrame(rows)
    d["years"] = (pd.to_datetime(d["end"]) - pd.to_datetime(d["start"])).dt.days / 365.25
    print(f"数据目录: {cfg.data_path}")
    print(f"标的数: {len(d)}   K线合计: {int(d['bars'].sum()):,}")
    print(f"时间范围: {d['start'].min()} ~ {d['end'].max()}")
    print(f"除权/价格重塑 bar 合计 {int(d['artifact_bars'].sum())}   "
          f"停牌缺口 {int(d['suspension_gaps'].sum())}   "
          f"零成交 bar {int(d['zero_volume_bars'].sum())}")
    print()
    print(d[["code", "name", "category", "bars", "start", "end", "years",
             "artifact_bars", "suspension_gaps"]].to_string(index=False))
    out = cfg.report_path / "data_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细 -> {out}")
    return 0


def _block_perm(n: int, block: int, rng) -> np.ndarray:
    """块置换表：把 n 天按 block 天切块，打乱块顺序，返回 old_pos -> new_pos 的映射。

    块长取前瞻周期 h，使置换后的序列在 h 阶内仍保有与原始重叠窗口相近的自相关。
    """
    block = max(1, int(block))
    perm = np.arange(n)
    starts = list(range(0, n, block))
    order = rng.permutation(len(starts))
    pos = 0
    for k in order:
        b0 = starts[k]
        b1 = min(n, b0 + block)
        length = b1 - b0
        perm[b0:b1] = np.arange(pos, pos + length)
        pos += length
    return perm


def _evaluate_shifted(p, ecfg, offsets: dict[int, int]):
    """与 evaluate_prepared 同构，但每个周期的「是否上涨」标签整体循环平移。"""
    info, n, sigs = p.info, p.n, p.sigs
    records: list[dict] = []
    meta = {
        "code": info.code, "name": info.name, "category": info.category,
        "board": info.board, "bars": int(n), "eval_bars": int(n - p.start_idx),
        "eval_start": str(p.dates[p.start_idx].date()),
        "split_date": str(p.dates[min(p.split_idx, n - 1)].date()),
        "end": str(p.dates[-1].date()), "artifact_bars": int(p.art_cum[-1]),
        "horizons": {},
    }
    for h in ecfg.horizons:
        idx, f_cc, _f_oc, up = p.masks(h)
        if idx.size < 100:
            continue
        up = np.roll(up, offsets[h] % max(1, up.size))
        base_up = float(up.mean())
        is_train = idx < p.split_idx
        era_vals = p.era_lab[idx]
        meta["horizons"][str(h)] = {"samples": int(idx.size), "base_up": base_up}
        for r in RULES:
            s_all = sigs[r.key][idx]
            trig = s_all != 0
            if not trig.any():
                continue
            st = _sub_stats(s_all[trig], f_cc[trig], up[trig], base_up, h, ecfg.overlap_adjust)
            tr = _sub_stats(s_all[trig & is_train], f_cc[trig & is_train],
                            up[trig & is_train], base_up, h, ecfg.overlap_adjust)
            ho = _sub_stats(s_all[trig & ~is_train], f_cc[trig & ~is_train],
                            up[trig & ~is_train], base_up, h, ecfg.overlap_adjust)
            eras_pos = eras_tot = 0
            for e in np.unique(era_vals):
                em = trig & (era_vals == e)
                ne = int(em.sum())
                if ne < ecfg.min_era_samples:
                    continue
                eras_tot += 1
                long_e = s_all[em] > 0
                hit_e = (int((long_e & up[em]).sum()) + int((~long_e & ~up[em]).sum())) / ne
                p0_e = (int(long_e.sum()) * base_up + int((~long_e).sum()) * (1 - base_up)) / ne
                if hit_e > p0_e:
                    eras_pos += 1
            records.append({
                "code": info.code, "name": info.name, "category": info.category,
                "board": info.board, "rule": r.key, "family": r.family,
                "desc": r.desc, "horizon": h,
                "n": st["n"], "hit": st["hit"], "acc": st["acc"], "base": st["base"],
                "edge": st["edge"], "p": st["p"], "n_eff": st["n_eff"],
                "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                "n_long": st["n_long"], "n_short": st["n_short"],
                "acc_long": st["acc_long"], "acc_short": st["acc_short"],
                "base_up": base_up, "trigger_rate": st["n"] / float(idx.size),
                "n_train": tr["n"], "acc_train": tr["acc"], "edge_train": tr["edge"],
                "p_train": tr["p"], "train_base": tr["base"],
                "n_hold": ho["n"], "acc_hold": ho["acc"], "edge_hold": ho["edge"],
                "p_hold": ho["p"], "hold_base": ho["base"],
                "eras_total": eras_tot, "eras_positive": eras_pos,
                "acc_exec": float("nan"), "edge_exec": float("nan"), "p_exec": 1.0,
                "mean_ret_long": float("nan"), "mean_ret_short": float("nan"),
                "uncond_mean_ret": float(f_cc.mean()),
            })
    if records:
        rej, qv = stats.bh_fdr([r["p"] for r in records], ecfg.fdr_q)
        for r, ok, q in zip(records, rej, qv):
            r["q_value"] = float(q) if np.isfinite(q) else float("nan")
            r["fdr_pass"] = bool(ok)
    return records, meta


def placebo_test(cfg: Config, n_symbols: int = 12, seed: int = 7,
                 pool: str = "mainstream", reps: int = 3) -> int:
    store = KlineStore(cfg.data_path, cfg.cache_path)
    ecfg = eval_config(cfg)
    members = resolve_pool(store, pool)
    if n_symbols:
        members = members[:n_symbols]
    rng = np.random.default_rng(seed)

    prepared = []
    for info, tag in members:
        df = store.load(info)
        p = prepare_symbol(df, info, ecfg)
        if p is not None:
            prepared.append(p)

    print("安慰剂检验：随机平移「未来是否上涨」标签，流程与门槛和真实扫描完全一致")
    print(f"  标的数 {len(prepared)}   重复次数 {reps}")
    print()
    print(f"{'轮次':<6}{'检验数':>9}{'训练入选':>10}{'入选率':>10}{'晋级':>7}{'晋级率':>10}")
    totals = {"tests": 0, "train": 0, "promoted": 0}
    for rep in range(reps):
        tests = train_pass = promoted = 0
        for p in prepared:
            offsets = {h: int(rng.integers(1, 5000)) for h in ecfg.horizons}
            recs, _ = _evaluate_shifted(p, ecfg, offsets)
            add_selection_flags(recs, ecfg)
            tests += len(recs)
            train_pass += int(sum(r["pass_train"] for r in recs))
            promoted += int(sum(r["promoted"] for r in recs))
        totals["tests"] += tests
        totals["train"] += train_pass
        totals["promoted"] += promoted
        print(f"{rep + 1:<6}{tests:>9}{train_pass:>10}"
              f"{100 * train_pass / max(tests, 1):>9.2f}%{promoted:>7}"
              f"{100 * promoted / max(tests, 1):>9.3f}%")

    tests = max(totals["tests"], 1)
    rate_tr, rate_pr = totals["train"] / tests, totals["promoted"] / tests
    print()
    print(f"安慰剂合计：检验 {totals['tests']:,}   训练入选率 {100 * rate_tr:.2f}%   "
          f"晋级率 {100 * rate_pr:.3f}%")
    print("对照：真实扫描的晋级率 = summary.json 里 totals.promoted / totals.tests")

    res_p = cfg.report_path / "placebo.json"
    res_p.parent.mkdir(parents=True, exist_ok=True)
    res_p.write_text(json.dumps({
        "seed": seed, "reps": reps, "symbols": len(prepared), "tests": totals["tests"],
        "train_pass": totals["train"], "promoted": totals["promoted"],
        "train_rate": rate_tr, "promoted_rate": rate_pr,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {res_p}")
    return 0
