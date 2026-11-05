"""今日信号：把扫描出来的规则套到最新一根 K 线上。

它回答的是「按这套规则，今天收盘后各个标的怎么看」，不是「明天一定涨/跌」。
输出同时给出规则的留出段战绩和历史触发率，让人自己判断要不要信。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import KlineStore
from .features import build_features
from .rules import rule_signals


def _latest_signals(store: KlineStore, info) -> tuple[dict[str, int], dict]:
    df = store.load(info)
    tail = df.tail(500).reset_index(drop=True)
    fs = build_features(tail, price_limit=info.price_limit)
    sigs = rule_signals(fs)
    last = {k: int(v[-1]) for k, v in sigs.items()}
    ctx = {
        "close": float(df["close"].iloc[-1]),
        "date": str(pd.to_datetime(df["date"].iloc[-1]).date()),
        "chg": float(df["pct_chg"].iloc[-1]) if "pct_chg" in df else float("nan"),
        "atr_pct": float(fs["atr_pct"][-1]) if np.isfinite(fs["atr_pct"][-1]) else None,
        "rsi14": float(fs["rsi14"][-1]) if np.isfinite(fs["rsi14"][-1]) else None,
        "pos60": float(fs["pos60"][-1]) if np.isfinite(fs["pos60"][-1]) else None,
    }
    return last, ctx


def predict_today(cfg: Config | None = None, name: str | None = None,
                  only: str | None = None, top: int = 30) -> int:
    cfg = cfg or Config.load()
    scan = cfg.report_path / (name or cfg.scan_name)
    adopted_p = scan / "adopted_rules.json"
    if not adopted_p.exists():
        print(f"找不到 {adopted_p}，请先运行 scan")
        return 2
    adopted = json.loads(adopted_p.read_text(encoding="utf-8"))
    if not adopted:
        print("本次扫描没有标的晋级（样本太小或门槛过严），没有可展示的今日信号。")
        print("可以先看 reports/<name>/report.html 里通过验收线的规则级结论。")
        return 0
    rec_p = scan / "results.json"
    ranking = []
    if rec_p.exists():
        payload = json.loads(rec_p.read_text(encoding="utf-8"))
        ranking = [r for r in payload.get("rule_ranking", []) if r.get("passes_consistency")]

    store = KlineStore(cfg.data_path, cfg.cache_path)
    rows = []
    for code, item in adopted.items():
        info = store.find(code, item.get("category"))
        if info is None:
            continue
        if only and (only not in code and only not in info.name):
            continue
        sig, ctx = _latest_signals(store, info)
        voted = []
        score = 0.0
        for r in item["rules"]:
            s = sig.get(r["rule"], 0)
            if s != 0:
                voted.append(f"{r['rule'].split('__')[-1]}({'多' if s > 0 else '空'},H{r['horizon']})")
                score += s * float(r["edge_hold"])
        rows.append({
            "code": code, "name": info.name, "category": info.category,
            "date": ctx["date"], "close": ctx["close"], "chg": ctx["chg"],
            "rsi14": ctx["rsi14"], "pos60": ctx["pos60"],
            "n_rules": len(item["rules"]), "n_vote": len(voted),
            "score": score, "votes": " ".join(voted) if voted else "",
        })

    if not rows:
        print("没有匹配的标的。")
        return 0

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(f"数据截止 {df['date'].iloc[0]}    已确认规则池：{len(adopted)} 个标的")
    print("说明：score = Σ 信号 × 该规则留出段优势；只是排序参考，不是预期收益\n")

    show = df.head(top)
    print(f"{'代码':<8}{'名称':<10}{'收盘':>9}{'涨跌':>8}{'RSI14':>7}{'60日位置':>9}{'规则':>5}{'信号':>5}  {'方向':<6} 触发规则")
    print("-" * 120)
    for r in show.itertuples():
        direction = "偏多" if r.score > 0 else ("偏空" if r.score < 0 else "中性")
        chg = f"{r.chg:+.2f}%" if r.chg == r.chg else "—"
        rsi = f"{r.rsi14:.0f}" if r.rsi14 == r.rsi14 else "—"
        pos = f"{r.pos60*100:.0f}%" if r.pos60 == r.pos60 else "—"
        print(f"{r.code:<8}{r.name:<10}{r.close:>9.2f}{chg:>8}{rsi:>7}{pos:>9}"
              f"{r.n_rules:>5}{r.n_vote:>5}  {direction:<6} {r.votes}")

    out = scan / "today_signals.json"
    df.to_json(out, orient="records", force_ascii=False, indent=1)
    print(f"\n明细 -> {out}")

    if ranking:
        print("\n规则级结论（通过验收线 A），及其今日在池子里的触发情况：")
        for r in ranking:
            key, h = r["rule"], r["horizon"]
            long_n = sum(1 for c, it in adopted.items()
                         if any(x["rule"] == key and x["horizon"] == h and x["today_signal"] > 0
                                for x in it["rules"]))
            short_n = sum(1 for c, it in adopted.items()
                          if any(x["rule"] == key and x["horizon"] == h and x["today_signal"] < 0
                                 for x in it["rules"]))
            print(f"  {key} H{h}: 今日 {long_n} 个标的看多 / {short_n} 个看空"
                  f"   （留出段 +{r['market_mean_edge']*100:.2f}pp, q={r['market_q']:.3f}）")
    return 0
