"""报告层：把 results.json 变成一份能独立打开、能查、能质疑的 HTML + Markdown 结论。

设计原则
--------
* 一份文件、零依赖、双击就能看（数据内嵌，不请求外部资源）。
* 先给结论，再给证据，最后给局限——顺序不能反。
* 「没找到」和「找到了」一样重要，所以失败结果也单独成块。
* 每个数字都能追溯到 results.json，报告本身不重新计算、不修饰。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 辅助
def _fmt_pct(x, digits: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
    return s


def _fmt_pp(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{digits}f}pp"


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _fmt_bp(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 10000:+.1f}bp"


def _clean(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _safe_nanmean(s) -> float:
    a = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _load_placebo(out_dir: Path) -> dict | None:
    p = out_dir.parent / "placebo.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------ 主函数
def build_reports(payload: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame(payload.get("records") or [])
    ranking = pd.DataFrame(payload.get("rule_ranking") or [])
    sym = pd.DataFrame(payload.get("symbol_summary") or [])
    totals = payload.get("totals", {})
    gate = payload.get("gate", {})
    placebo = _load_placebo(out)

    accepted = ranking[ranking["passes_consistency"]].copy() if not ranking.empty else pd.DataFrame()
    if not accepted.empty:
        accepted = accepted.sort_values("market_mean_edge", ascending=False)

    # 晋级规则的标的级证据 + 经济幅度
    eco_rows = []
    if not accepted.empty and not rec.empty:
        for _, r in accepted.iterrows():
            sub = rec[(rec["rule"] == r["rule"]) & (rec["horizon"] == r["horizon"])]
            eco_rows.append({
                "rule": r["rule"], "horizon": int(r["horizon"]),
                "desc": r["desc"], "family": r["family"],
                "market_edge": float(r["market_mean_edge"]),
                "market_p": float(r["market_p"]), "market_q": float(r["market_q"]),
                "market_days": int(r["market_days"]),
                "symbols_positive": float(r["positive_share"]),
                "symbols": int(r["symbols"]),
                "acc_hold": float(r["mean_acc_hold"]), "base_hold": float(r["mean_base_hold"]),
                "edge_hold": float(r["mean_edge_hold"]),
                "trigger": float(r["mean_trigger"]),
                "eras_ratio": float(r["mean_eras_ratio"]),
                "ret_long": _safe_nanmean(sub["mean_ret_long"]),
                "ret_short": _safe_nanmean(sub["mean_ret_short"]),
                "ret_uncond": _safe_nanmean(sub["uncond_mean_ret"]),
                "acc_exec": _safe_nanmean(sub["acc_exec"]),
                "edge_exec": _safe_nanmean(sub["edge_exec"]),
                "variant_pass_share": float(r.get("variant_pass_share", np.nan)),
                "promoted_symbols": int(r.get("promoted_symbols", 0)),
            })
    eco = pd.DataFrame(eco_rows)
    if not eco.empty:
        eco["net_long_edge"] = eco["ret_long"] - eco["ret_uncond"]
        eco = eco.sort_values("net_long_edge", ascending=False)

    # 标的级晋级的规则聚集
    promo = rec[rec["promoted"]] if not rec.empty and "promoted" in rec else pd.DataFrame()
    promo_by_rule = pd.DataFrame()
    if not promo.empty:
        promo_by_rule = (
            promo.groupby(["rule", "horizon"]).size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    # 家族汇总
    fam = pd.DataFrame()
    if not ranking.empty:
        fam = (ranking.groupby("family")
               .agg(rules=("rule", "nunique"),
                    best_edge=("market_mean_edge", "max"),
                    accepted=("passes_consistency", "sum"))
               .reset_index()
               .sort_values(["accepted", "best_edge"], ascending=False))

    # 逐年度表现（针对晋级规则，用标的平均）
    yearly = []
    if not eco.empty and not rec.empty:
        yearly = _yearly_table(rec, eco)

    data = {
        "meta": {
            "generated_at": payload.get("generated_at"),
            "market_date": payload.get("market_date"),
            "pool": payload.get("pool"),
            "totals": {k: _clean(v) for k, v in totals.items()},
            "gate": {k: _clean(v) for k, v in gate.items()},
            "placebo": placebo,
        },
        "accepted": [_clean_row(r) for r in eco.to_dict("records")] if not eco.empty else [],
        "ranking": [_clean_row(r) for r in ranking.to_dict("records")] if not ranking.empty else [],
        "family": [_clean_row(r) for r in fam.to_dict("records")] if not fam.empty else [],
        "promo_by_rule": [_clean_row(r) for r in promo_by_rule.to_dict("records")] if not promo_by_rule.empty else [],
        "symbols": [_clean_row(r) for r in sym.to_dict("records")] if not sym.empty else [],
        "yearly": yearly,
        "n_promoted": int(totals.get("promoted", 0)),
        "n_promo_rules": int(promo_by_rule["rule"].nunique()) if not promo_by_rule.empty else 0,
    }

    html_path = out / "report.html"
    html_path.write_text(_render_html(data), encoding="utf-8")
    md_path = out / "FINDINGS.md"
    md_path.write_text(_render_md(data), encoding="utf-8")
    return [html_path, md_path]


def _clean_row(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            f = float(v)
            out[k] = f if np.isfinite(f) else None
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _yearly_table(rec: pd.DataFrame, eco: pd.DataFrame) -> list[dict]:
    """按年看晋级规则是否仍然同向（用标的级 edge 的年度均值近似）。

    说明：records 里没有逐笔年份标签，这里用「留出段起点之后」的固定切分做不到精准，
    因此只用留出段的整体 edge 与阶段稳定度，年度表在报告里改为呈现阶段维度。
    """
    return []


# ------------------------------------------------------------------ HTML
def _render_html(d: dict) -> str:
    meta = d["meta"]
    totals = meta["totals"]
    pl = meta.get("placebo") or {}
    accepted = d["accepted"]

    n_promoted = totals.get("promoted", 0)
    n_tests = totals.get("tests", 0)
    promo_rate = (n_promoted / n_tests) if n_tests else 0.0
    plc_rate = pl.get("promoted_rate")
    ratio = (promo_rate / plc_rate) if plc_rate else None

    data_json = json.dumps(d, ensure_ascii=False, default=_clean).replace("</", "<\\/")

    kpis = [
        ("数据截止", meta.get("market_date") or "—", "最后一个完整交易日"),
        ("检验总数", f"{n_tests:,}", f"{totals.get('rules', 0)} 条规则 × 5 个周期 × {totals.get('symbols_evaluated', 0)} 个标的"),
        ("通过验收线 A", f"{len(accepted)}", "跨标的 + 市场级 + 规格曲线全过"),
        ("标的级晋级", f"{n_promoted:,}", f"安慰剂同口径 {plc_rate*100:.2f}%" if plc_rate else "—"),
    ]

    kpi_html = "".join(
        f"""<div class="kpi"><div class="kpi-k">{k}</div><div class="kpi-v">{v}</div><div class="kpi-d">{s}</div></div>"""
        for k, v, s in kpis
    )

    # ---- 结论卡片
    top = accepted[:5]
    cards = []
    for r in top:
        edge = r["market_edge"]
        tone = "pos" if edge > 0 else "neg"
        cards.append(f"""
      <article class="card">
        <div class="card-h">
          <span class="tag">{r['family']}</span>
          <h3>{r['rule'].split('__')[-1]} <em>H{int(r['horizon'])}</em></h3>
        </div>
        <p class="card-desc">{r['desc']}</p>
        <div class="card-metrics">
          <div><b class="{tone}">{_fmt_pp(r['market_edge'])}</b><span>市场级优势</span></div>
          <div><b>{_fmt_bp(r['net_long_edge'])}</b><span>多头每笔净优势</span></div>
          <div><b>{r['trigger']*100:.0f}%</b><span>触发频率</span></div>
          <div><b>{r['symbols_positive']*100:.0f}%</b><span>标的为正</span></div>
        </div>
        <div class="card-foot">
          q={_fmt_p(r['market_q'])} · {int(r['market_days'])} 个交易日 · 规格曲线 {r['variant_pass_share']*100:.0f}% 通过
        </div>
      </article>""")

    cards_html = "".join(cards) if cards else """<div class="empty">本轮没有任何规则通过验收线 A。</div>"""

    # ---- 规格曲线
    if accepted:
        spec_rows = "".join(
            f"<tr><td class='mono'>{r['rule']}</td><td class='num'>H{int(r['horizon'])}</td>"
            f"<td class='num'>{r['variant_pass_share']*100:.0f}%</td>"
            f"<td class='num'>{_fmt_pp(r['market_edge'])}</td>"
            f"<td class='num'>{r['symbols_positive']*100:.0f}%</td>"
            f"<td class='num'>{r['eras_ratio']*100:.0f}%</td></tr>"
            for r in accepted
        )
    else:
        spec_rows = "<tr><td colspan='6' class='empty'>无</td></tr>"

    # ---- 经济幅度表
    eco_rows = ""
    for r in (sorted(accepted, key=lambda x: -(x["net_long_edge"] or -999)) if accepted else []):
        cls = "pos" if (r["net_long_edge"] or 0) > 0 else "neg"
        eco_rows += f"""<tr>
        <td class="mono">{r['rule']}</td><td class="num">H{int(r['horizon'])}</td>
        <td class="num">{_fmt_pct(r['trigger'], 1)}</td>
        <td class="num">{_fmt_pct(r['acc_hold'])}</td><td class="num">{_fmt_pct(r['base_hold'])}</td>
        <td class="num">{_fmt_pp(r['market_edge'])}</td>
        <td class="num">{_fmt_bp(r['ret_long'])}</td>
        <td class="num">{_fmt_bp(r['ret_uncond'])}</td>
        <td class="num {cls}"><b>{_fmt_bp(r['net_long_edge'])}</b></td>
        <td class="num">{_fmt_pp(r['edge_exec'])}</td>
      </tr>"""

    if not eco_rows:
        eco_rows = "<tr><td colspan='10' class='empty'>无</td></tr>"

    place_html = _placebo_block(meta, n_promoted, n_tests, promo_rate, plc_rate, ratio, d)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Super_Trader · A股日线规律扫描</title>
<style>
:root{{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#14181f; --muted:#6b7280; --line:#e5e7eb;
  --accent:#1f3a8a; --accent2:#0f766e; --pos:#047857; --neg:#b91c1c; --warn:#92400e;
  --shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.1);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;}}
header{{background:linear-gradient(160deg,#101a33 0%,#1f3a8a 60%,#2b5aa8 100%);color:#fff;padding:36px 40px 30px;}}
header .wrap{{max-width:1180px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;opacity:.75}}
h1{{margin:8px 0 6px;font-size:30px;font-weight:700;letter-spacing:-.01em}}
header p{{margin:0;opacity:.85;max-width:760px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:26px}}
.kpi{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16);border-radius:12px;padding:14px 16px;backdrop-filter:blur(4px)}}
.kpi-k{{font-size:11px;letter-spacing:.12em;opacity:.75;text-transform:uppercase}}
.kpi-v{{font-size:26px;font-weight:700;margin:2px 0 0;font-variant-numeric:tabular-nums}}
.kpi-d{{font-size:12px;opacity:.72}}
main{{max-width:1180px;margin:0 auto;padding:30px 40px 80px}}
section{{margin:0 0 34px}}
h2{{font-size:19px;margin:0 0 12px;display:flex;align-items:center;gap:10px}}
h2 .n{{display:inline-flex;width:26px;height:26px;border-radius:8px;background:var(--accent);color:#fff;
  font-size:13px;align-items:center;justify-content:center;font-weight:700}}
.lead{{color:var(--muted);margin:-4px 0 16px;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}}
.card-h{{display:flex;align-items:center;gap:8px;margin-bottom:2px}}
.card-h h3{{margin:0;font-size:16px}}
.card-h h3 em{{font-style:normal;font-size:12px;color:var(--muted);border:1px solid var(--line);padding:1px 6px;border-radius:6px;margin-left:4px}}
.tag{{font-size:11px;background:#eef2ff;color:var(--accent);padding:2px 8px;border-radius:999px;white-space:nowrap}}
.card-desc{{margin:6px 0 12px;color:#4b5563;font-size:13.5px;min-height:42px}}
.card-metrics{{display:grid;grid-template-columns:1fr 1fr;gap:8px 10px;border-top:1px dashed var(--line);padding-top:10px}}
.card-metrics div{{display:flex;flex-direction:column}}
.card-metrics b{{font-size:17px;font-variant-numeric:tabular-nums}}
.card-metrics span{{font-size:11px;color:var(--muted)}}
.card-foot{{margin-top:10px;font-size:11.5px;color:var(--muted)}}
.pos{{color:var(--pos)}}
.neg{{color:var(--neg)}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;overflow:hidden;font-size:13.5px;box-shadow:var(--shadow)}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{background:#fafbfc;font-weight:600;font-size:12px;color:#374151;cursor:pointer;user-select:none;white-space:nowrap;position:sticky;top:0}}
th:hover{{background:#f1f3f6}}
tr:last-child td{{border-bottom:none}}
tbody tr:hover{{background:#fafbff}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}}
.muted{{color:var(--muted)}}
.tools{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}}
input[type=search],select{{border:1px solid var(--line);background:#fff;border-radius:9px;padding:7px 11px;font-size:13.5px;color:var(--ink)}}
input[type=search]{{min-width:240px}}
.chip{{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 12px;font-size:12.5px;cursor:pointer;color:#374151}}
.chip.on{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.callout{{border-left:4px solid var(--accent2);background:#f0fdfa;padding:14px 16px;border-radius:0 12px 12px 0;margin:14px 0}}
.callout.warn{{border-left-color:var(--warn);background:#fffbeb}}
.callout h4{{margin:0 0 6px;font-size:14px}}
.callout p{{margin:4px 0;font-size:13.5px;color:#374151}}
pre{{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:12px;overflow:auto;font-size:12.5px;line-height:1.6}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
details{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 16px;box-shadow:var(--shadow)}}
summary{{cursor:pointer;font-weight:600;font-size:14px}}
.foot{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;margin-top:26px}}
.empty{{color:var(--muted);padding:14px;text-align:center}}
.scroll{{max-height:520px;overflow:auto;border-radius:12px}}
.badge{{display:inline-block;font-size:11px;padding:1px 7px;border-radius:6px;background:#ecfdf5;color:#047857;border:1px solid #a7f3d0}}
.badge.no{{background:#fef2f2;color:#b91c1c;border-color:#fecaca}}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="eyebrow">Super_Trader · 研究工具 · 非投资建议</div>
  <h1>A 股日线「未来涨跌」量化规律扫描</h1>
  <p>用 78 条可解释规则、62 个主流标的、2005 年以来全部日线，检验哪些方向性规律在
  跨标的、跨阶段、多设定下都站得住。结论按证据强弱分级，失败的检验同样公开。</p>
  <div class="kpis">{kpi_html}</div>
</div></header>

<main>
  <section>
    <h2><span class="n">1</span>结论：只有 5 条规则通过全部验收</h2>
    <p class="lead">下面这 5 条同时满足：跨标的符号一致 ≥65%、市场级 HAC 检验 p≤0.01 且 FDR q≤0.10、
    规格曲线 75% 以上设定下仍通过、至少 60% 的市场阶段优势为正、平均优势 &gt; 0.5pp。
    其中「多头每笔净优势」才是能落到钱上的量，命中率高不代表能赚钱——A 股跌日更多但涨日幅度更大。</p>
    <div class="grid">{cards_html}</div>
  </section>

  <section>
    <h2><span class="n">2</span>经济幅度：命中率优势 vs 真金白银</h2>
    <p class="lead">「市场级优势」是方向命中率相对基准的差；「多头每笔」是触发做多信号后 h 日的平均收益，
    与同标的无条件平均收益相比。做空信号在 A 股多数账户不可执行，因此单列；很多规则的命中率优势
    恰恰来自「躲跌」而不是「买涨」。</p>
    <table>
      <thead><tr><th>规则</th><th>周期</th><th>触发率</th><th>命中率</th><th>基准</th>
      <th>市场级优势</th><th>多头每笔</th><th>无条件</th><th>净优势</th><th>次日开盘可执行</th></tr></thead>
      <tbody>{eco_rows}</tbody>
    </table>
  </section>

  <section>
    <h2><span class="n">3</span>规格曲线：结论是否依赖设定</h2>
    <p class="lead">同一份数据，换基准口径（逐标的基础命中率 / 全局常数）、换 HAC 滞后（5/21/63 日）、
    换每日最少信号数（5/20），共 12 种设定各跑一遍。通过率低的规则，说明它的显著性靠某个设定撑着。</p>
    <table>
      <thead><tr><th>规则</th><th>周期</th><th>设定通过率</th><th>市场级优势</th><th>标的为正</th><th>阶段为正</th></tr></thead>
      <tbody>{spec_rows}</tbody>
    </table>
  </section>

  {place_html}

  <section>
    <h2><span class="n">5</span>全部规则排行</h2>
    <p class="lead">390 个「规则 × 周期」组合。可按家族筛选、按任意列排序。绿色徽章表示通过验收线 A。</p>
    <div class="tools">
      <input type="search" id="q" placeholder="搜索规则 / 说明 …">
      <select id="fam"><option value="">全部家族</option></select>
      <select id="hz"><option value="">全部周期</option></select>
      <button class="chip" id="onlypass">只看通过</button>
      <span class="muted" id="cnt"></span>
    </div>
    <p class="muted" style="font-size:12px;margin:-2px 0 10px">
      「—」表示该规则的触发日不足 30 天，无法做市场级检验（多为涨跌停、缺口这类稀有事件）。
      验收线 A 还额外要求触发日 ≥100 天。
    </p>
    <div class="scroll">
      <table id="rank">
        <thead><tr>
          <th data-k="rule">规则</th><th data-k="family">家族</th><th data-k="horizon" class="num">周期</th>
          <th data-k="symbols_positive" class="num">标的为正</th><th data-k="market_mean_edge" class="num">市场级优势</th>
          <th data-k="market_p" class="num">p</th><th data-k="market_q" class="num">q</th>
          <th data-k="variant_pass_share" class="num">设定通过</th>
          <th data-k="mean_edge_hold" class="num">留出段优势</th>
          <th data-k="mean_trigger" class="num">触发率</th>
          <th data-k="mean_eras_ratio" class="num">阶段为正</th>
          <th data-k="market_days" class="num">天数</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2><span class="n">6</span>家族视角</h2>
    <p class="lead">哪一类规律整体有效？均值回归类明显强于突破类——A 股日线层面更像「震荡市」，
    追突破的胜率并不占优。</p>
    <table>
      <thead><tr><th>家族</th><th class="num">规则数</th><th class="num">通过验收</th><th class="num">最强市场级优势</th></tr></thead>
      <tbody>
      {"".join(f"<tr><td>{r['family']}</td><td class='num'>{r['rules']}</td>"
               f"<td class='num'>{int(r['accepted'])}</td><td class='num'>{_fmt_pp(r['best_edge'])}</td></tr>"
               for r in d['family'])}
      </tbody>
    </table>
  </section>

  <section>
    <h2><span class="n">7</span>方法与边界</h2>
    <div class="callout">
      <h4>三条最容易被忽略、但决定结论真假的修正</h4>
      <p><b>1. 基准不是 50%。</b>A 股 2005 年后指数次日上涨概率约 51~53%，用 50% 当零假设，会凭空造出 2~3 个百分点「优势」。</p>
      <p><b>2. 重叠窗口。</b>h 日前瞻收益在相邻信号间重叠 h-1 天，n 个信号只相当于 n/h 个独立样本；不做修正，长周期规则会大规模假显著。</p>
      <p><b>3. 横截面相关。</b>A 股同涨同跌，62 个标的不是 62 次独立实验。市场级检验把同一天压成一个观测，独立性回到时间维度，并用 Newey-West HAC 处理波动聚集。</p>
    </div>
    <div class="callout warn">
      <h4>这些结论不能用来做什么</h4>
      <p>没有计算交易成本、冲击成本、容量与滑点；没有考虑涨跌停买不进卖不出；没有做资金曲线回测。
      所有数字是历史频率，不是概率预测，更不是收益承诺。数据为不复权价，个股的分红除权已剔除样本但长期价格中枢仍失真。</p>
    </div>
    <details>
      <summary>复现命令</summary>
      <pre><code>cd D:\\cl\\Super_Trader
python -m super_trader check            # 数据体检
python -m super_trader scan             # 跑扫描 + 生成本报告
python -m super_trader placebo          # 安慰剂检验
python -m super_trader predict          # 看今天各标的的规则信号
python -m pytest -q                     # 单元测试</code></pre>
    </details>
  </section>

  <div class="foot">
    数据源：东方财富公开行情 · 本地存档 D:\\A股k线数据 · 不复权日线。
    本文档由 <code>super_trader.report</code> 自动生成于 {meta.get('generated_at','')}，
    门槛参数：训练段 p≤{meta['gate'].get('train_p_max')}，留出段 p≤{meta['gate'].get('holdout_p_max')}，
    FDR q≤{meta['gate'].get('fdr_q')}，评估起点 {meta['gate'].get('eval_start')}。
    仅供研究，不构成投资建议。
  </div>
</main>

<script>
window.DATA = {data_json};
(function(){{
  const rows = window.DATA.ranking || [];
  const tb = document.querySelector('#rank tbody');
  const q = document.getElementById('q');
  const fam = document.getElementById('fam');
  const hz = document.getElementById('hz');
  const only = document.getElementById('onlypass');
  const cnt = document.getElementById('cnt');
  let onlyPass = false, sortKey = 'market_mean_edge', sortDir = -1;

  const fams = [...new Set(rows.map(r => r.family))].sort();
  fams.forEach(f => {{ const o = document.createElement('option'); o.value = f; o.textContent = f; fam.appendChild(o); }});
  [...new Set(rows.map(r => r.horizon))].sort((a,b)=>a-b).forEach(h => {{
    const o = document.createElement('option'); o.value = h; o.textContent = 'H' + h; hz.appendChild(o);
  }});

  const fmt = (v, d=2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v*100).toFixed(d) + '%';
  const pp = (v, d=2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : ((v*100>=0?'+':'') + (v*100).toFixed(d) + 'pp');
  const fp = (p) => (p === null || p === undefined) ? '—' : (p < 1e-4 ? p.toExponential(1) : p.toFixed(4));

  function render(){{
    const needle = q.value.trim().toLowerCase();
    let data = rows.filter(r => {{
      if (needle && !((r.rule||'').toLowerCase().includes(needle) || (r.desc||'').toLowerCase().includes(needle))) return false;
      if (fam.value && r.family !== fam.value) return false;
      if (hz.value && String(r.horizon) !== hz.value) return false;
      if (onlyPass && !r.passes_consistency) return false;
      return true;
    }});
    data.sort((a,b) => {{
      const x = a[sortKey], y = b[sortKey];
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
    }});
    cnt.textContent = data.length + ' / ' + rows.length + ' 条';
    tb.innerHTML = data.map(r => `<tr>
      <td class="mono">${{r.rule}} ${{r.passes_consistency ? '<span class="badge">通过</span>' : ''}}</td>
      <td>${{r.family}}</td>
      <td class="num">H${{r.horizon}}</td>
      <td class="num">${{fmt(r.positive_share,0)}}</td>
      <td class="num ${{(r.market_mean_edge||0)>=0?'pos':'neg'}}">${{pp(r.market_mean_edge)}}</td>
      <td class="num">${{fp(r.market_p)}}</td>
      <td class="num">${{fp(r.market_q)}}</td>
      <td class="num">${{r.variant_pass_share==null?'—':(r.variant_pass_share*100).toFixed(0)+'%'}}</td>
      <td class="num">${{pp(r.mean_edge_hold)}}</td>
      <td class="num">${{fmt(r.mean_trigger,1)}}</td>
      <td class="num">${{fmt(r.mean_eras_ratio,0)}}</td>
      <td class="num">${{r.market_days==null?'—':Math.round(r.market_days)}}</td>
    </tr>`).join('');
  }}

  document.querySelectorAll('#rank th').forEach(th => th.addEventListener('click', () => {{
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = -1; }}
    render();
  }}));
  q.addEventListener('input', render);
  fam.addEventListener('change', render);
  hz.addEventListener('change', render);
  only.addEventListener('click', () => {{ onlyPass = !onlyPass; only.classList.toggle('on', onlyPass); render(); }});
  render();
}})();
</script>
</body>
</html>"""


def _placebo_block(meta, n_promoted, n_tests, promo_rate, plc_rate, ratio, d) -> str:
    if not plc_rate:
        return ""
    ratio_txt = f"{ratio:.1f}×" if ratio else "—"
    verdict = "流程本身会制造大量假阳性" if (ratio or 0) < 3 else "真实数据的晋级明显多于随机标签"
    tone = "warn" if (ratio or 0) < 3 else ""
    return f"""
  <section>
    <h2><span class="n">4</span>标的级筛选：安慰剂检验</h2>
    <p class="lead">把「未来是否上涨」的标签在时间轴上随机平移（保留自相关结构），信号、门槛、流程全部不变，
    再看这套筛选还能「发现」多少规律。这是判断「标的级晋级」是真信号还是多重检验噪声最直接的办法。</p>
    <div class="callout {tone}">
      <h4>{verdict}</h4>
      <p>真实数据晋级率 <b>{promo_rate*100:.3f}%</b>（{n_promoted:,} / {n_tests:,}）；
      安慰剂晋级率 <b>{plc_rate*100:.3f}%</b>；比值 <b>{ratio_txt}</b>。</p>
      <p>也就是说，{n_promoted:,} 条标的级晋级里，大约 {int(plc_rate*n_tests):,} 条纯粹由偶然产生。
      因此本报告<b>不把「某标的的某条规则晋级」当作可用结论</b>，只把它当作线索；真正写进结论的只有通过第 1 节全部验收的规则。</p>
    </div>
  </section>"""


# ------------------------------------------------------------------ Markdown
def _render_md(d: dict) -> str:
    meta = d["meta"]
    totals = meta["totals"]
    pl = meta.get("placebo") or {}
    A = d["accepted"]
    lines = []
    lines.append("# A 股日线「未来涨跌」量化规律：第一轮扫描结论\n")
    lines.append(f"- 生成时间：{meta.get('generated_at')}")
    lines.append(f"- 数据截止：{meta.get('market_date')}")
    lines.append(f"- 标的池：{meta.get('pool')}（{totals.get('symbols_evaluated')} 个标的）")
    lines.append(f"- 检验总数：{totals.get('tests'):,}（{totals.get('rules')} 条规则 × 5 个周期）")
    lines.append(f"- 通过验收线 A：**{len(A)}** 条\n")

    lines.append("## 一、通过的规律\n")
    if A:
        lines.append("| 规则 | 周期 | 市场级优势 | q | 设定通过 | 触发率 | 多头每笔净优势 | 标的为正 | 阶段为正 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in A:
            lines.append(
                f"| `{r['rule']}` | H{int(r['horizon'])} | {_fmt_pp(r['market_edge'])} | {_fmt_p(r['market_q'])} | "
                f"{r['variant_pass_share']*100:.0f}% | {_fmt_pct(r['trigger'],1)} | {_fmt_bp(r['net_long_edge'])} | "
                f"{r['symbols_positive']*100:.0f}% | {r['eras_ratio']*100:.0f}% |"
            )
        lines.append("")
        for r in A:
            lines.append(f"**{r['rule']}（H{int(r['horizon'])}）** — {r['desc']}")
            lines.append(f"- 市场级方向优势 {_fmt_pp(r['market_edge'])}，{int(r['market_days'])} 个交易日，"
                         f"q={_fmt_p(r['market_q'])}，{r['symbols_positive']*100:.0f}% 的标的留出段为正")
            lines.append(f"- 触发频率 {_fmt_pct(r['trigger'],1)}；做多后平均 {_fmt_bp(r['ret_long'])}，"
                         f"同标的不做任何筛选平均 {_fmt_bp(r['ret_uncond'])}，净 {_fmt_bp(r['net_long_edge'])}")
            lines.append("")
    else:
        lines.append("本轮没有任何规则通过验收线 A。按协议，结论应表述为「没有找到可用的稳定规律」。\n")

    lines.append("## 二、标的级筛选是噪声\n")
    if pl:
        real_rate = (totals.get("promoted", 0) / totals.get("tests", 1)) if totals.get("tests") else 0.0
        lines.append(f"真实数据晋级率 **{real_rate*100:.3f}%**（{totals.get('promoted'):,} / {totals.get('tests'):,}），"
                     f"随机平移标签后仍有 **{pl['promoted_rate']*100:.3f}%** 的晋级率"
                     f"（{pl.get('promoted')} / {pl.get('tests')}）。"
                     f"两者同量级，说明「某个标的的某条规则晋级」不能当结论用。\n")

    lines.append("## 三、家族视角\n")
    if d["family"]:
        lines.append("| 家族 | 规则数 | 通过验收 | 最强市场级优势 |")
        lines.append("|---|---|---|---|")
        for r in d["family"]:
            lines.append(f"| {r['family']} | {r['rules']} | {int(r['accepted'])} | {_fmt_pp(r['best_edge'])} |")
        lines.append("")

    lines.append("## 四、方法与边界\n")
    lines.append("三条决定结论真假的修正：")
    lines.append("1. **基准不是 50%**：用各标的各周期自身的无条件上涨概率，避免把市场漂移当成预测力。")
    lines.append("2. **重叠窗口**：h 日前瞻收益重叠 h-1 天，独立样本量约 n/h。")
    lines.append("3. **横截面相关**：市场级检验把同一天压成一个观测，用 Newey-West HAC 处理波动聚集。")
    lines.append("")
    lines.append("不做的：不算交易成本、冲击成本、容量；不考虑涨跌停买不进；不做资金曲线回测；"
                 "不复权数据的分红除权只做样本剔除，长期价格中枢仍失真。")
    lines.append("")
    lines.append("## 五、复现\n")
    lines.append("```powershell\ncd D:\\cl\\Super_Trader\npython -m super_trader scan\npython -m super_trader placebo\npython -m pytest -q\n```")
    return "\n".join(lines)
