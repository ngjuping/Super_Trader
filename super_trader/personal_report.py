"""个性化规则报告：每个标的自己的规律（HTML + Markdown）。

和市场级报告分开，因为两者的证据等级不同：
市场级问「这是不是全市场规律」，个性化问「这个标的自己有没有可复现的偏好」。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .report import _fmt_p, _fmt_pct, _fmt_pp


def _fold_bars(edges, height: int = 24) -> str:
    """四段留出期的方向优势画成小柱状图（正绿负红）。"""
    vals = [e for e in (edges or []) if e is not None]
    if not vals:
        return "<span class='muted'>—</span>"
    mx = max(0.01, max(abs(v) for v in vals))
    bars = []
    for v in vals:
        h = max(2.0, abs(v) / mx * height)
        color = "#047857" if v > 0 else "#b91c1c"
        top = (height - h) if v > 0 else height
        bars.append(
            f"<span style='display:inline-block;width:9px;height:{height}px;position:relative;"
            f"margin-right:2px;vertical-align:top'>"
            f"<span style='position:absolute;left:0;right:0;top:{top}px;height:{h}px;"
            f"background:{color};border-radius:1px;opacity:.85'></span></span>"
        )
    label = " ".join(f"{v*100:+.0f}" for v in vals)
    return (f"<span title='{label} pp'>{''.join(bars)}</span> "
            f"<span class='muted' style='font-size:11px'>{label}</span>")


def _fmt_num(x, digits=1, default="—"):
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return f"{v:.{digits}f}" if np.isfinite(v) else default


def build_personal_report(payload: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_sym = payload.get("per_symbol_best") or []
    all_pairs = payload.get("by_symbol") or []
    watch = payload.get("watchlist") or []
    totals = payload.get("totals") or {}
    gate = payload.get("gate") or {}

    n_sym = payload.get("symbols", 0)
    cover = totals.get("validated_symbols", 0) or 0
    plc_pairs = totals.get("placebo_pairs")
    plc_syms = totals.get("placebo_symbols")
    ratio = totals.get("ratio")
    gain = totals.get("gain_mean")
    gain_plc = totals.get("gain_placebo")
    gain_pos = totals.get("gain_positive", 0)

    fam_count: dict[str, int] = {}
    hz_count: dict[int, int] = {}
    for r in all_pairs:
        fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
        hz_count[int(r["horizon"])] = hz_count.get(int(r["horizon"]), 0) + 1
    fam_str = "、".join(f"{k} {v}" for k, v in sorted(fam_count.items(), key=lambda x: -x[1])) or "—"
    hz_str = "、".join(f"H{k} {v}" for k, v in sorted(hz_count.items())) or "—"

    def row_html(r: dict, show_gain: bool = True) -> str:
        g = r.get("gain")
        g_cls = "pos" if (g is not None and g > 0) else "neg"
        g_txt = _fmt_pp(g) if g is not None else "—"
        spec = _fmt_pp(r.get("specificity")) if r.get("specificity") is not None else "—"
        mkt = r.get("market_best_rule") or "—"
        return f"""<tr>
          <td class="mono">{r['code']}<br><span class="muted">{r['name']}</span></td>
          <td><span class="tag">{r['family']}</span><br><span class="mono" style="font-size:11.5px">{r['rule'].split('__')[-1]}</span></td>
          <td class="num">H{int(r['horizon'])}</td>
          <td class="num pos"><b>{_fmt_pp(r['edge_hold'])}</b></td>
          <td class="num">{spec}</td>
          <td>{_fold_bars(r.get('fold_edges'))}</td>
          <td class="num">{_fmt_pct(r['acc_hold'])} / {_fmt_pct(r['base_hold'])}</td>
          <td class="num">{_fmt_pct(r['trigger_rate'], 1)}</td>
          <td class="num">{_fmt_p(r['p_hold'])}</td>
          {f'<td class="num" style="font-size:11.5px">{mkt}</td><td class="num {g_cls}"><b>{g_txt}</b></td>' if show_gain else ''}
        </tr>"""

    head_common = """<th>标的</th><th>规则</th><th class="num">周期</th>
      <th class="num">留出段优势</th><th class="num">特异性</th><th>四段复现</th>
      <th class="num">命中率/基准</th><th class="num">触发率</th><th class="num">p</th>"""
    head_gain = head_common + """<th class="num">通用规则最好</th><th class="num">个性化增益</th>"""

    body = "".join(row_html(r) for r in per_sym) or "<tr><td colspan='11' class='empty'>没有标的通过验证</td></tr>"
    pairs_body = "".join(row_html(r, show_gain=False) for r in all_pairs[:80]) or \
        "<tr><td colspan='9' class='empty'>无</td></tr>"
    watch_body = "".join(row_html(r, show_gain=False) for r in watch[:40]) or \
        "<tr><td colspan='9' class='empty'>无</td></tr>"

    ratio_ok = (ratio or 0) >= 3
    gain_ok = gain is not None and gain_plc is not None and (gain - gain_plc) > 0.002

    idio = [r for r in per_sym
            if (r.get("gain") or -1) > 0 and (r.get("specificity") or -1) > 0]
    idio.sort(key=lambda r: -(r.get("gain") or 0))
    idio_chips = "".join(
        f"<span class='chip'><b>{r['name']}</b> {r['rule'].split('__')[-1]} H{int(r['horizon'])} "
        f"<span class='pos'>{_fmt_pp(r.get('gain'))}</span></span>"
        for r in idio[:12]) or "<span class='muted'>无</span>"
    n_idio = len(idio)

    gain_para = (
        f"真实数据上，每个标的挑出自己最优的规则后，平均比「只用 5 条通用规则里的最好那条」多 "
        f"<b>{_fmt_pp(gain)}</b>。但换一组随机标签、用<b>同一套挑最优的流程</b>，也能挑出 "
        f"<b>{_fmt_pp(gain_plc)}</b> 的「优势」——这部分不是规律，而是「从 390 个假设里挑最大」"
        f"必然带来的偏差（挑得越多，最大值越好看）。"
        f"真实增益比这个偏差还小，说明<b>对多数标的而言，通用规则已经拿走了大部分可解释的边际</b>，"
        f"个性化挑选更多是在不同噪声之间换手。"
    )
    idio_para = (
        f"真正值得单独对待的是 <b>{n_idio}</b> 个标的：它们的个性化规则既优于通用规则（增益 &gt; 0），"
        f"又确实「这个标的格外适合这条规则」（特异性 &gt; 0）。它们分布在均值回归、日历、趋势三类，"
        f"说明个体差异不是「所有标的都偏某一家」，而是各有所长。"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Super_Trader · 个性化规则</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#14181f;--muted:#6b7280;--line:#e5e7eb;
--accent:#1f3a8a;--accent2:#0f766e;--pos:#047857;--neg:#b91c1c;--warn:#92400e}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}}
header{{background:linear-gradient(160deg,#101a33 0%,#1f3a8a 60%,#2b5aa8 100%);color:#fff;padding:36px 40px 30px}}
header .wrap,main{{max-width:1240px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;opacity:.75}}
h1{{margin:8px 0 6px;font-size:30px;font-weight:700}}
header p{{margin:0;opacity:.85;max-width:820px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:26px}}
.kpi{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16);border-radius:12px;padding:14px 16px}}
.kpi-k{{font-size:11px;letter-spacing:.12em;opacity:.75;text-transform:uppercase}}
.kpi-v{{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}}
.kpi-d{{font-size:12px;opacity:.72}}
main{{padding:30px 40px 80px}}
section{{margin:0 0 32px}}
h2{{font-size:19px;margin:0 0 12px;display:flex;align-items:center;gap:10px}}
h2 .n{{display:inline-flex;width:26px;height:26px;border-radius:8px;background:var(--accent);color:#fff;
font-size:13px;align-items:center;justify-content:center;font-weight:700}}
.lead{{color:var(--muted);margin:-4px 0 14px;max-width:960px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
border-radius:12px;overflow:hidden;font-size:13px;box-shadow:0 1px 2px rgba(16,24,40,.06)}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{background:#fafbfc;font-weight:600;font-size:11.5px;color:#374151;white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
tbody tr:hover{{background:#fafbff}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.muted{{color:var(--muted)}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
.tag{{font-size:11px;background:#eef2ff;color:var(--accent);padding:2px 8px;border-radius:999px}}
.callout{{border-left:4px solid var(--accent2);background:#f0fdfa;padding:14px 16px;border-radius:0 12px 12px 0;margin:14px 0}}
.callout.warn{{border-left-color:var(--warn);background:#fffbeb}}
.callout h4{{margin:0 0 6px;font-size:14px}}
.callout p{{margin:4px 0;font-size:13.5px;color:#374151}}
.empty{{color:var(--muted);text-align:center;padding:16px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}
.chip{{font-size:12px;background:#fff;border:1px solid var(--line);border-radius:999px;padding:3px 10px}}
.foot{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;margin-top:26px}}
.scroll{{max-height:620px;overflow:auto;border-radius:12px}}
</style></head>
<body>
<header><div class="wrap">
  <div class="eyebrow">Super_Trader · 第二阶段 · 研究工具</div>
  <h1>个性化规则：每个标的、每个周期，各找各的规律</h1>
  <p>不再是「一条规则打天下」。每个标的独立选规则，但必须通过留出期的
  <b>4 段连续复现</b>（每一段都同向为正）才算数，再用同样的流程在随机标签上量出假阳性率。</p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-k">通过验证</div><div class="kpi-v">{cover} / {n_sym}</div><div class="kpi-d">标的数（每标的 1 条主规则）</div></div>
    <div class="kpi"><div class="kpi-k">规则-标的对</div><div class="kpi-v">{totals.get('validated_pairs', 0)}</div><div class="kpi-d">{totals.get('distinct_rules', 0)} 条不同规则 / {payload.get('tests', 0):,} 次检验</div></div>
    <div class="kpi"><div class="kpi-k">安慰剂</div><div class="kpi-v">{_fmt_num(plc_pairs)}</div><div class="kpi-d">随机标签下的自我「验证」条数（{_fmt_num(plc_syms)} 个标的）</div></div>
    <div class="kpi"><div class="kpi-k">个性化增益</div><div class="kpi-v">{_fmt_pp(gain)}</div><div class="kpi-d">相对通用规则，安慰剂 {_fmt_pp(gain_plc)}</div></div>
  </div>
</div></header>
<main>
  <section>
    <h2><span class="n">1</span>每个标的的主规则</h2>
    <p class="lead">「留出段优势」= 方向命中率 − 该标的自身无条件命中率。
    「特异性」= 本标的的优势 − 同一规则在全部标的上的平均优势，&gt;0 才说明这个标的格外适合它。
    「个性化增益」= 本标的的个性化最优 − 通用规则在本标的上能拿到的最好。
    中间四根小柱是留出期切成 4 段后每段的方向优势，<b>四段全绿才算通过</b>。</p>
    <table>
      <thead><tr>{head_gain}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    <p class="muted" style="font-size:12.5px;margin-top:8px">
      家族分布：{fam_str}；周期分布：{hz_str}（H1 样本最多，天然更容易过「每段≥20 次」这一关，存在样本量偏向）。
    </p>
  </section>

  <section>
    <h2><span class="n">2</span>个性化到底比「一条规则打天下」多赚了多少</h2>
    <div class="callout {'warn' if not gain_ok else ''}">
      <h4>{'个性化确实带来了额外优势' if gain_ok else '个性化增益没有明显超过「挑最大」的偏差'}</h4>
      <p>{gain_para}</p>
      <p>{idio_para}</p>
      <div class="chips">{idio_chips}</div>
    </div>
    <div class="callout">
      <h4>真实信号 vs 噪声</h4>
      <p>{totals.get('validated_pairs', 0)} 条规则-标的对通过四段复现，随机标签下平均只有
      {_fmt_num(plc_pairs)} 条（{_fmt_num(plc_syms)} 个标的），信噪比 <b>{_fmt_num(ratio)}×</b>。
      {('个体差异确实存在：不是「某个标的运气好」，而是一批标的各自有能连续穿越 4 个时间段的偏好。'
        if ratio_ok else '信噪比不足 3 倍，验证结果需要更谨慎对待。')}</p>
    </div>
  </section>

  <section>
    <h2><span class="n">3</span>全部通过验证的规则-标的对</h2>
    <p class="lead">同一个标的可能有多条规则各自通过了四段复现（例如既有月末效应又有 RSI 超跌），
    它们不是互斥关系，可以叠加成一套该标的的规则组。</p>
    <div class="scroll">
      <table>
        <thead><tr>{head_common}</tr></thead>
        <tbody>{pairs_body}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2><span class="n">4</span>候选级（3/4 段复现，未达全部门槛）</h2>
    <p class="lead">方向一致但有一段时间掉头，或者样本略薄。列为观察名单，不当作结论。</p>
    <table>
      <thead><tr>{head_common}</tr></thead>
      <tbody>{watch_body}</tbody>
    </table>
  </section>

  <section>
    <h2><span class="n">5</span>门槛与局限</h2>
    <div class="callout">
      <h4>通过验证需要同时满足</h4>
      <p>训练段触发 ≥{gate.get('min_train')} 次、方向优势 &gt; 0、p ≤ {gate.get('train_p_max')}；
      留出段触发 ≥{gate.get('min_holdout')} 次、优势 &gt; {gate.get('min_edge')}；
      留出期切成 {gate.get('n_folds')} 段、每段触发 ≥{gate.get('min_fold')} 次且
      {gate.get('min_folds_positive')}/{gate.get('n_folds')} 段方向为正；留出段整体 p ≤ {gate.get('holdout_p_max')}。</p>
    </div>
    <div class="callout warn">
      <h4>仍然不能直接拿来交易</h4>
      <p>没有计算交易成本与冲击成本；没有处理涨停买不进；样本期仍是同一段历史，
      不存在真正的未来验证。个性化规则的样本量天然比市场级小，结论更脆，
      必须靠后续新数据做前瞻跟踪才能升级。</p>
    </div>
  </section>

  <div class="foot">
    生成于 {payload.get('generated_at', '')} · 标的池 {payload.get('pool')} ·
    {payload.get('symbols')} 个标的 / {payload.get('rules')} 条规则 / 周期 {payload.get('horizons')}。
    通用规则取自市场级扫描的 {", ".join(payload.get('accepted_market_pairs') or ["（无）"])}。
    仅供研究，不构成投资建议。
  </div>
</main></body></html>"""

    html_path = out / "report_personalized.html"
    html_path.write_text(html, encoding="utf-8")
    md_path = out / "PERSONALIZED.md"
    md_path.write_text(_render_md(payload), encoding="utf-8")
    return [html_path, md_path]


def _render_md(payload: dict) -> str:
    per_sym = payload.get("per_symbol_best") or []
    totals = payload.get("totals") or {}
    lines = ["# 个性化规则：每个标的自己的规律\n"]
    lines.append(f"- 生成时间：{payload.get('generated_at')}")
    lines.append(f"- 标的池：{payload.get('pool')}（{payload.get('symbols')} 个）")
    lines.append(f"- 检验总数：{payload.get('tests'):,}")
    lines.append(f"- 通过验证：{totals.get('validated_pairs')} 条规则-标的对 / "
                 f"{totals.get('validated_symbols')} 个标的 / {totals.get('distinct_rules')} 条不同规则")
    lines.append(f"- 安慰剂：平均 {_fmt_num(totals.get('placebo_pairs'))} 条 / "
                 f"{_fmt_num(totals.get('placebo_symbols'))} 个标的，信噪比 {_fmt_num(totals.get('ratio'))}×")
    lines.append(f"- 个性化增益：{_fmt_pp(totals.get('gain_mean'))}"
                 f"（安慰剂 {_fmt_pp(totals.get('gain_placebo'))}，"
                 f"{totals.get('gain_positive')}/{len(per_sym)} 个标的为正）\n")
    lines.append("## 每个标的的主规则\n")
    if per_sym:
        lines.append("| 标的 | 规则 | 周期 | 留出段优势 | 特异性 | 四段复现 | 触发率 | 通用规则最好 | 增益 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in per_sym:
            fe = r.get("fold_edges") or []
            fold_s = " ".join(f"{x*100:+.0f}" if x is not None else "—" for x in fe)
            lines.append(
                f"| {r['name']}({r['code']}) | {r['rule']} | H{int(r['horizon'])} | "
                f"{_fmt_pp(r['edge_hold'])} | {_fmt_pp(r.get('specificity'))} | {fold_s} | "
                f"{_fmt_pct(r['trigger_rate'], 1)} | {r.get('market_best_rule') or '—'} | "
                f"{_fmt_pp(r.get('gain'))} |")
    else:
        lines.append("没有标的通过 4 段复现验证。\n")
    return "\n".join(lines)
