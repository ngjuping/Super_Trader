"""横截面层报告：相对强弱、波动分层、行业轮动。"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


def _pct(x, d=2):
    try:
        v = float(x)
        return "—" if not np.isfinite(v) else f"{v * 100:.{d}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, d=3):
    try:
        v = float(x)
        return "—" if not np.isfinite(v) else f"{v:.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _tone(t, pos=2.0, neg=-2.0):
    if t is None or not np.isfinite(t):
        return ""
    return "pos" if t >= pos else ("neg" if t <= neg else "")


def build_xs_html(payload: dict, out_path: str | Path | None = None) -> Path:
    out = Path(out_path) if out_path else Path("crosssection.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = payload.get("results") or {}
    ic = res.get("ic") or []
    absolute = res.get("absolute") or []
    shape = res.get("panel_shape") or [0, 0]
    rng = res.get("date_range") or ["", ""]
    cfg = payload.get("config") or {}

    def row_ic(r):
        return f"""<tr>
          <td>{'相对强弱 RS' + str(r['feature']) if r['kind'] == 'relative_strength' else '波动率 20 日'}</td>
          <td class="num">H{r['horizon']}</td>
          <td class="num">{_num(r['ic_mean'])}</td>
          <td class="num {_tone(r['ic_t'])}"><b>{_num(r['ic_t'], 2)}</b></td>
          <td class="num">{_pct(r['ic_positive_share'], 1)}</td>
          <td class="num">{_pct(r['spread_mean'])}</td>
          <td class="num {_tone(r['spread_t'])}">{_num(r['spread_t'], 2)}</td>
          <td class="num muted">{r['days']:,}</td>
        </tr>"""

    def row_abs(r):
        edge = (r["top_up_rate"] - r["all_up_rate"]) if r["top_up_rate"] is not None else None
        return f"""<tr>
          <td>RS{r['lookback']}</td><td class="num">H{r['horizon']}</td>
          <td class="num">{_pct(r['top_up_rate'])}</td>
          <td class="num">{_pct(r['all_up_rate'])}</td>
          <td class="num">{_pct(r['bottom_up_rate'])}</td>
          <td class="num {'pos' if (edge or 0) > 0 else 'neg'}">{_pct(edge)}</td>
        </tr>"""

    best = max((r for r in ic if r["ic_t"] is not None and np.isfinite(r["ic_t"])),
               key=lambda r: abs(r["ic_t"]), default=None)
    headline = (f"最强信号：{best['feature']} / H{best['horizon']}，IC {_num(best['ic_mean'])}"
                f"，t = {_num(best['ic_t'], 2)}" if best else "无有效横截面结果")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super_Trader · 横截面：相对强弱与行业轮动</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--ink:#121a2b;--muted:#6b7280;--line:#e3e8ef;
--accent:#1e3a8a;--pos:#047857;--neg:#b91c1c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
header{{background:linear-gradient(150deg,#0d1b36,#1e3a8a 65%,#2f5bb7);color:#fff;padding:34px 40px 26px}}
header .in,main{{max-width:1240px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;opacity:.72}}
h1{{margin:8px 0 6px;font-size:28px}}
header p{{margin:0;opacity:.86;max-width:880px;font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:13px;margin-top:22px}}
.kpi{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.17);border-radius:12px;padding:13px 15px}}
.kpi-k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;opacity:.75}}
.kpi-v{{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums}}
.kpi-d{{font-size:12px;opacity:.72}}
main{{padding:24px 40px 60px}}
section{{margin-bottom:28px}}
h2{{font-size:18px;margin:0 0 10px}}
.lead{{color:var(--muted);font-size:13.5px;margin:-2px 0 12px;max-width:1000px}}
table{{width=100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
border-radius:12px;overflow:hidden;font-size:13.5px;width:100%}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:#f8fafc;font-size:12px;color:#374151}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}} .muted{{color:var(--muted)}}
.note{{background:#fffbeb;border-left:4px solid #b45309;padding:13px 16px;border-radius:0 10px 10px 0;
font-size:13.5px;color:#3f3f46}}
footer{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}}
</style></head><body>
<header><div class="in">
  <div class="eyebrow">Super_Trader · 横截面层</div>
  <h1>相对强弱与行业轮动</h1>
  <p>同一时点跨标的比较：涨幅排名靠前的标的，下一段是否继续跑赢中位数？
  收盘价面板 {shape[0]:,} 个交易日 × {shape[1]:,} 个标的（{escape(str(rng[0]))} ~ {escape(str(rng[1]))}）。</p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-k">面板规模</div><div class="kpi-v">{shape[1]:,}</div><div class="kpi-d">标的数 / {shape[0]:,} 个交易日</div></div>
    <div class="kpi"><div class="kpi-k">最强信号</div><div class="kpi-v">{_num(best['ic_t'], 2) if best else '—'}</div><div class="kpi-d">IC t 值（|t|&gt;2 才算有信号）</div></div>
    <div class="kpi"><div class="kpi-k">检验组合</div><div class="kpi-v">{len(ic)}</div><div class="kpi-d">{len(cfg.get('lookbacks', []))} 个回看 × {len(cfg.get('horizons', []))} 个周期</div></div>
    <div class="kpi"><div class="kpi-k">结论</div><div class="kpi-v">{escape(headline.split('：',1)[-1][:18])}</div><div class="kpi-d">详见下表</div></div>
  </div>
</div></header>
<main>
  <section>
    <h2>1. 横截面 IC 与分位价差</h2>
    <p class="lead">IC = 当天「相对强弱排名」与「未来相对收益」的 Spearman 相关，逐日算完再对时间取均值；
    t 值用 Newey-West（同一天几千只股票算一个观测，避免把横截面相关性当成独立样本）。
    分位价差 = 最靠前 1/{cfg.get('quantiles', 5)} 减最靠后 1/{cfg.get('quantiles', 5)} 的未来相对收益。</p>
    <table><thead><tr><th>特征</th><th class="num">周期</th><th class="num">IC 均值</th>
      <th class="num">IC t</th><th class="num">IC&gt;0 占比</th>
      <th class="num">分位价差</th><th class="num">价差 t</th><th class="num">有效天数</th></tr></thead>
      <tbody>{''.join(row_ic(r) for r in ic)}</tbody></table>
  </section>

  <section>
    <h2>2. 绝对方向：高分位未来上涨的比例</h2>
    <p class="lead">这一节不剥离市场涨跌，直接看「排名靠前的标的未来上涨概率」有没有高于全样本基准。</p>
    <table><thead><tr><th>相对强弱</th><th class="num">周期</th><th class="num">前 20% 上涨率</th>
      <th class="num">全样本上涨率</th><th class="num">后 20% 上涨率</th><th class="num">超额</th></tr></thead>
      <tbody>{''.join(row_abs(r) for r in absolute)}</tbody></table>
  </section>

  <section>
    <h2>3. 怎么读</h2>
    <div class="note">
      <p><b>IC 的正负号</b>：正号代表「前期涨得多的，后面继续跑赢」（动量）；
      负号代表「前期涨得多的，后面跑输」（反转）。A 股短周期通常是反转，长周期不确定。</p>
      <p><b>|t| &gt; 2 才算有统计信号</b>，而且这里做了 9 个组合的检验，按 5% 水平光靠运气就会有
      约 0.45 个 |t|&gt;2 的假信号，看到一两个显著不要激动。</p>
      <p><b>分位价差是等权、未扣成本、未考虑涨跌停与流动性</b>，真实可交易性更差；
      高换手的横截面策略对成本极其敏感。</p>
      <p><b>全市场标的含已退市与新股</b>，逐日横截面要求当天有足够多标的交易，
      停牌股不参与当天排名，因此面板不是严格平衡的。</p>
    </div>
  </section>

  <footer>生成于 {escape(str(payload.get('generated_at') or ''))} · 仅供研究，不构成投资建议。</footer>
</main></body></html>"""
    out.write_text(html, encoding="utf-8", newline="\n")
    return out
