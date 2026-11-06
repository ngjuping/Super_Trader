"""全历史拟合结果的排行页（内样本）。"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path


def _pct(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v * 100:.{d}f}%"


def build_fullfit_html(payload: dict, out_path: str | Path | None = None) -> Path:
    rows = payload.get("records") or []
    totals = payload.get("totals") or {}
    cfg = payload.get("config") or {}
    out = Path(out_path) if out_path else Path("fullfit.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    def tr(r):
        tone = "pos" if (r["weighted_accuracy"] or 0) >= .6 else ""
        d = r.get("latest_direction")
        cls = "up" if d == 1 else ("down" if d == -1 else "flat")
        if r.get("probability") is not None:
            label = f'{r["direction_label"]} {_pct(r["probability"])}'
        else:
            base = r.get("base_up_rate")
            label = (f'中性（该周期历史上涨率 {_pct(base)}）' if base is not None
                     else '中性')
        inv = ' <span class="warn">反向</span>' if r.get("inverted") else ""
        rules = " + ".join(escape(x.split("__")[-1]) for x in (r.get("rules") or []))
        return f"""<tr data-acc="{r['weighted_accuracy'] or 0}" data-n="{r['predictions'] or 0}">
          <td class="num muted">{r['rank']}</td>
          <td><span class="mono">{escape(r['code'])}</span> <b>{escape(r['name'])}</b>
              <span class="tag">{escape(r['category'])}</span>{inv}</td>
          <td class="num acc"><b class="{tone}">{_pct(r['weighted_accuracy'])}</b></td>
          <td class="num"><b>{r['horizon']}</b></td>
          <td class="dir {cls}">{label}</td>
          <td class="num">{_pct(r['recent_accuracy'])}</td>
          <td class="num muted">{_pct(r['plain_accuracy'])}</td>
          <td class="num"><b>{(r['predictions'] or 0):,}</b></td>
          <td class="num">{_pct(r['coverage'], 1)}</td>
          <td class="mono" style="font-size:12px">{rules}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super_Trader · 全历史拟合准确率</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--ink:#121a2b;--muted:#6b7280;--line:#e3e8ef;
--accent:#1e3a8a;--pos:#047857;--warn:#b45309}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
header{{background:linear-gradient(150deg,#0d1b36,#1e3a8a 65%,#2f5bb7);color:#fff;padding:34px 40px 26px}}
header .in,main{{max-width:1300px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;opacity:.72}}
h1{{margin:8px 0 6px;font-size:28px}}
header p{{margin:0;opacity:.86;max-width:860px;font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin-top:22px}}
.kpi{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.17);border-radius:12px;padding:13px 15px}}
.kpi-k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;opacity:.75}}
.kpi-v{{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}}
.kpi-d{{font-size:12px;opacity:.72}}
main{{padding:24px 40px 60px}}
.banner{{background:#fffbeb;border-left:4px solid var(--warn);padding:14px 16px;
border-radius:0 10px 10px 0;font-size:13.5px;color:#3f3f46;margin-bottom:22px}}
.banner b{{color:#92400e}}
.controls{{margin:0 0 10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
input[type=search]{{padding:8px 12px;border:1px solid var(--line);border-radius:9px;font-size:14px;min-width:220px}}
.table-wrap{{max-height:76vh;overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
thead th{{position:sticky;top:0;background:#f8fafc;font-size:12px;color:#374151;z-index:2}}
tbody tr:hover{{background:#f8faff}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#475569}}
.muted{{color:var(--muted)}} .pos{{color:var(--pos)}} .warn{{color:var(--warn);font-size:11px}}
.dir{{font-weight:700}} .dir.up{{color:#b91c1c}} .dir.down{{color:#047857}} .dir.flat{{color:var(--muted);font-weight:400}}
.tag{{font-size:10.5px;background:#eef2ff;color:var(--accent);padding:1px 7px;border-radius:999px;margin-left:4px}}
footer{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;margin-top:22px}}
</style></head><body>
<header><div class="in">
  <div class="eyebrow">Super_Trader · 全历史拟合（不分验证集/测试集）</div>
  <h1>各标的：第 N 根后的涨跌概率</h1>
  <p>全部历史参与拟合；样本权重按半衰期 {cfg.get('half_life_days', 365):.0f} 天衰减，
  最近一年权重最高。每个标的在 78 条规则 × 5 个持有期中挑选，不达标就叠加或换规则。</p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-k">标的数</div><div class="kpi-v">{totals.get('symbols', 0)}</div>
      <div class="kpi-d">全部 ≥ 50%（反向 {totals.get('inverted', 0)} 个）</div></div>
    <div class="kpi"><div class="kpi-k">加权准确率均值</div><div class="kpi-v">{_pct(totals.get('weighted_accuracy'))}</div>
      <div class="kpi-d">中位数 {_pct(totals.get('median_accuracy'))}</div></div>
    <div class="kpi"><div class="kpi-k">未加权准确率</div><div class="kpi-v">{_pct(totals.get('plain_accuracy'))}</div>
      <div class="kpi-d">每次预测等权，更能反映整体</div></div>
    <div class="kpi"><div class="kpi-k">预测次数</div><div class="kpi-v">{(totals.get('predictions') or 0):,}</div>
      <div class="kpi-d">中性（无信号/平票）不计入</div></div>
  </div>
</div></header>
<main>
  <div class="banner">
    <b>这是内样本数字，不是未来准确率。</b>全历史参与拟合且不做任何切分，所以不存在未参与拟合的数据；
    「加权准确率」衡量的是规则把历史解释得多好。同一批数据上继续增加规则、换方向、调阈值，
    数字还能更高，但那不代表预测能力。真实前瞻效果需要等新数据。<br>
    另外：加权准确率普遍高于未加权，说明这些规则在最近一两年比在更早的历史上更贴合；
    如果市场风格切换回去，表现会回落。
  </div>
  <div class="controls">
    <input type="search" id="q" placeholder="搜索代码 / 名称 / 规则">
    <span class="muted" id="status" style="font-size:12.5px"></span>
  </div>
  <div class="table-wrap"><table id="t">
    <thead><tr><th>#</th><th>标的</th><th class="num">加权准确率</th>
      <th class="num">持有期</th><th>第N根后</th>
      <th class="num">最近一年</th><th class="num">未加权</th>
      <th class="num">预测次数</th><th class="num">覆盖率</th><th>选中规则</th></tr></thead>
    <tbody>{''.join(tr(r) for r in rows)}</tbody>
  </table></div>
  <footer>
    生成于 {escape(str(payload.get('generated_at') or ''))} · 方法 {escape(str(payload.get('method') or ''))}
    · 排除依赖事后日期的规则：{escape(', '.join(payload.get('excluded_rules') or []))}
    · 仅供研究，不构成投资建议。
  </footer>
</main>
<script>
const rows = () => [...document.querySelectorAll('#t tbody tr')];
document.getElementById('q').addEventListener('input', e => {{
  const k = e.target.value.trim().toLowerCase(); let shown = 0;
  rows().forEach(r => {{ const ok = !k || r.textContent.toLowerCase().includes(k);
    r.hidden = !ok; if (ok) shown++; }});
  document.getElementById('status').textContent = `显示 ${{shown}} / ${{rows().length}} 个标的`;
}});
</script></body></html>"""
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def build_from_json(source: str | Path, out_path: str | Path | None = None) -> Path:
    src = Path(source)
    payload = json.loads(src.read_text(encoding="utf-8"))
    return build_fullfit_html(payload, out_path or src.parent / "ranking_fullfit.html")
