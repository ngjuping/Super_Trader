"""标的级历史预测准确率排行（中性不计入分母）。

只读取 accuracy.json 的测试段汇总，不重算、不挑选：排序仅按历史条件准确率，
附预测次数与 95% 区间，避免把「只预测几次全对」误读成最准。
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

from .stats import wilson_ci


def _f(x, digits=2, suffix=""):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{digits}f}{suffix}"


def _pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def aggregate_by_symbol(records: list[dict]) -> list[dict]:
    """把同一标的的各持有期测试结果合并；中性仍不进分母。"""
    groups: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r.get("category") or "", r.get("code") or "")
        g = groups.setdefault(key, {
            "category": key[0], "code": key[1], "name": r.get("name") or "",
            "predictions": 0, "correct": 0, "eligible": 0, "neutral": 0,
            "up": 0, "down": 0, "flat": 0,
            "baseline_hits": 0.0, "periods": [], "models": 0,
        })
        t = r.get("test") or {}
        n = int(t.get("predictions") or 0)
        c = int(t.get("correct") or 0)
        g["predictions"] += n
        g["correct"] += c
        g["eligible"] += int(t.get("eligible") or 0)
        g["neutral"] += int(t.get("neutral") or 0)
        g["up"] += int(t.get("up_predictions") or 0)
        g["down"] += int(t.get("down_predictions") or 0)
        g["flat"] += int(t.get("flat_actual") or 0)
        if t.get("majority_baseline") is not None:
            g["baseline_hits"] += float(t["majority_baseline"]) * int(t.get("eligible") or 0)
        if r.get("model"):
            g["models"] += 1
        g["periods"].append({
            "horizon": int(r.get("horizon") or 0),
            "predictions": n, "correct": c,
            "accuracy": (c / n) if n else None,
            "eligible": int(t.get("eligible") or 0),
            "baseline": t.get("majority_baseline"),
        })

    rows = []
    for g in groups.values():
        n, c = g["predictions"], g["correct"]
        used = [p for p in g["periods"] if p["predictions"] > 0]
        best = max(used, key=lambda p: (p["accuracy"], p["predictions"])) if used else None
        worst = min(used, key=lambda p: (p["accuracy"], -p["predictions"])) if used else None
        lo, hi = wilson_ci(c, n) if n else (None, None)
        rows.append({
            **{k: g[k] for k in ("category", "code", "name", "predictions", "correct",
                                 "eligible", "neutral", "up", "down", "flat", "models")},
            "accuracy": (c / n) if n else None,
            "coverage": (n / g["eligible"]) if g["eligible"] else None,
            "ci_lo": lo, "ci_hi": hi,
            "baseline": (g["baseline_hits"] / g["eligible"]) if g["eligible"] else None,
            "edge_pp": ((c / n) - g["baseline_hits"] / g["eligible"]) * 100
            if n and g["eligible"] else None,
            "periods_used": len(used),
            "best_period": best, "worst_period": worst,
        })
    # 主排序：准确率降序；同准确率时预测次数多的在前，避免小样本占先
    rows.sort(key=lambda r: (-(r["accuracy"] if r["accuracy"] is not None else -1),
                             -r["predictions"], r["code"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def build_ranking(records: list[dict]) -> dict:
    symbols = aggregate_by_symbol(records)
    total_n = sum(r["predictions"] for r in symbols)
    total_c = sum(r["correct"] for r in symbols)
    eligible = sum(r["eligible"] for r in symbols)
    horizon_rows = []
    for r in records:
        t = r.get("test") or {}
        n = int(t.get("predictions") or 0)
        horizon_rows.append({
            "category": r.get("category") or "", "code": r.get("code") or "",
            "name": r.get("name") or "", "horizon": int(r.get("horizon") or 0),
            "predictions": n, "correct": int(t.get("correct") or 0),
            "accuracy": (int(t.get("correct") or 0) / n) if n else None,
            "eligible": int(t.get("eligible") or 0),
            "coverage": (n / int(t["eligible"])) if t.get("eligible") else None,
            "baseline": t.get("majority_baseline"),
            "has_model": bool(r.get("model")),
        })
    horizon_rows.sort(key=lambda r: (-(r["accuracy"] if r["accuracy"] is not None else -1),
                                     -r["predictions"], r["code"], r["horizon"]))
    return {
        "symbols": symbols, "horizons": horizon_rows,
        "totals": {"symbols": len(symbols), "predictions": total_n,
                   "correct": total_c, "eligible": eligible,
                   "accuracy": (total_c / total_n) if total_n else None,
                   "coverage": (total_n / eligible) if eligible else None,
                   "median_accuracy": _median([r["accuracy"] for r in symbols
                                               if r["accuracy"] is not None])},
    }


def _median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def build_ranking_html(payload: dict, source: dict | None = None,
                       out_path: str | Path | None = None) -> Path:
    data = build_ranking(payload.get("records") or [])
    meta = (source or payload).get("totals") or {}
    t = data["totals"]
    out = Path(out_path) if out_path else Path("ranking.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    def row_symbol(r: dict) -> str:
        acc = r["accuracy"]
        width = (acc or 0) * 100
        tone = "pos" if (r["edge_pp"] or 0) > 0 else ("neg" if (r["edge_pp"] or 0) < 0 else "")
        small = " small" if r["predictions"] < 30 else ""
        return f"""<tr class="row{small}" data-acc="{acc if acc is not None else -1}"
          data-n="{r['predictions']}" data-cov="{r['coverage'] if r['coverage'] is not None else -1}"
          data-edge="{r['edge_pp'] if r['edge_pp'] is not None else -999}">
          <td class="num muted">{r['rank']}</td>
          <td><span class="mono">{escape(r['code'])}</span> <b>{escape(r['name'])}</b>
              <span class="tag">{escape(r['category'])}</span></td>
          <td class="num acc"><div class="bar"><i style="width:{width:.1f}%"></i></div>
              <b>{_pct(acc)}</b></td>
          <td class="num"><b>{r['predictions']:,}</b></td>
          <td class="num">{r['correct']:,}</td>
          <td class="num muted">[{_pct(r['ci_lo'])}, {_pct(r['ci_hi'])}]</td>
          <td class="num">{_pct(r['coverage'])}</td>
          <td class="num muted">{_pct(r['baseline'])}</td>
          <td class="num {tone}">{_f(r['edge_pp'], 2, 'pp') if r['edge_pp'] is not None else '—'}</td>
          <td class="num">{r['periods_used']}</td>
        </tr>"""

    def row_h(r: dict) -> str:
        small = " small" if r["predictions"] < 30 else ""
        return f"""<tr class="row{small}" data-acc="{r['accuracy'] if r['accuracy'] is not None else -1}"
          data-n="{r['predictions']}">
          <td><span class="mono">{escape(r['code'])}</span> {escape(r['name'])}
              <span class="tag">{escape(r['category'])}</span></td>
          <td class="num">H{r['horizon']}</td>
          <td class="num acc"><b>{_pct(r['accuracy'])}</b></td>
          <td class="num"><b>{r['predictions']:,}</b></td>
          <td class="num">{r['correct']:,}</td>
          <td class="num">{_pct(r['coverage'])}</td>
          <td class="num muted">{_pct(r['baseline'])}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super_Trader · 标的准确率排行</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--ink:#121a2b;--muted:#6b7280;--line:#e3e8ef;
--accent:#1e3a8a;--pos:#047857;--neg:#b91c1c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
header{{background:linear-gradient(150deg,#0d1b36,#1e3a8a 65%,#2f5bb7);color:#fff;padding:34px 40px 28px}}
header .in,main{{max-width:1320px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;opacity:.72}}
h1{{margin:8px 0 6px;font-size:29px}}
header p{{margin:0;opacity:.85;max-width:820px;font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:13px;margin-top:24px}}
.kpi{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.17);border-radius:12px;padding:13px 15px}}
.kpi-k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;opacity:.75}}
.kpi-v{{font-size:25px;font-weight:700;font-variant-numeric:tabular-nums}}
.kpi-d{{font-size:12px;opacity:.72}}
main{{padding:26px 40px 70px}}
section{{margin-bottom:30px}}
h2{{font-size:18px;margin:0 0 10px;display:flex;gap:9px;align-items:center}}
h2 .n{{display:inline-flex;width:25px;height:25px;border-radius:7px;background:var(--accent);
color:#fff;font-size:12.5px;align-items:center;justify-content:center;font-weight:700}}
.lead{{color:var(--muted);margin:-2px 0 12px;font-size:13.5px;max-width:1000px}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 12px}}
input[type=search]{{padding:8px 12px;border:1px solid var(--line);border-radius:9px;
font-size:14px;min-width:230px;background:#fff}}
button.chip{{border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 13px;
font-size:12.5px;cursor:pointer;color:#374151}}
button.chip.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
.table-wrap{{max-height:74vh;overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
thead th{{position:sticky;top:0;background:#f8fafc;font-size:12px;color:#374151;z-index:2}}
th.sortable{{cursor:pointer;user-select:none}}
th.sortable:hover{{color:var(--accent)}}
tbody tr:hover{{background:#f8faff}}
tr.small td{{opacity:.62}}
tr.small td.acc b::after{{content:" ·小样本";font-size:10.5px;color:#b45309;font-weight:400}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#475569}}
.muted{{color:var(--muted)}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
.tag{{font-size:10.5px;background:#eef2ff;color:var(--accent);padding:1px 7px;border-radius:999px;margin-left:4px}}
.acc{{white-space:nowrap}}
.bar{{display:inline-block;width:76px;height:6px;background:#eef1f6;border-radius:3px;
vertical-align:middle;margin-right:8px;overflow:hidden}}
.bar i{{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#1e3a8a)}}
.note{{background:#fffbeb;border-left:4px solid #b45309;padding:13px 16px;border-radius:0 10px 10px 0;
font-size:13.5px;color:#3f3f46}}
footer{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}}
</style></head><body>
<header><div class="in">
  <div class="eyebrow">Super_Trader · 准确率优先模式 · 测试段回顾</div>
  <h1>各标的历史预测准确率排行</h1>
  <p>按标的汇总其全部持有期的测试段结果。准确率 = 正确次数 ÷ 发出涨跌预测的次数，
  <b>中性（弃权）不计入分母</b>；默认按准确率从高到低。</p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-k">标的数</div><div class="kpi-v">{t['symbols']}</div>
      <div class="kpi-d">共 {len(data['horizons'])} 个标的×持有期模型</div></div>
    <div class="kpi"><div class="kpi-k">总准确率</div><div class="kpi-v">{_pct(t['accuracy'])}</div>
      <div class="kpi-d">{t['correct']:,} / {t['predictions']:,} 次预测</div></div>
    <div class="kpi"><div class="kpi-k">标的准确率中位数</div><div class="kpi-v">{_pct(t['median_accuracy'])}</div>
      <div class="kpi-d">一半标的在此之上</div></div>
    <div class="kpi"><div class="kpi-k">覆盖率</div><div class="kpi-v">{_pct(t['coverage'])}</div>
      <div class="kpi-d">其余 {_pct(1 - (t['coverage'] or 0))} 保持中性</div></div>
  </div>
</div></header>
<main>
  <section>
    <h2><span class="n">1</span>标的排行（全部持有期合并）</h2>
    <p class="lead">点击表头可按该列排序；「小样本」= 预测次数少于 30，准确率波动很大，请同时看 95% 区间。</p>
    <div class="controls">
      <input type="search" id="q" placeholder="搜索代码 / 名称 / 类别">
      <span class="muted" style="font-size:12.5px">最少预测次数：</span>
      <button class="chip on" data-min="0">全部</button>
      <button class="chip" data-min="30">≥30</button>
      <button class="chip" data-min="100">≥100</button>
      <button class="chip" data-min="300">≥300</button>
      <span class="muted" id="status" style="font-size:12.5px"></span>
    </div>
    <div class="table-wrap"><table id="rank">
      <thead><tr>
        <th>#</th><th>标的</th>
        <th class="sortable" data-key="acc">准确率</th>
        <th class="sortable" data-key="n">预测次数</th>
        <th>正确次数</th><th>95% 区间</th><th>覆盖率</th>
        <th>对照（多数方向）</th>
        <th class="sortable" data-key="edge">相对对照</th>
        <th>有效持有期</th>
      </tr></thead>
      <tbody>{''.join(row_symbol(r) for r in data['symbols'])}</tbody>
    </table></div>
  </section>

  <section>
    <h2><span class="n">2</span>按标的×持有期明细</h2>
    <p class="lead">同一标的在不同持有期上准确率差别很大，单独看合并值会掩盖这一点。</p>
    <div class="controls">
      <input type="search" id="q2" placeholder="搜索代码 / 名称 / 持有期(H1…)">
      <span class="muted" id="status2" style="font-size:12.5px"></span>
    </div>
    <div class="table-wrap"><table id="detail">
      <thead><tr><th>标的</th><th class="num">持有期</th><th class="num">准确率</th>
        <th class="num">预测次数</th><th class="num">正确次数</th>
        <th class="num">覆盖率</th><th class="num">对照</th></tr></thead>
      <tbody>{''.join(row_h(r) for r in data['horizons'])}</tbody>
    </table></div>
  </section>

  <section>
    <h2><span class="n">3</span>怎么读这张表</h2>
    <div class="note">
      <p><b>准确率只对「发出预测的那些时点」成立。</b>覆盖率低说明大多数时候选择了中性，把中性的时点
      算成「没错」会让数字失真，所以这里始终同时给出预测次数。</p>
      <p><b>排名靠前≠更可靠。</b>预测次数少的标的很容易出现 60%+ 的准确率，这是小样本波动，
      不是能力差异；请对照 95% 区间与「相对对照」列判断。</p>
      <p><b>这不是未来准确率。</b>测试段是历史后 25%，且整段历史已被本项目反复研究；
      真实前瞻表现需要等新数据。所有数值取自 <code>reports/accuracy/accuracy.json</code> 的测试段汇总。</p>
    </div>
  </section>

  <footer>
    生成于 {escape(str(payload.get('generated_at') or ''))} · 方法 {escape(str(payload.get('method') or ''))}
    · 全池准确率 {_pct(meta.get('accuracy'))}（{meta.get('correct', 0):,}/{meta.get('predictions', 0):,}）
    · 排序仅依据历史条件准确率，不构成投资建议。
  </footer>
</main>
<script>
const rows = () => [...document.querySelectorAll('#rank tbody tr')];
let minN = 0, keyword = '';
function apply() {{
  let shown = 0;
  rows().forEach(r => {{
    const n = +r.dataset.n;
    const text = r.textContent.toLowerCase();
    const ok = n >= minN && (!keyword || text.includes(keyword));
    r.hidden = !ok; if (ok) shown++;
  }});
  document.getElementById('status').textContent = `显示 ${{shown}} / ${{rows().length}} 个标的`;
}}
document.querySelectorAll('button.chip').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('button.chip').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); minN = +b.dataset.min; apply();
}}));
document.getElementById('q').addEventListener('input', e => {{
  keyword = e.target.value.trim().toLowerCase(); apply();
}});
const q2 = document.getElementById('q2');
q2.addEventListener('input', () => {{
  const k = q2.value.trim().toLowerCase(); let shown = 0;
  const list = [...document.querySelectorAll('#detail tbody tr')];
  list.forEach(r => {{
    const ok = !k || r.textContent.toLowerCase().includes(k);
    r.hidden = !ok; if (ok) shown++;
  }});
  document.getElementById('status2').textContent = `显示 ${{shown}} / ${{list.length}} 行`;
}});
let dir = -1, key = 'acc';
document.querySelectorAll('th.sortable').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.key;
  dir = (k === key) ? -dir : -1; key = k;
  const body = document.querySelector('#rank tbody');
  const sorted = rows().sort((a, b) => dir * (a.dataset[k] - b.dataset[k]));
  sorted.forEach(r => body.appendChild(r));
  document.querySelectorAll('#rank tbody tr').forEach((r, i) => r.cells[0].textContent = i + 1);
}}));
apply();
</script>
</body></html>"""
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def build_from_json(source_path: str | Path, out_path: str | Path | None = None) -> Path:
    src = Path(source_path)
    payload = json.loads(src.read_text(encoding="utf-8"))
    target = Path(out_path) if out_path else src.parent / "ranking.html"
    return build_ranking_html(payload, out_path=target)
