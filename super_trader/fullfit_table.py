"""全市场明细表：按类型区分，含中性占比、准确率、预测次数与持有期。

页面用内嵌 JSON + 前端渲染：一次只渲染所选类型（指数/ETF/个股），
避免 6,000 多行 DOM 卡住浏览器。
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np

TYPE_ORDER = ("指数", "ETF", "个股")
DRIFT_HIGH = 0.85
DRIFT_LOW = 0.15


def _pct(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v * 100:.{d}f}%" if np.isfinite(v) else "—"


def enrich(records: list[dict]) -> list[dict]:
    """补中性占比、相对单边基准的技巧值，并标记单边标的。"""
    rows = []
    for r in records:
        eligible = r.get("eligible") or 0
        preds = r.get("predictions") or 0
        neutral = max(eligible - preds, 0)
        base = r.get("base_up_rate")
        acc = r.get("weighted_accuracy")
        skill = None
        if base is not None and acc is not None:
            skill = acc - max(float(base), 1.0 - float(base))
        one_sided = base is not None and (base > DRIFT_HIGH or base < DRIFT_LOW)
        rows.append({
            **r,
            "neutral_count": neutral,
            "neutral_share": (neutral / eligible) if eligible else None,
            "coverage": (preds / eligible) if eligible else None,
            "skill_pp": skill * 100 if skill is not None else None,
            "one_sided": bool(one_sided),
        })
    return rows


BOND_HINTS = ("债", "信用", "货币", "现金", "短融", "添利", "日利", "理财", "国债", "企债")


def flag_tiny_move(rows: list[dict], cfg, threshold: float = 0.005,
                   min_accuracy: float = 0.80) -> int:
    """标记「几乎不波动」的标的（货币 ETF、短融、债券指数）。

    触发条件（任一）：准确率 ≥80%，或名称含债/货币/现金等关键词。
    债类指数经常既有 50% 左右的基准涨率又有极小的波动，只按准确率筛会漏。
    """
    from .data import KlineStore, SymbolInfo

    def looks_special(r):
        name = str(r.get("name") or "")
        return (any(k in name for k in BOND_HINTS)
                or (r.get("weighted_accuracy") or 0) >= min_accuracy)

    suspects = [r for r in rows if not r["one_sided"] and not r.get("tiny_move")
                and looks_special(r)]
    if not suspects:
        return 0
    store = KlineStore(cfg.data_path, cfg.cache_path)
    cat = store.catalog()
    lookup = {(str(row["category"]), str(row["code"])): row for _, row in cat.iterrows()}
    marked = 0
    for r in suspects:
        row = lookup.get((str(r.get("category")), str(r.get("code"))))
        if row is None:
            continue
        info = SymbolInfo(code=str(row["code"]), name=str(row["name"]),
                          category=str(row["category"]), path=str(row["path"]),
                          board=str(row["board"]), price_limit=float(row["price_limit"]))
        try:
            df = store.load(info)
        except Exception:
            continue
        close = df["close"].to_numpy(float)
        h = int(r["horizon"])
        if close.size <= h + 20:
            continue
        fwd = close[h:] / close[:-h] - 1.0
        med = float(np.nanmedian(np.abs(fwd)))
        if med < threshold:
            r["tiny_move"] = True
            r["median_move"] = med
            marked += 1
    return marked


def summarize_by_type(rows: list[dict]) -> list[dict]:
    out = []
    for t in TYPE_ORDER:
        sub = [r for r in rows if r.get("category") == t]
        if not sub:
            continue
        acc = [r["weighted_accuracy"] for r in sub if r.get("weighted_accuracy") is not None]
        neu = [r["neutral_share"] for r in sub if r.get("neutral_share") is not None]
        out.append({"type": t, "count": len(sub),
                    "accuracy": float(np.mean(acc)) if acc else None,
                    "median_acc": float(np.median(acc)) if acc else None,
                    "neutral": float(np.mean(neu)) if neu else None})
    return out


def _pack(r: dict) -> list:
    d = r.get("latest_direction")
    if r.get("probability") is not None:
        call = f"{r['direction_label']} {_pct(r['probability'])}"
        cls = "up" if d == 1 else "down"
    else:
        base = r.get("base_up_rate")
        call = f"中性（基准涨率 {_pct(base)}）" if base is not None else "中性"
        cls = "flat"
    return [r["rank"], str(r["code"]), str(r["name"]), str(r.get("category") or ""),
            round(float(r["weighted_accuracy"] or 0), 4),
            round(float(r["plain_accuracy"] or 0), 4),
            round(float(r["neutral_share"] or 0), 4),
            int(r["predictions"] or 0), int(r["eligible"] or 0), int(r["horizon"]),
            call, cls, round(float(r.get("skill_pp") or 0), 1),
            round(float(r.get("recent_accuracy") or 0), 4),
            " + ".join(x.split("__")[-1] for x in (r.get("rules") or [])[:2])]


def build_table_html(payload: dict, out_path: str | Path | None = None, cfg=None) -> Path:
    all_rows = enrich(payload.get("records") or [])
    if cfg is not None:
        flag_tiny_move(all_rows, cfg)
    special = [r for r in all_rows if r["one_sided"] or r.get("tiny_move")]
    rows = [r for r in all_rows if not r["one_sided"] and not r.get("tiny_move")]
    rows.sort(key=lambda r: (-(r.get("weighted_accuracy") or 0), -(r.get("predictions") or 0)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    special.sort(key=lambda r: -(r.get("weighted_accuracy") or 0))
    one_sided = [r for r in special if r["one_sided"]]
    tiny = [r for r in special if r.get("tiny_move")]
    stats = summarize_by_type(rows)

    out = Path(out_path) if out_path else Path("table_fullmarket.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    cards = "".join(
        f"""<div class="card"><div class="card-k">{escape(s['type'])}</div>
        <div class="card-v">{s['count']:,}</div>
        <div class="card-d">平均准确率 {_pct(s['accuracy'])} · 中位 {_pct(s['median_acc'])}<br>
        平均中性占比 {_pct(s['neutral'])}</div></div>""" for s in stats)

    chips = "".join(
        f'<button class="chip{" on" if s["type"] == TYPE_ORDER[0] else ""}" '
        f'data-type="{escape(s["type"])}">{escape(s["type"])} {s["count"]:,}</button>' for s in stats)

    special_rows = "".join(
        f"""<tr><td><span class="mono">{escape(str(r['code']))}</span> {escape(str(r['name']))}
        <span class="tag t{escape(str(r['category']))}">{escape(str(r['category']))}</span></td>
        <td>{'单边（基准涨率异常）' if r['one_sided'] else '几乎不波动'}</td>
        <td class="num">{_pct(r['weighted_accuracy'])}</td>
        <td class="num">{_pct(r['neutral_share'], 1)}</td>
        <td class="num">{(r['predictions'] or 0):,}</td>
        <td class="num">{_pct(r.get('base_up_rate'), 1)}</td>
        <td class="num">{_pct(r.get('median_move'), 3)}</td>
        <td class="num">H{r['horizon']}</td></tr>""" for r in special) or \
        "<tr><td colspan='8' class='muted'>无</td></tr>"

    data_json = json.dumps([_pack(r) for r in rows], ensure_ascii=False, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super_Trader · 全市场准确率明细</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--ink:#121a2b;--muted:#6b7280;--line:#e3e8ef;--accent:#1e3a8a;--pos:#047857;--neg:#b91c1c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:14.5px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
header{{background:linear-gradient(150deg,#0d1b36,#1e3a8a 65%,#2f5bb7);color:#fff;padding:32px 40px 24px}}
header .in,main{{max-width:1500px;margin:0 auto}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;opacity:.72}}
h1{{margin:8px 0 6px;font-size:27px}}
header p{{margin:0;opacity:.86;max-width:900px;font-size:13.5px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:22px}}
.card{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.17);border-radius:12px;padding:13px 15px}}
.card-k{{font-size:11px;letter-spacing:.12em;opacity:.78}}
.card-v{{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}}
.card-d{{font-size:11.5px;opacity:.75;line-height:1.5}}
main{{padding:22px 40px 60px}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 12px}}
input[type=search]{{padding:8px 12px;border:1px solid var(--line);border-radius:9px;font-size:14px;min-width:240px}}
button.chip{{border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 14px;
font-size:12.5px;cursor:pointer;color:#374151}}
button.chip.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
.spacer{{flex:1}}
.table-wrap{{max-height:74vh;overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
thead th{{position:sticky;top:0;background:#f8fafc;font-size:11.5px;color:#374151;z-index:2}}
th.sortable{{cursor:pointer;user-select:none}} th.sortable:hover{{color:var(--accent)}}
tbody tr:hover{{background:#f8faff}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#475569}}
.small{{font-size:11.5px}} .muted{{color:var(--muted)}}
.tag{{font-size:10.5px;padding:1px 7px;border-radius:999px;margin-left:5px;border:1px solid transparent}}
.t指数{{background:#eef2ff;color:#3730a3;border-color:#c7d2fe}}
.tETF{{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}}
.t个股{{background:#fff7ed;color:#9a3412;border-color:#fed7aa}}
.dir{{font-weight:700}} .dir.up{{color:#b91c1c}} .dir.down{{color:#047857}} .dir.flat{{color:var(--muted);font-weight:400}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
h2{{font-size:17px;margin:0 0 8px}}
.lead{{color:#6b7280;font-size:13px;margin:0 0 10px}}
footer{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;margin-top:20px}}
</style></head><body>
<header><div class="in">
  <div class="eyebrow">Super_Trader · 全市场 · 内样本</div>
  <h1>全市场 {len(rows):,} 个标的明细</h1>
  <p>中性占比 = 可评估样本里没有发出预测的比例；准确率只统计发出预测的样本。
  每个标的在 105 条规则 × 1~20 根持有期中挑选胜率最高的一根。
  「超出单边」= 准确率 − 单边猜基准（涨/跌里较高的那个），衡量是否真的优于无脑猜一边。</p>
  <p style="margin-top:8px;font-size:13px;opacity:.8">
  另有 <b>{len(special)} 个特殊标的</b>（{len(one_sided)} 个单边 + {len(tiny)} 个几乎不波动）
  列在页尾，不参与排名——它们的「高准确率」只反映趋势或面值特性。</p>
  <div class="cards">{cards}</div>
</div></header>
<main>
  <div class="controls">
    {chips}
    <span class="spacer"></span>
    <input type="search" id="q" placeholder="搜索代码 / 名称 / 规则">
    <span class="muted small" id="status"></span>
  </div>
  <div class="table-wrap"><table id="t">
    <thead><tr>
      <th class="sortable" data-k="0">#</th><th>标的</th>
      <th class="sortable num" data-k="4">准确率</th>
      <th class="sortable num" data-k="5">未加权</th>
      <th class="sortable num" data-k="6">中性占比</th>
      <th class="sortable num" data-k="7">预测次数</th>
      <th class="num">可评估</th>
      <th class="sortable num" data-k="9">持有期</th>
      <th>第N根后</th>
      <th class="sortable num" data-k="12">超出单边</th>
      <th class="sortable num" data-k="13">最近一年</th>
      <th>规则</th>
    </tr></thead>
    <tbody id="tb"></tbody>
  </table></div>

  <section style="margin-top:26px">
    <h2>特殊标的（不参与排名）</h2>
    <p class="lead">单边标的：几乎天天同向；低波动标的：h 根后中位涨跌幅 &lt; 0.5%（货币 ETF、短融等）。</p>
    <div class="table-wrap" style="max-height:38vh"><table>
      <thead><tr><th>标的</th><th>原因</th><th class="num">准确率</th>
        <th class="num">中性占比</th><th class="num">预测次数</th><th class="num">基准涨率</th>
        <th class="num">中位涨跌幅</th><th class="num">持有期</th></tr></thead>
      <tbody>{special_rows}</tbody>
    </table></div>
  </section>

  <footer>
    生成于 {escape(str(payload.get('generated_at') or ''))} · 方法 {escape(str(payload.get('method') or ''))} ·
    全部为全历史拟合的内样本数字，不代表未来准确率；仅供研究，不构成投资建议。
  </footer>
</main>
<script>
const DATA = {data_json};
const ACC=4, PLAIN=5, NEU=6, N=7, ELIG=8, H=9, CALL=10, CLS=11, SKILL=12, REC=13, RULES=14;
let type = '{escape(TYPE_ORDER[0])}', kw = '', sortKey = null, sortDir = -1;
const pct = v => (v*100).toFixed(2) + '%';
const pct1 = v => (v*100).toFixed(1) + '%';
function esc(s) {{ return String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function view() {{
  let rows = DATA.filter(r => r[3] === type);
  if (kw) rows = rows.filter(r => (r[1]+r[2]+r[RULES]).toLowerCase().includes(kw));
  if (sortKey !== null) {{
    rows = rows.slice().sort((a, b) => {{
      const x = a[sortKey], y = b[sortKey];
      return (typeof x === 'number' && typeof y === 'number') ? sortDir*(x-y) : sortDir*String(x).localeCompare(String(y));
    }});
  }}
  return rows;
}}
function render() {{
  const rows = view();
  document.getElementById('tb').innerHTML = rows.map(r => {{
    const hi = r[ACC] >= 0.7 ? ' style="color:#047857;font-weight:700"' : '';
    const sk = r[SKILL] > 0 ? 'pos' : 'neg';
    return `<tr><td class="num muted">${{r[0]}}</td>
      <td><span class="mono">${{esc(r[1])}}</span> <b>${{esc(r[2])}}</b><span class="tag t${{esc(r[3])}}">${{esc(r[3])}}</span></td>
      <td class="num"><b${{hi}}>${{pct(r[ACC])}}</b></td>
      <td class="num">${{pct(r[PLAIN])}}</td>
      <td class="num">${{pct1(r[NEU])}}</td>
      <td class="num"><b>${{r[N].toLocaleString()}}</b></td>
      <td class="num muted">${{r[ELIG].toLocaleString()}}</td>
      <td class="num">H${{r[H]}}</td>
      <td class="dir ${{r[CLS]}}">${{esc(r[CALL])}}</td>
      <td class="num ${{sk}}">${{r[SKILL] > 0 ? '+' : ''}}${{r[SKILL].toFixed(1)}}</td>
      <td class="num">${{r[REC] ? pct(r[REC]) : '—'}}</td>
      <td class="mono small">${{esc(r[RULES])}}</td></tr>`;
  }}).join('');
  document.getElementById('status').textContent = `显示 ${{rows.length.toLocaleString()}} / ${{DATA.length.toLocaleString()}}`;
}}
document.querySelectorAll('button.chip').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('button.chip').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); type = b.dataset.type; render();
}}));
document.getElementById('q').addEventListener('input', e => {{ kw = e.target.value.trim().toLowerCase(); render(); }});
document.querySelectorAll('th.sortable').forEach(th => th.addEventListener('click', () => {{
  const k = +th.dataset.k;
  sortDir = (k === sortKey) ? -sortDir : -1; sortKey = k; render();
}}));
render();
</script></body></html>"""
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def build_from_json(source: str | Path, out_path: str | Path | None = None, cfg=None) -> Path:
    src = Path(source)
    payload = json.loads(src.read_text(encoding="utf-8"))
    return build_table_html(payload, out_path or src.parent / "table_fullmarket.html", cfg=cfg)
