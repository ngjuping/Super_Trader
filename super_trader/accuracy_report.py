"""Standalone, descriptive accuracy/abstention reports (standard library only)."""
from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_NOTES = (
    "主指标为发出方向预测后的条件准确率：correct / predictions；中性（弃权）不进入准确率分母，也不计为正确。实际涨跌为零时，已发出的上涨/下跌预测均计为错判。",
    "覆盖率为 predictions / eligible；所有准确率必须同时查看覆盖率和预测样本量 n。减少预测可以提高条件准确率，但不代表整体预测能力提高。",
    "eligible 排除了事后发现含异常价格重塑（artifact）的前瞻窗口，覆盖率仅针对这些清洁标签样本，不代表实时全部可发出预测的覆盖率；发出信号时无法预知未来异常。",
    "弃权对照使用同一冻结规则集及 top-k，仅将阈值设为 0.50；无活跃规则或投票平局仍保持中性，并非强制每次预测。准确率差为弃权准确率减去该对照准确率，单位为百分点（pp），仅作描述，不保证改进或未来表现。",
    "准确率模式排除 month_start_long、month_end_long、pre_holiday_long 规则，避免不完整月份或后续交易日期引入前视；本次 excluded_rules 字段另列如下。",
    "时间顺序按 50% / 25% / 25% 划分：拟合段选择规则，校准段选择 top-k 与弃权阈值，测试段冻结规则和参数；所有分段边界均按预测 horizon 净化（purge），避免标签跨界。具体日期见逐项明细。",
    "校准段最低样本支持仅是选型约束，不保证未来准确率。本方法不以收益、利润、p 值或 FDR 作为准入门槛；方向准确率不等同于可交易收益。",
    "历史测试数据已在项目范围内被检视，并非从未接触的未来留出集。报告属于回顾性描述，不构成未来预测保证。",
    "汇总使用微观加权：先合计 correct、predictions、eligible，再计算准确率与覆盖率；不是逐模型准确率的简单平均。相关标的及重叠预测周期并非独立样本，样本总数不是独立证据数量。",
    "基准仅作对照：多数方向、始终看涨、始终看跌和选定基准均按输入指标展示；分母或样本选择不同时，不应直接宣称相对优势。最新 score 是模型分数，不是经过概率校准的胜率；最新方向也不是交易建议。",
)
_METRICS = (
    ("eligible", "可评估样本", False), ("predictions", "预测 n", False),
    ("correct", "正确数", False), ("accuracy", "条件准确率", True),
    ("coverage", "覆盖率", True), ("up_predictions", "看涨数", False),
    ("down_predictions", "看跌数", False), ("neutral", "中性数", False),
    ("flat_actual", "实际平盘数", False),
    ("majority_baseline", "多数方向基准", True),
    ("always_up_accuracy", "始终看涨准确率", True),
    ("always_down_accuracy", "始终看跌准确率", True),
    ("selected_baseline", "选定基准", True),
)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _number(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return _text(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.2%}" if percent else f"{number:g}"


def _difference_pp(accuracy: Any, unfiltered_accuracy: Any) -> str:
    if accuracy is None or unfiltered_accuracy is None:
        return "—"
    try:
        difference = (float(accuracy) - float(unfiltered_accuracy)) * 100
    except (TypeError, ValueError, OverflowError):
        return "—"
    return f"{difference:+.2f} pp" if math.isfinite(difference) else "—"


def _md(value: Any) -> str:
    return html.escape(_text(value), quote=True).replace("|", "&#124;").replace("\r", " ").replace("\n", "<br>")


def _table(headers: list[str], rows: list[list[Any]], markdown: bool = False) -> str:
    if markdown:
        lines = ["| " + " | ".join(_md(h) for h in headers) + " |",
                 "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(_md(c) for c in row) + " |" for row in rows)
        if not rows:
            lines.append("| 暂无数据 |" + " — |" * (len(headers) - 1))
        return "\n".join(lines)
    head = "".join(f"<th scope=\"col\">{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_text(c))}</td>" for c in row) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">暂无数据</td></tr>'
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_accuracy_report(payload: dict, out_dir: str | Path) -> list[Path]:
    """Write report_accuracy.html and ACCURACY.md; return their paths in that order.

    Input metrics are displayed, not silently recomputed or used to select models.
    None/missing metrics render as an em dash rather than an invented zero.
    """
    data = _mapping(payload)
    totals = _mapping(data.get("totals"))
    records = [_mapping(r) for r in (data.get("records") or [])]
    horizons = [_mapping(r) for r in (data.get("by_horizon") or [])]
    summary_keys = ("predictions", "correct", "eligible", "accuracy", "coverage", "symbols", "models")
    summary_headers = ["预测 n", "正确数", "可评估样本", "条件准确率", "覆盖率", "标的数", "模型数"]
    summary_rows = [[_number(totals.get(k), k in ("accuracy", "coverage")) for k in summary_keys]]
    horizon_rows = [[_text(r.get("horizon"))] + [_number(r.get(k), k in ("accuracy", "coverage")) for k in summary_keys] for r in horizons]
    identity_headers = ["代码", "名称", "类别", "周期", "状态"]
    comparison_headers = ["弃权准确率", "弃权预测 n", "弃权正确数", "0.50 对照准确率", "0.50 对照预测 n", "0.50 对照正确数", "描述性准确率差"]
    def comparison(values):
        return [_number(values.get("accuracy"), True), _number(values.get("predictions")),
                _number(values.get("correct")), _number(values.get("unfiltered_accuracy"), True),
                _number(values.get("unfiltered_predictions")), _number(values.get("unfiltered_correct")),
                _difference_pp(values.get("accuracy"), values.get("unfiltered_accuracy"))]
    overview, dates, metrics, models, record_comparisons = [], [], [], [], []
    for record in records:
        identity = [_text(record.get(k)) for k in ("code", "name", "category", "horizon", "status")]
        test = _mapping(record.get("test"))
        latest = _mapping(record.get("latest"))
        direction = {1: "上涨 (+1)", -1: "下跌 (-1)", 0: "中性 / 弃权 (0)"}.get(latest.get("direction"), "—")
        overview.append(identity + [_number(test.get("accuracy"), True), _number(test.get("coverage"), True), _number(test.get("predictions")), _number(test.get("eligible")), _text(latest.get("date")), direction, _number(latest.get("score"))])
        dates.append(identity + [_text(record.get(k)) for k in ("fit_start", "fit_end", "cal_start", "cal_end", "test_start", "test_end")])
        unfiltered = _mapping(record.get("test_without_abstention"))
        record_comparisons.append(identity + comparison({**test,
            "unfiltered_accuracy": unfiltered.get("accuracy"),
            "unfiltered_predictions": unfiltered.get("predictions"),
            "unfiltered_correct": unfiltered.get("correct")}))
        for key, label in (("calibration", "校准"), ("test", "测试"),
                           ("test_without_abstention", "测试：0.50 对照")):
            values = _mapping(record.get(key))
            metrics.append(identity + [label] + [_number(values.get(k), p) for k, _, p in _METRICS])
        model = _mapping(record.get("model"))
        rules = model.get("rules")
        rule_text = "；".join(_text(r) for r in rules) if isinstance(rules, (list, tuple)) else _text(rules)
        models.append(identity + [_number(model.get("k")), _number(model.get("threshold")), rule_text or "—"])
    sections = [
        ("总体测试表现（微观加权）", summary_headers, summary_rows),
        ("弃权与 0.50 对照（微观加权、描述性）", comparison_headers, [comparison(totals)]),
        ("按预测周期汇总", ["周期"] + summary_headers, horizon_rows),
        ("逐项弃权对照", identity_headers + comparison_headers, record_comparisons),
        ("本次排除规则（excluded_rules）", ["字段", "内容"], [["excluded_rules", _text(data.get("excluded_rules"))]]),
        ("标的与最新方向", identity_headers + ["测试准确率", "测试覆盖率", "测试预测 n", "测试可评估样本", "最新日期", "最新方向", "最新分数"], overview),
        ("时间分段与净化边界", identity_headers + ["拟合开始", "拟合结束", "校准开始", "校准结束", "测试开始", "测试结束"], dates),
        ("校准与测试完整指标", identity_headers + ["分段"] + [label for _, label, _ in _METRICS], metrics),
        ("冻结模型参数", identity_headers + ["top-k", "弃权阈值", "规则"], models),
    ]
    generated = _text(data.get("generated_at"))
    method = _text(data.get("method", "accuracy_abstention_v1"))
    config = json.dumps(data.get("config") or {}, ensure_ascii=False, indent=2, default=str)
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in _NOTES)
    content = "".join(f"<section><h2>{html.escape(title)}</h2>{_table(headers, rows)}</section>" for title, headers, rows in sections)
    document = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>方向准确率与中性弃权报告</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#182638;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1480px;margin:32px auto;padding:0 24px}h1{margin-bottom:4px}h2{font-size:20px}section,.notice{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px;margin:20px 0}.meta{color:#586779}.notice{border-left:5px solid #c58123}li{margin:8px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e8f0;vertical-align:top;min-width:80px;overflow-wrap:anywhere}th{background:#edf3fa;white-space:nowrap}tr:hover{background:#f8fafc}input{font:inherit;border:1px solid #9aacc1;border-radius:6px;padding:10px;width:min(100%,560px)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:16px}small{color:#586779}@media print{input,.search{display:none}main{margin:0;padding:0}section{break-inside:auto}.table-wrap{overflow:visible}th,td{padding:4px;font-size:10px}}
</style></head><body><main>'''
    document += f'<h1>方向准确率与中性弃权报告</h1><p class="meta">生成时间：{html.escape(generated)} · 方法：{html.escape(method)}</p>'
    document += f'<aside class="notice"><h2>口径与限制（请先阅读）</h2><ul>{notes_html}</ul></aside>'
    document += '<div class="search"><label for="search">搜索表格（代码、名称、状态、规则等）</label><br><input id="search" type="search" placeholder="输入关键词；清空恢复全部行"><br><small>仅筛选展示行，不改变总体指标或汇总分母。</small><p id="search-status" aria-live="polite"></p></div>'
    document += content + f'<section><h2>运行配置</h2><pre>{html.escape(config)}</pre></section>'
    document += '''</main><script>
const search=document.getElementById('search');
search.addEventListener('input',()=>{const q=search.value.trim().toLocaleLowerCase();let shown=0,total=0;document.querySelectorAll('tbody tr').forEach(row=>{const match=!q||row.textContent.toLocaleLowerCase().includes(q);row.hidden=!match;total++;if(match)shown++;});document.getElementById('search-status').textContent=q?`显示 ${shown} / ${total} 行（汇总值不变）`:'';});
</script></body></html>'''
    markdown = ["# 方向准确率与中性弃权报告", f"生成时间：{_md(generated)}", f"方法：{_md(method)}", "", "## 口径与限制", ""]
    markdown.extend(f"- {note}" for note in _NOTES)
    for title, headers, rows in sections:
        markdown.extend(["", f"## {title}", "", _table(headers, rows, markdown=True)])
    # Indentation avoids user-supplied backticks ending a Markdown code fence.
    markdown.extend(["", "## 运行配置", "", *["    " + line for line in config.splitlines()], ""])
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "report_accuracy.html", output / "ACCURACY.md"]
    paths[0].write_text(document, encoding="utf-8", newline="\n")
    paths[1].write_text("\n".join(markdown), encoding="utf-8", newline="\n")
    return paths
