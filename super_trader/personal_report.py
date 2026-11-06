"""探索性个性化规则报告，明确区分周期选择与统计功效。"""
from __future__ import annotations

from html import escape
from pathlib import Path
import math


def _num(value, scale=1, digits=2):
    try:
        v = float(value) * scale
        return f"{v:.{digits}f}" if math.isfinite(v) else "—"
    except (ValueError, TypeError):
        return "—"


def build_personal_report(payload: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = payload.get('totals', {})
    diagnostics = payload.get('horizon_diagnostics', [])
    candidates = payload.get('per_symbol_best', [])
    gate = payload.get('gate', {})
    md = [
        '# 个性化规则：周期偏向修正版', '',
        f"生成：{payload.get('generated_at', '')}；方法：{payload.get('method_revision', '')}", '',
        '## 修正内容', '',
        '- 每条规则只在训练段选择一个持有周期，按方向化毛超额收益÷持有天数排序；不按留出命中率或p值选周期。',
        '- 多头、空头及双向信号均使用同一评分公式；纯多头评分不再恒为零。',
        '- 训练基准仅来自训练段。训练/留出及各折边界按原始K线位置净化，标签终点不能跨界。',
        '- 功效单独披露，不新增强制长周期入选的放宽门槛，也不以功效提示再次淘汰长周期。',
        '- H1/H3/H5/H10/H20是日线持有期，不是日线/60分钟这样的K线频率。', '',
        '## 周期诊断', '',
        '|持有期|候选检验|训练段选中|最终候选|原始样本中位数|n_eff中位数|最小可检出优势(pp)|功效提示数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for d in diagnostics:
        md.append(f"|H{d['horizon']}|{d['tests']}|{d['chosen_train']}|{d['validated']}|"
                  f"{_num(d['median_n_hold'], digits=0)}|{_num(d['median_n_eff'])}|"
                  f"{_num(d['median_mdes'],100)}|{d['power_limited']}|")
    md += ['', 'n_eff=n/h是保守近似；最小可检出优势按双侧5%显著性、80%功效的正态近似计算，未计入基准估计误差。',
           f"功效提示使用 n_eff<{gate.get('min_power_eff')} 或该门槛>{_num(gate.get('max_mdes'),100)}pp，仅提示证据薄弱。",
           '长周期独立信息较少是数据限制，不能通过调低显著性门槛消除。H1仍占多数不自动意味着选择偏向未修复。', '',
           '## 探索性候选', '',
           f"{totals.get('validated_pairs',0)}个规则-标的-周期候选，覆盖{totals.get('validated_symbols',0)}个标的。", '',
           '|标的|类别|规则|持有期|留出方向优势(pp)|四段优势(pp)|n_eff|训练评分(bp/日)|',
           '|---|---|---|---|---:|---|---:|---:|']
    for r in candidates:
        folds = ' / '.join(_num(v,100) for v in r.get('fold_edges', []))
        md.append(f"|{r['name']} ({r['code']})|{r['category']}|{r['rule']}|H{r['horizon']}|"
                  f"{_num(r['edge_hold'],100)}|{folds}|{_num(r.get('n_eff_hold'))}|"
                  f"{_num(r.get('daily_train'),10000)}|")
    md += ['', '## 不能据此宣称的结论', '',
           '- 这些历史数据已被反复查看、筛选和修改口径，不再是未触碰的确认性留出集。候选需要未来新数据验证。',
           '- 净化边界消除前瞻窗口交叉，但四段不因此统计独立；n/h没有精确建模信号聚集。',
           '- 同时尝试多条规则仍有多重检验问题；候选数量不是已证实规律数量。',
           f"- 段内同步置换标签与收益的诊断计数：{totals.get('placebo_runs', [])}；"
           '该置换破坏时间相关性，仅作敏感性诊断，不是严格假发现率，也不能据此证明个体差异。',
           '- 原先+1.01pp与安慰剂+4.83pp不是匹配的无偏增益估计；不再将其相减或据此认定17个标的值得差异化。',
           '- 训练评分是方向化毛超额收益代理，不是扣费净利润；未建模重叠持仓、容量及A股T+1。',
           '- 长周期功效不足应解释为证据不足，不应解释为不存在规律。', '']
    markdown = '\n'.join(md)
    # Minimal self-contained renderer: data are escaped, no external assets.
    parts = []
    table = []
    def flush():
        if not table:
            return
        heads = [escape(x.strip()) for x in table[0].strip('|').split('|')]
        rows = table[2:]
        parts.append('<div class="scroll"><table><thead><tr>' + ''.join('<th>'+x+'</th>' for x in heads) + '</tr></thead><tbody>')
        for row in rows:
            parts.append('<tr>'+''.join('<td>'+escape(x.strip())+'</td>' for x in row.strip('|').split('|'))+'</tr>')
        parts.append('</tbody></table></div>')
        table.clear()
    for line in md:
        if line.startswith('|'):
            table.append(line)
            continue
        flush()
        if line.startswith('# '): parts.append('<h1>'+escape(line[2:])+'</h1>')
        elif line.startswith('## '): parts.append('<h2>'+escape(line[3:])+'</h2>')
        elif line: parts.append('<p>'+escape(line)+'</p>')
    flush()
    html = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Super_Trader · 周期偏向修正</title><style>
body{margin:0;background:#f6f7f9;color:#172033;font:15px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1300px;margin:auto;padding:30px}h1{color:#1f3a8a}h2{border-bottom:2px solid #dbe3ee;padding-top:20px}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #e5e7eb;text-align:left}th{background:#eaf0fa;white-space:nowrap}p{max-width:1150px}</style></head><body><main>'''+''.join(parts)+'</main></body></html>'
    html_path = out / 'report_personalized.html'
    md_path = out / 'PERSONALIZED.md'
    html_path.write_text(html, encoding='utf-8', newline='\n')
    md_path.write_text(markdown, encoding='utf-8', newline='\n')
    return [html_path, md_path]
