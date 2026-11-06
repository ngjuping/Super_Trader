"""命令行入口：scan / report / predict / check / rules / placebo。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from dataclasses import asdict
from datetime import datetime


def _load_cfg(args) -> Config:
    cfg = Config.load(args.config)
    if getattr(args, "data_root", None):
        cfg = cfg.merged({"data_root": args.data_root})
    if getattr(args, "report_dir", None):
        cfg = cfg.merged({"report_dir": args.report_dir})
    return cfg


def cmd_scan(args) -> int:
    from .scan import run_scan

    cfg = _load_cfg(args)
    outcome = run_scan(
        cfg,
        pool=args.pool,
        top=args.top,
        out_name=args.name,
        limit=args.limit,
        verbose=not args.quiet,
    )
    if not args.no_report:
        from .report import build_reports

        paths = build_reports(outcome.payload, outcome.out_dir)
        print("报告：")
        for p in paths:
            print(f"  {p}")
    return 0


def cmd_report(args) -> int:
    from .report import build_reports

    cfg = _load_cfg(args)
    name = args.name or cfg.scan_name
    out_dir = cfg.report_path / name
    res = out_dir / "results.json"
    if not res.exists():
        print(f"找不到 {res}，请先运行 scan", file=sys.stderr)
        return 2
    payload = json.loads(res.read_text(encoding="utf-8"))
    for p in build_reports(payload, out_dir):
        print(p)
    return 0


def cmd_predict(args) -> int:
    from .predict import predict_today

    cfg = _load_cfg(args)
    return predict_today(cfg, args.name, only=args.symbol, top=args.top)


def cmd_check(args) -> int:
    from .diagnose import data_check

    cfg = _load_cfg(args)
    return data_check(cfg)


def cmd_rules(args) -> int:
    from .rules import RULES

    rows = sorted(RULES, key=lambda r: (r.family, r.name))
    print(f"共 {len(rows)} 条规则\n")
    fam = None
    for r in rows:
        if r.family != fam:
            fam = r.family
            print(f"【{fam}】")
        print(f"  {r.name:<26s} {r.desc}")
    return 0


def cmd_personalize(args) -> int:
    from .personal_report import build_personal_report
    from .personalize import run_personalize

    cfg = _load_cfg(args)
    payload = run_personalize(cfg, pool=args.pool, reps=args.reps, seed=args.seed)
    out_dir = cfg.report_path / "personalized"
    for p in build_personal_report(payload, out_dir):
        print(f"  {p}")
    return 0


def cmd_accuracy(args) -> int:
    from .accuracy import AccuracyConfig, run_accuracy
    config = AccuracyConfig(min_fit_signals=args.min_fit_signals,
                            min_calibration_predictions=args.min_calibration_predictions)
    run_accuracy(_load_cfg(args), pool=args.pool, config=config, limit=args.limit)
    return 0


def cmd_fullfit(args) -> int:
    from .fullfit import FullFitConfig, run_fullfit
    from .fullfit_report import build_fullfit_html

    cfg = _load_cfg(args)
    config = FullFitConfig(half_life_days=args.half_life, target_accuracy=args.target,
                           max_rules=args.max_rules, min_eff_n=args.min_eff_n,
                           weight_floor=args.weight_floor, min_coverage=args.min_coverage)
    payload = run_fullfit(cfg, pool=args.pool, config=config, limit=args.limit,
                          workers=args.workers,
                          progress_path=str(cfg.report_path / "fullfit" / "progress.txt"))
    out = build_fullfit_html(payload, cfg.report_path / "fullfit" / "ranking_fullfit.html")
    print(out)
    return 0


def cmd_crosssection(args) -> int:
    from .crosssection import XSConfig, evaluate, load_panel
    from .crosssection_report import build_xs_html

    cfg = _load_cfg(args)
    config = XSConfig(start=args.start, horizons=tuple(args.horizons),
                      lookbacks=tuple(args.lookbacks))
    out_dir = cfg.report_path / "crosssection"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"读取面板（{args.start} 起，最少 {config.min_bars} 根）…", flush=True)
    panel, meta = load_panel(cfg, config, cache=out_dir / "panel.parquet")
    print(f"面板 {panel.shape[0]} 交易日 × {panel.shape[1]} 标的", flush=True)
    results = evaluate(panel, meta, config)
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "method": "cross_section_rs_v1", "config": asdict(config),
               "results": results}
    (out_dir / "crosssection.json").write_text(
        json.dumps(payload, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8", newline="\n")
    print(build_xs_html(payload, out_dir / "report_crosssection.html"))
    for r in results["ic"]:
        print(f"  {r['feature']:<8s} H{r['horizon']:<2d} IC={r['ic_mean']:+.4f} "
              f"t={r['ic_t']:+.2f} 天数={r['days']}", flush=True)
    return 0


def cmd_ranking(args) -> int:
    from .accuracy_ranking import build_from_json

    cfg = _load_cfg(args)
    src = cfg.report_path / "accuracy" / "accuracy.json"
    if not src.exists():
        print(f"找不到 {src}，请先运行 accuracy", file=sys.stderr)
        return 2
    print(build_from_json(src))
    return 0


def cmd_placebo(args) -> int:
    from .diagnose import placebo_test

    cfg = _load_cfg(args)
    return placebo_test(cfg, n_symbols=args.symbols, seed=args.seed, reps=args.reps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="super-trader",
        description="A 股日线「未来涨跌」量化规律挖掘（研究工具，不构成投资建议）",
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.json）")
    p.add_argument("--data-root", default=None, help="覆盖日线数据目录")
    p.add_argument("--report-dir", default=None, help="覆盖报告输出目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="跑规则库扫描并生成报告")
    s.add_argument("--pool", default="mainstream",
                   help="标的池：mainstream / 宽基 / 行业 / ETF / 龙头 / auto")
    s.add_argument("--top", type=int, default=None, help="auto 池取前 N 个")
    s.add_argument("--limit", type=int, default=None, help="只跑前 N 个标的（调试用）")
    s.add_argument("--name", default=None, help="本次扫描名（输出目录名）")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--no-report", action="store_true", help="只出数据，不生成 HTML")
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("report", help="用已有 results.json 重新生成报告")
    r.add_argument("--name", default=None)
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("predict", help="打印每个标的最新规则信号（今日怎么看）")
    d.add_argument("--name", default=None)
    d.add_argument("--symbol", default=None, help="只看某个代码/名称")
    d.add_argument("--top", type=int, default=30, help="最多显示 N 个标的")
    d.set_defaults(func=cmd_predict)

    c = sub.add_parser("check", help="数据体检：行数、时间范围、除权跳变、停牌缺口")
    c.set_defaults(func=cmd_check)

    ru = sub.add_parser("rules", help="列出规则库")
    ru.set_defaults(func=cmd_rules)

    pl = sub.add_parser("placebo", help="安慰剂检验：打乱收益标签后看还剩多少『规律』")
    pl.add_argument("--symbols", type=int, default=0, help="用前 N 个标的（0=全部）")
    pl.add_argument("--reps", type=int, default=3, help="重复轮数")
    pl.add_argument("--seed", type=int, default=7)
    pl.set_defaults(func=cmd_placebo)

    per = sub.add_parser("personalize", help="个性化规则：每个标的各选各的规则（4 段复现验证）")
    per.add_argument("--pool", default="mainstream")
    per.add_argument("--reps", type=int, default=2, help="安慰剂重复轮数")
    per.add_argument("--seed", type=int, default=17)
    per.set_defaults(func=cmd_personalize)

    acc = sub.add_parser("accuracy", help="准确率优先：看涨/看跌/中性，按时间分拟合、校准和测试")
    acc.add_argument("--pool", default="mainstream")
    acc.add_argument("--limit", type=int, default=None)
    acc.add_argument("--min-fit-signals", type=int, default=60)
    acc.add_argument("--min-calibration-predictions", type=int, default=30)
    acc.set_defaults(func=cmd_accuracy)

    xs = sub.add_parser("xs", help="横截面：相对强弱/行业轮动 IC 检验")
    xs.add_argument("--start", default="2015-01-01")
    xs.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    xs.add_argument("--lookbacks", type=int, nargs="+", default=[5, 20, 60])
    xs.set_defaults(func=cmd_crosssection)

    rank = sub.add_parser("ranking", help="按标的历史预测准确率生成排行 HTML（中性不计入分母）")
    rank.set_defaults(func=cmd_ranking)

    ff = sub.add_parser("fullfit", help="全历史拟合（不分验证/测试集，近期加权）并输出排行 HTML")
    ff.add_argument("--pool", default="mainstream")
    ff.add_argument("--limit", type=int, default=None)
    ff.add_argument("--half-life", type=float, default=365.0, help="权重半衰期（天）")
    ff.add_argument("--weight-floor", type=float, default=0.10, help="老数据权重下限")
    ff.add_argument("--min-coverage", type=float, default=0.10, help="集成最低覆盖率")
    ff.add_argument("--min-eff-n", type=float, default=30.0, help="有效样本量下限")
    ff.add_argument("--target", type=float, default=0.55)
    ff.add_argument("--max-rules", type=int, default=5)
    ff.add_argument("--workers", type=int, default=1, help="并行进程数（全市场建议 8）")
    ff.set_defaults(func=cmd_fullfit)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
