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


def cmd_table(args) -> int:
    from .fullfit_table import build_from_json

    cfg = _load_cfg(args)
    src = cfg.report_path / "fullfit" / "fullfit.json"
    if not src.exists():
        print(f"找不到 {src}，请先运行 fullfit", file=sys.stderr)
        return 2
    print(build_from_json(src, cfg=cfg))
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


def _serve(cfg, broker, market_fn, interval_s: int) -> int:
    import signal
    import time

    from .crypto_supervisor import run_supervisor

    stop = {"flag": False}

    def _stop(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def sleep_fn(seconds: int) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            if stop["flag"]:
                return False
            time.sleep(min(1.0, max(0.0, end - time.time())))
        return not stop["flag"]

    print("监督进程启动。账户不是独占账户。杠杆不是 1 或存在他人仓单时不开新仓。", flush=True)
    return run_supervisor(
        cfg, broker, market_fn, sleep_fn=sleep_fn, now_fn=lambda: int(time.time() * 1000),
        stop_fn=lambda: stop["flag"], interval_s=interval_s,
    )


def _supervise(cfg, client, interval_s: int) -> int:
    import time

    from .bybit_public import bars_to_frame, default_getter, fetch_klines
    from .crypto_live import FROZEN_UNIVERSE, MarketView, closed_daily_view
    from .crypto_xs import prepare_panels

    def market_fn():
        now_ms = int(time.time() * 1000)
        try:
            getter = default_getter()
            frames = {}
            for symbol in FROZEN_UNIVERSE:
                bars = fetch_klines("linear", symbol, as_of_ms=now_ms, getter=getter)
                frames[symbol] = bars_to_frame(bars)
            _open, close, turnover = prepare_panels(frames)
            return closed_daily_view(close, turnover, now_ms)
        except Exception:
            return MarketView(0, True)

    return _serve(cfg, client, market_fn, interval_s)


def cmd_crypto_xs_live(args) -> int:
    """实盘入口。未解锁时只打印阻断项，不创建私有客户端。"""
    import os
    import time

    from .bybit_private import BybitPrivateClient, load_credentials, urllib_transport
    from .bybit_public import BybitPublicError, default_getter, fetch_klines
    from .crypto_live import (
        FROZEN_UNIVERSE, Ledger, MarketView, closed_daily_view, load_live_config,
        readiness, run_cycle,
    )

    try:
        cfg = load_live_config(args.config, os.environ)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = readiness(cfg)
    if args.preflight or not (args.once or args.monitor or args.supervise):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if report["entries_blocked"] else 0
    if args.supervise and args.interval < 1:
        print("监督间隔必须至少 1 秒", file=sys.stderr)
        return 2
    if not cfg.entries_opted_in:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("开仓条件没有满足，未连接私有交易接口。", file=sys.stderr)
        return 2
    fixture = os.environ.get("SUPER_TRADER_OFFLINE_FIXTURE", "").strip()
    if fixture:
        if not args.supervise:
            print("离线夹具只能用于 --supervise 进程测试", file=sys.stderr)
            return 2
        from .crypto_supervisor import offline_fixture
        broker, market_fn = offline_fixture(fixture)
        return _serve(cfg, broker, market_fn, args.interval)
    try:
        credentials = load_credentials(os.environ)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = BybitPrivateClient(
        credentials, urllib_transport(), lambda: int(time.time() * 1000),
    )
    if args.supervise:
        return _supervise(cfg, client, args.interval)
    ledger = Ledger.load(cfg.ledger_path)
    now_ms = int(time.time() * 1000)
    if args.monitor:
        market = MarketView(0, True)
    else:
        try:
            getter = default_getter()
            frames = {}
            for symbol in FROZEN_UNIVERSE:
                bars = fetch_klines("linear", symbol, as_of_ms=now_ms, getter=getter)
                from .bybit_public import bars_to_frame
                frames[symbol] = bars_to_frame(bars)
            from .crypto_xs import prepare_panels
            _open, close, turnover = prepare_panels(frames)
            market = closed_daily_view(close, turnover, now_ms)
        except (BybitPublicError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    result = run_cycle(cfg, ledger, client, market, now_ms)
    public = {
        "action": result["action"],
        "blockers": result["blockers"],
        "pnl": result["pnl"],
        "budget": result["budget"],
        "orders": [order.symbol + ":" + order.side for order in result["orders"]],
        "cancels": len(result["cancels"]),
    }
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return 0 if result["action"] in {"rebalanced", "monitored", "reduced", "no_trade", "halted"} else 2


def cmd_crypto_xs(args) -> int:
    """加密货币横截面纸面路径。--live 只打印关闭原因，不会连接私有交易接口。"""
    import json

    from .bybit_public import BybitPublicError, fetch_klines, fetch_launch_ms, frames_from_payload
    from .crypto_xs import (
        CryptoXSConfig, PaperLimits, Policy, live_block_reason, prepare_panels,
        research_config, run_crypto_xs,
    )

    if args.live:
        print(live_block_reason({
            "universe": args.confirm_universe,
            "category": args.confirm_category,
            "funding": args.confirm_funding,
            "leverage": args.confirm_leverage,
            "loss_limit": args.confirm_loss_limit,
        }))
        print("实盘入口是 crypto-xs-live。纸面这条 --live 不会下单，也不会读取密钥。")
        return 2

    pinned_flags = [args.lookback, args.horizon, args.top_k, args.direction]
    if any(flag is not None for flag in pinned_flags) and not all(flag is not None for flag in pinned_flags):
        print("固定规则时，--lookback、--horizon、--top-k、--direction 必须同时给出", file=sys.stderr)
        return 2
    symbols = tuple(part.strip().upper() for part in args.symbols.split(",") if part.strip()) if args.symbols else None
    try:
        cfg = research_config(args.category, symbols)
        limits = PaperLimits(
            max_name_weight=args.max_name_weight,
            max_gross=args.max_gross,
            max_net=args.max_net,
        )
        cfg = CryptoXSConfig(
            category=cfg.category,
            symbols=cfg.symbols,
            paper_equity_usdt=args.paper_equity,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            min_history=args.min_history,
            min_names=args.min_names,
            min_median_turnover=args.min_turnover,
            liquidity_window=args.liquidity_window,
            lookbacks=tuple(args.lookbacks) if args.lookbacks else cfg.lookbacks,
            horizons=tuple(args.horizons) if args.horizons else cfg.horizons,
            top_ks=tuple(args.top_ks) if args.top_ks else cfg.top_ks,
            holding_bars=args.rebalance_every,
            limits=limits,
            universe_source=cfg.universe_source,
        )
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    as_of_ms = int(args.as_of_ms) if args.as_of_ms is not None else int(datetime.now().timestamp() * 1000)
    try:
        if args.bars_json:
            payload = json.loads(Path(args.bars_json).read_text(encoding="utf-8"))
            if args.as_of_ms is None and payload.get("as_of_ms") is not None:
                as_of_ms = int(payload["as_of_ms"])
            frames = frames_from_payload(payload)
        else:
            from .bybit_public import default_getter
            getter = default_getter()
            frames = {}
            for symbol in cfg.symbols:
                launch = fetch_launch_ms(cfg.category, symbol, getter)
                bars = fetch_klines(cfg.category, symbol, as_of_ms=as_of_ms, getter=getter)
                from .bybit_public import bars_to_frame
                frames[symbol] = bars_to_frame(bars, launch_ms=launch)
        present = {
            symbol: frame for symbol, frame in frames.items()
            if symbol in cfg.symbols and len(frame)
        }
        missing = [symbol for symbol in cfg.symbols if symbol not in present]
        if missing:
            print("以下交易对没有已收盘 K 线: " + ", ".join(missing), file=sys.stderr)
            return 2
        open_, close, turnover = prepare_panels(present)
        pinned = None
        if args.direction is not None:
            pinned = Policy(
                args.lookback, args.horizon, args.top_k, args.direction,
                rebalance_every=args.rebalance_every or args.horizon,
            )
        report = run_crypto_xs(open_, close, turnover, cfg, as_of_ms=as_of_ms, pinned=pinned)
    except (BybitPublicError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out = Path(args.report) if args.report else Path("reports") / "crypto_xs" / f"{cfg.category}_paper.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    policy = report["policy"]
    print("加密货币横截面纸面结果（未下单）")
    print(f"  category: {cfg.category}    做空: {'允许' if report['allows_short'] else '不允许，现货只做多和现金'}")
    print(f"  宇宙来源: {cfg.universe_source}    交易对: {', '.join(cfg.symbols)}")
    print(f"  {report['survivorship']}")
    print(f"  {report['a_share_accuracy']}")
    if policy is None:
        print("  规则: 拟合/校准没有选出可执行的横截面规则，保持空仓")
    else:
        print(
            f"  规则: {policy['direction']} lookback={policy['lookback']} "
            f"hold={policy['rebalance_every']} top_k={policy['top_k']} "
            f"选择方式={report['selection']['mode']}"
        )
    test = report["test_simulation"]
    if test:
        ending = test["ending_equity"]
        ending_text = "无法盯市" if ending is None else f"{ending:.4f}"
        print(
            f"  测试段纸面记账: 再平衡 {test['rebalances']} 次, "
            f"费用 {test['fees']:.4f}, 成交额 {test['traded_notional']:.4f}, "
            f"换手 {test['turnover']:.4f}, 期末权益 {ending_text}"
        )
        print(f"  {test['note']}")
    latest = report["latest"]
    print(f"  最新已收盘目标: {latest['reason']}  权重 {latest['target_weights'] or '无'}")
    print(f"  {report['live_block_reason']}")
    print(f"  报告: {out}")
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

    tb = sub.add_parser("table", help="把 fullfit 结果做成按类型区分的全市场明细 HTML")
    tb.set_defaults(func=cmd_table)

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

    cx = sub.add_parser(
        "crypto-xs",
        help="Bybit 加密货币横截面纸面交易（公开日线，不下单）",
    )
    cx.add_argument("--category", required=True, choices=["spot", "linear", "inverse"],
                    help="必须写明。spot 只做多；linear/inverse 才允许多空")
    cx.add_argument("--symbols", default=None,
                    help="逗号分隔的交易对。不写时，spot/linear 使用研究默认宇宙；inverse 必须写")
    cx.add_argument("--bars-json", default=None, help="本地已收盘 K 线 JSON，用于离线纸面，不访问交易所")
    cx.add_argument("--as-of-ms", type=int, default=None, help="判定收盘和过期的 UTC 毫秒时间")
    cx.add_argument("--paper-equity", type=float, default=10_000.0,
                    help="纸面记账权益（USDT）。不是实盘资金")
    cx.add_argument("--fee-bps", type=float, default=10.0, help="研究用费率假设，不是账户实际费率")
    cx.add_argument("--slippage-bps", type=float, default=5.0, help="研究用滑点假设")
    cx.add_argument("--min-history", type=int, default=60)
    cx.add_argument("--min-names", type=int, default=6)
    cx.add_argument("--min-turnover", type=float, default=1_000_000.0, help="过去窗口成交额中位数下限，USDT")
    cx.add_argument("--liquidity-window", type=int, default=20)
    cx.add_argument("--lookbacks", type=int, nargs="+", default=None)
    cx.add_argument("--horizons", type=int, nargs="+", default=None)
    cx.add_argument("--top-ks", type=int, nargs="+", default=None)
    cx.add_argument("--lookback", type=int, default=None, help="与 horizon/top-k/direction 一起固定规则")
    cx.add_argument("--horizon", type=int, default=None)
    cx.add_argument("--top-k", type=int, default=None)
    cx.add_argument("--direction", choices=["momentum", "reversal"], default=None)
    cx.add_argument("--rebalance-every", type=int, default=None, help="持有几根日线后再平衡，默认等于 horizon")
    cx.add_argument("--max-name-weight", type=float, default=0.5)
    cx.add_argument("--max-gross", type=float, default=1.0, help="纸面总敞口上限，1 表示不加杠杆")
    cx.add_argument("--max-net", type=float, default=1.0)
    cx.add_argument("--report", default=None, help="纸面报告 JSON 路径")
    cx.add_argument("--live", action="store_true", help="请求实盘。当前构建会拒绝并退出")
    cx.add_argument("--confirm-universe", action="store_true")
    cx.add_argument("--confirm-category", action="store_true")
    cx.add_argument("--confirm-funding", action="store_true")
    cx.add_argument("--confirm-leverage", action="store_true")
    cx.add_argument("--confirm-loss-limit", action="store_true")
    cx.set_defaults(func=cmd_crypto_xs)

    live = sub.add_parser(
        "crypto-xs-live",
        help="Bybit 线性合约横截面实盘。默认只做核验，杠杆不是 1 时不开仓",
    )
    live.add_argument("--config", default="config/crypto_xs_live.example.json")
    live.add_argument("--preflight", action="store_true", help="只打印阻断项，不连接交易所")
    live.add_argument("--once", action="store_true", help="跑一轮。条件不满足时不会下单")
    live.add_argument("--monitor", action="store_true", help="只做一次风险检查，不新开横截面篮子")
    live.add_argument("--supervise", action="store_true",
                      help="持续监督：UTC 日线收盘后再平衡，其间只盯 20 USDT 亏损")
    live.add_argument("--interval", type=int, default=60, help="监督循环间隔秒数，默认 60")
    live.set_defaults(func=cmd_crypto_xs_live)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
