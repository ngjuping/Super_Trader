"""Bybit 加密货币横截面纸面交易。

这一层沿用 A 股横截面的问题：同一时刻比较谁强谁弱，而不是给单个币做择时。
可执行路径只使用已收盘的 UTC 日线，信号在收盘形成，成交假设在下一根开盘。
方向只在拟合段锁定，持有参数只在校准段挑选，测试段不再改规则。

这里没有全历史翻转，没有 A 股涨跌停，也没有中国交易日历。
纸面资金、费用和权重上限是本仓库写下的研究默认值，不是用户确认过的实盘配置。
本模块不发送订单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np
import pandas as pd

from .bybit_public import (
    DAY_MS, allows_short, is_stale, require_category,
)
from .crosssection import ic_series

SURVIVORSHIP_NOTE = (
    "交易对列表是调用方给出的、或本仓库写下的「当前仍在名单上的」研究宇宙。"
    "已经下架的名字不在里面。上市时间或第一根已收盘 K 线之前不会回填价格。"
    "每天的资格用当天及以前的数据判断。这不是无幸存者偏差的宇宙。"
)
A_SHARE_NOTE = (
    "仓库里保存的 A 股方向准确率 50.38% 不适用于这条加密货币横截面，本报告不使用该数字。"
)
ISOLATION_NOTE = (
    "纸面结果只写在本地报告里。它不读取密钥，不查询持仓，不撤销或接管任何已有订单，"
    "包括 Bybit 上已有的 XAUT 与 SOL 策略。研究默认宇宙里的 SOLUSDT 只参与公开行情排名。"
)
RESEARCH_UNIVERSE_USDT = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
)
REQUIRED_LIVE_CONFIRMATIONS = (
    "universe", "category", "funding", "leverage", "loss_limit",
)
ACCOUNT_WIDE_ACTIONS = frozenset({
    "cancel_all", "cancelall", "close_all", "closeall",
    "cancel_all_orders", "close_all_positions",
})


class LiveTradingDisabled(RuntimeError):
    """实盘通道关闭。不是下单失败后的重试信号。"""


class IsolationError(ValueError):
    """订单超出本次交易对宇宙，或试图做全账户撤单/平仓。"""


@dataclass(frozen=True)
class PaperLimits:
    """纸面敞口上限。约束的是目标权重，不是交易所杠杆。"""

    max_name_weight: float = 0.5
    max_gross: float = 1.0
    max_net: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.max_name_weight <= 1:
            raise ValueError("max_name_weight 必须在 (0, 1] 内")
        if self.max_gross <= 0 or self.max_net <= 0:
            raise ValueError("max_gross 和 max_net 必须为正")


@dataclass(frozen=True)
class CryptoXSConfig:
    """研究/纸面配置。category 无默认值，leverage 不是本配置的字段。"""

    category: str
    symbols: tuple[str, ...]
    paper_equity_usdt: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    min_history: int = 60
    min_names: int = 6
    min_median_turnover: float = 1_000_000.0
    liquidity_window: int = 20
    fit_fraction: float = 0.50
    calibration_fraction: float = 0.25
    lookbacks: tuple[int, ...] = (5, 20, 60)
    horizons: tuple[int, ...] = (1, 5, 20)
    top_ks: tuple[int, ...] = (1, 2, 3)
    min_fit_days: int = 30
    min_cal_rebalances: int = 3
    holding_bars: int | None = None
    max_staleness_ms: int = 2 * DAY_MS
    limits: PaperLimits = field(default_factory=PaperLimits)
    universe_source: str = "caller"

    def __post_init__(self) -> None:
        require_category(self.category)
        if not self.symbols:
            raise ValueError("至少要有一个交易对")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("交易对列表有重复")
        if self.paper_equity_usdt <= 0:
            raise ValueError("纸面权益必须为正；这不是实盘资金")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("费用和滑点不能为负")
        if self.min_history < 2 or self.min_names < 2 or self.liquidity_window < 1:
            raise ValueError("历史长度、横截面个数或流动性窗口不合法")
        if not 0 < self.fit_fraction < 1 or not 0 < self.calibration_fraction < 1:
            raise ValueError("拟合/校准比例必须在 (0, 1) 内")
        if self.fit_fraction + self.calibration_fraction >= 1:
            raise ValueError("拟合段加校准段必须给测试段留出位置")
        for name, values in (("lookback", self.lookbacks), ("horizon", self.horizons),
                             ("top_k", self.top_ks)):
            if not values or any(int(v) < 1 for v in values):
                raise ValueError(f"{name} 必须是正整数")
        if self.min_fit_days < 1 or self.min_cal_rebalances < 1:
            raise ValueError("拟合天数和校准再平衡次数必须为正")
        if self.holding_bars is not None and self.holding_bars < 1:
            raise ValueError("持有根数必须为正")
        if self.max_staleness_ms < DAY_MS:
            raise ValueError("过期阈值至少要覆盖一根日线")
        if self.universe_source not in {"caller", "research_default"}:
            raise ValueError("universe_source 只能是 caller 或 research_default")


@dataclass(frozen=True)
class Policy:
    lookback: int
    horizon: int
    top_k: int
    direction: str
    rebalance_every: int
    weighting: str = "equal"

    def __post_init__(self) -> None:
        if self.direction not in {"momentum", "reversal"}:
            raise ValueError("direction 只能是 momentum 或 reversal")
        if self.weighting != "equal":
            raise ValueError("当前只实现等权；其他权重方案需要单独的政策决定")
        if min(self.lookback, self.horizon, self.top_k, self.rebalance_every) < 1:
            raise ValueError("持有与回看参数必须为正")


def research_config(category: str, symbols: tuple[str, ...] | None = None) -> CryptoXSConfig:
    """纸面研究默认值。调用方没写交易对时，现货/线性合约用当前名单，反向合约不猜。"""
    require_category(category)
    if symbols is None:
        if category == "inverse":
            raise ValueError("inverse 没有研究默认宇宙，请显式传入 --symbols")
        symbols = RESEARCH_UNIVERSE_USDT
        source = "research_default"
    else:
        source = "caller"
    return CryptoXSConfig(category=category, symbols=tuple(symbols), universe_source=source)


def live_block_reason(confirmations: dict | None = None) -> str:
    confirmations = confirmations or {}
    missing = [name for name in REQUIRED_LIVE_CONFIRMATIONS if not confirmations.get(name)]
    parts = ["实盘交易保持关闭"]
    if missing:
        parts.append("尚未由用户确认: " + ", ".join(missing))
    parts.append("本构建没有私有下单通道，不会开仓、撤单或平仓")
    return "；".join(parts)


def refuse_live(confirmations: dict | None = None) -> None:
    raise LiveTradingDisabled(live_block_reason(confirmations))


def reject_account_action(action: str) -> None:
    key = action.strip().lower().replace("-", "_")
    if key in ACCOUNT_WIDE_ACTIONS:
        raise IsolationError("拒绝全账户撤单或全账户平仓；只允许本次宇宙内的单个目标仓位")
    if key != "set_target":
        raise IsolationError("只允许 set_target；没有其他订单动作")


def client_order_id(category: str, symbol: str, signal_ms: int, weight: float) -> str:
    raw = f"{category}|{symbol}|{int(signal_ms)}|{weight:.8f}"
    return "cx" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def prepare_panels(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对齐到共同的 UTC 日线。缺的日子保持空值，不向前填充，也不用别的币补。"""
    usable = {symbol: _utc_daily(frame) for symbol, frame in frames.items() if len(frame)}
    if not usable:
        raise ValueError("没有可用的已收盘 K 线")
    symbols = tuple(sorted(usable))
    start = min(frame.index.min() for frame in usable.values())
    end = max(frame.index.max() for frame in usable.values())
    index = pd.date_range(start, end, freq="D", tz="UTC")
    def take(column: str) -> pd.DataFrame:
        data = {}
        for symbol in symbols:
            data[symbol] = usable[symbol][column].reindex(index)
        return pd.DataFrame(data, index=index)
    return take("open"), take("close"), take("turnover")


def _utc_daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx.floor("D")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def causal_relative_return(close: pd.DataFrame, turnover: pd.DataFrame,
                           lookback: int, cfg: CryptoXSConfig) -> pd.DataFrame:
    """截至当根收盘的相对强弱。滚动窗口和 shift 都只看过去。"""
    past = close / close.shift(lookback) - 1.0
    history = close.notna().rolling(cfg.min_history, min_periods=cfg.min_history).sum()
    liquidity = turnover.rolling(cfg.liquidity_window, min_periods=cfg.liquidity_window).median()
    eligible = (
        close.notna()
        & close.shift(lookback).notna()
        & (history >= cfg.min_history)
        & (liquidity >= cfg.min_median_turnover)
    )
    scored = past.where(eligible)
    counts = scored.notna().sum(axis=1)
    relative = scored.sub(scored.median(axis=1, skipna=True), axis=0)
    return relative.where(counts >= cfg.min_names)


def forward_relative(close: pd.DataFrame, horizon: int, min_names: int) -> pd.DataFrame:
    """只作为已经走完的标签。不能拿来生成当下的下单目标。"""
    future = close.shift(-horizon) / close - 1.0
    future = future.where(close.notna() & close.shift(-horizon).notna())
    counts = future.notna().sum(axis=1)
    relative = future.sub(future.median(axis=1, skipna=True), axis=0)
    return relative.where(counts >= min_names)


def ranked_symbols(row: pd.Series) -> list[str]:
    """相对强弱从高到低。分数相同则按交易对代码升序，结果与输入顺序无关。"""
    items = [(float(value), symbol) for symbol, value in row.items() if np.isfinite(value)]
    items.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in items]


def apply_exposure_limits(weights: dict[str, float], limits: PaperLimits,
                          *, allow_short: bool) -> dict[str, float]:
    capped: dict[str, float] = {}
    for symbol, weight in weights.items():
        value = float(weight)
        if not allow_short and value < 0:
            continue
        if abs(value) > limits.max_name_weight:
            value = math.copysign(limits.max_name_weight, value)
        if value != 0:
            capped[symbol] = value
    gross = sum(abs(value) for value in capped.values())
    if gross > limits.max_gross and gross > 0:
        scale = limits.max_gross / gross
        capped = {symbol: value * scale for symbol, value in capped.items()}
    net = sum(capped.values())
    if abs(net) > limits.max_net and abs(net) > 0:
        scale = limits.max_net / abs(net)
        capped = {symbol: value * scale for symbol, value in capped.items()}
    return {symbol: value for symbol, value in capped.items() if value != 0}


def target_weights(order: list[str], policy: Policy, category: str,
                   limits: PaperLimits, min_names: int) -> dict[str, float]:
    """等权。现货只有多头和现金；永续才有对称空头。不够一篮子就空仓。"""
    short_ok = allows_short(category)
    if len(order) < min_names:
        return {}
    k = policy.top_k
    if policy.direction == "momentum":
        longs = order[:k]
        shorts = order[-k:] if short_ok else []
    else:
        longs = list(reversed(order[-k:]))
        shorts = order[:k] if short_ok else []
    if short_ok:
        if len(order) < 2 * k or len(longs) < k or len(shorts) < k:
            return {}
        if set(longs) & set(shorts):
            return {}
        raw = {symbol: 0.5 / k for symbol in longs}
        raw.update({symbol: -0.5 / k for symbol in shorts})
    else:
        if len(longs) < k:
            return {}
        raw = {symbol: 1.0 / k for symbol in longs}
    return apply_exposure_limits(raw, limits, allow_short=short_ok)


@dataclass
class PaperBook:
    cash: float
    qty: dict[str, float]
    filled_ids: set[str]
    fees: float = 0.0
    traded_notional: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for symbol, qty in self.qty.items():
            if abs(qty) < 1e-12:
                continue
            price = prices.get(symbol)
            if price is None or not math.isfinite(price):
                raise ValueError(f"无法按已有价格盯市: {symbol}")
            total += qty * price
        return total


def _prices_at(row: pd.Series, symbols: list[str]) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for symbol in symbols:
        value = row[symbol]
        if not np.isfinite(value):
            return None
        out[symbol] = float(value)
    return out


def _fill_price(open_px: float, side: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return open_px * (1.0 + slip) if side == "buy" else open_px * (1.0 - slip)


def apply_targets(book: PaperBook, weights: dict[str, float], opens: dict[str, float], *,
                  decision_equity: float, fee_bps: float, slippage_bps: float,
                  category: str, signal_ms: int, allowed_symbols: tuple[str, ...]) -> dict:
    """把目标权重换成下一根开盘的成交。同一 client id 再次出现则跳过。"""
    require_category(category)
    short_ok = allows_short(category)
    allowed = set(allowed_symbols)
    unknown = [symbol for symbol in weights if symbol not in allowed]
    if unknown:
        raise IsolationError("目标超出本次宇宙: " + ", ".join(sorted(unknown)))
    if decision_equity <= 0:
        return {"applied": False, "reason": "non_positive_equity", "fills": []}

    held = [symbol for symbol, qty in book.qty.items() if abs(qty) >= 1e-12]
    names = sorted(set(weights) | set(held))
    missing = [symbol for symbol in names if symbol not in opens or not math.isfinite(opens[symbol])]
    if missing:
        return {"applied": False, "reason": "missing_open", "fills": []}

    plans = []
    for symbol in names:
        weight = float(weights.get(symbol, 0.0))
        if not short_ok and weight < 0:
            weight = 0.0
        side_for_target = "buy" if weight > 0 else "sell"
        open_px = opens[symbol]
        trade_px = _fill_price(open_px, side_for_target, slippage_bps) if weight != 0 else open_px
        target_qty = 0.0 if weight == 0 else weight * decision_equity / trade_px
        if not short_ok:
            target_qty = max(0.0, target_qty)
        current = float(book.qty.get(symbol, 0.0))
        delta = target_qty - current
        if not short_ok and current + delta < -1e-12:
            delta = -current
        order_id = client_order_id(category, symbol, signal_ms, weight)
        plans.append({
            "symbol": symbol, "weight": weight, "delta": delta, "open": open_px,
            "client_order_id": order_id, "duplicate": order_id in book.filled_ids,
        })

    fresh = [plan for plan in plans if not plan["duplicate"] and abs(plan["delta"]) > 1e-12]
    if plans and all(plan["duplicate"] for plan in plans):
        return {"applied": False, "reason": "duplicate", "fills": []}

    sells = [plan for plan in fresh if plan["delta"] < 0]
    buys = [plan for plan in fresh if plan["delta"] > 0]

    def cost(delta: float, open_px: float) -> tuple[float, float, float]:
        side = "buy" if delta > 0 else "sell"
        px = _fill_price(open_px, side, slippage_bps)
        notional = abs(delta) * px
        fee = notional * fee_bps / 10_000.0
        cash_change = -(delta * px + fee)
        return px, fee, cash_change

    for plan in sells:
        px, fee, cash_change = cost(plan["delta"], plan["open"])
        book.cash += cash_change
        book.qty[plan["symbol"]] = book.qty.get(plan["symbol"], 0.0) + plan["delta"]
        book.fees += fee
        book.traded_notional += abs(plan["delta"]) * px
        book.filled_ids.add(plan["client_order_id"])
        plan.update({"fill_price": px, "fee": fee})

    buy_costs = []
    for plan in buys:
        px, fee, _ = cost(plan["delta"], plan["open"])
        buy_costs.append(abs(plan["delta"]) * px + fee)
    total_cost = sum(buy_costs)
    scale = 1.0
    if total_cost > book.cash and total_cost > 0:
        scale = max(0.0, book.cash / total_cost)
    for plan in buys:
        plan["delta"] *= scale
        if abs(plan["delta"]) <= 1e-12:
            book.filled_ids.add(plan["client_order_id"])
            plan.update({"fill_price": None, "fee": 0.0, "scaled_to_zero": True})
            continue
        px, fee, cash_change = cost(plan["delta"], plan["open"])
        book.cash += cash_change
        book.qty[plan["symbol"]] = book.qty.get(plan["symbol"], 0.0) + plan["delta"]
        book.fees += fee
        book.traded_notional += abs(plan["delta"]) * px
        book.filled_ids.add(plan["client_order_id"])
        plan.update({"fill_price": px, "fee": fee, "scale": scale})

    for symbol, qty in list(book.qty.items()):
        if abs(qty) < 1e-10:
            book.qty[symbol] = 0.0
    if not short_ok:
        if book.cash < -1e-6 or any(qty < -1e-8 for qty in book.qty.values()):
            raise RuntimeError("现货纸面账本出现负现金或空头，这是实现错误")
    fills = [plan for plan in plans if not plan["duplicate"]]
    return {"applied": bool(fresh), "reason": "filled" if fresh else "flat", "fills": fills, "scale": scale}


def _signal_indices(n: int, start: int, end: int, step: int, first: int) -> list[int]:
    i = max(start, first)
    out = []
    while i + 1 < end:
        out.append(i)
        i += step
    return out


def simulate(open_: pd.DataFrame, close: pd.DataFrame, turnover: pd.DataFrame,
             policy: Policy, cfg: CryptoXSConfig, *, start: int, end: int) -> dict:
    """在 [start, end) 内从空仓开始做纸面再平衡。成交价是信号 bar 的下一根开盘。"""
    rel = causal_relative_return(close, turnover, policy.lookback, cfg)
    book = PaperBook(cash=cfg.paper_equity_usdt, qty={}, filled_ids=set())
    first = max(policy.lookback, cfg.min_history - 1)
    rebalances = 0
    skipped = 0
    for i in _signal_indices(len(close), start, end, policy.rebalance_every, first):
        order = ranked_symbols(rel.iloc[i])
        weights = target_weights(order, policy, cfg.category, cfg.limits, cfg.min_names)
        held = [symbol for symbol, qty in book.qty.items() if abs(qty) > 0]
        mark_names = sorted(set(held) | set(weights))
        marked = _prices_at(close.iloc[i], mark_names) if mark_names else {}
        if marked is None:
            skipped += 1
            continue
        equity = book.cash if not mark_names else book.equity(marked)
        needed = sorted(set(held) | set(weights))
        opens = _prices_at(open_.iloc[i + 1], needed) if needed else {}
        if opens is None:
            skipped += 1
            continue
        signal_ms = int(close.index[i].timestamp() * 1000)
        result = apply_targets(
            book, weights, opens, decision_equity=equity, fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps, category=cfg.category, signal_ms=signal_ms,
            allowed_symbols=tuple(close.columns),
        )
        if result["applied"]:
            rebalances += 1
        elif result["reason"] in {"missing_open", "non_positive_equity"}:
            skipped += 1
    end_mark_idx = max(start, end - 1)
    held = [symbol for symbol, qty in book.qty.items() if abs(qty) > 0]
    ending = book.cash
    if held:
        marked = _prices_at(close.iloc[end_mark_idx], held)
        ending = None if marked is None else book.equity(marked)
    return {
        "rebalances": rebalances,
        "skipped": skipped,
        "ending_equity": ending,
        "fees": book.fees,
        "traded_notional": book.traded_notional,
        "turnover": book.traded_notional / cfg.paper_equity_usdt,
        "cash": book.cash,
        "qty": {symbol: qty for symbol, qty in book.qty.items() if abs(qty) > 0},
    }


def fit_direction(relative: pd.DataFrame, future: pd.DataFrame, fit_end: int,
                  horizon: int, cfg: CryptoXSConfig) -> str | None:
    """拟合段的秩相关符号决定动量或反转。测试段不参与。均值恰好为 0 则不做。"""
    ic = ic_series(relative, future, cfg.min_names)
    if fit_end - horizon <= 0:
        return None
    mask = np.zeros(len(ic), dtype=bool)
    mask[: fit_end - horizon] = True
    used = ic.where(pd.Series(mask, index=ic.index)).dropna()
    if len(used) < cfg.min_fit_days:
        return None
    mean = float(used.mean())
    if not math.isfinite(mean) or mean == 0:
        return None
    return "momentum" if mean > 0 else "reversal"


def select_policy(open_: pd.DataFrame, close: pd.DataFrame, turnover: pd.DataFrame,
                  cfg: CryptoXSConfig) -> tuple[Policy | None, dict]:
    """方向看拟合段，参数看校准段纸面权益。把测试段价格涂掉不应改变结果。"""
    n = len(close)
    fit_end = int(n * cfg.fit_fraction)
    cal_end = int(n * (cfg.fit_fraction + cfg.calibration_fraction))
    if not (0 < fit_end < cal_end < n):
        raise ValueError("样本太短，无法分成拟合、校准和测试")
    best: Policy | None = None
    best_key = None
    candidates = []
    for lookback in cfg.lookbacks:
        relative = causal_relative_return(close, turnover, lookback, cfg)
        for horizon in cfg.horizons:
            future = forward_relative(close, horizon, cfg.min_names)
            direction = fit_direction(relative, future, fit_end, horizon, cfg)
            if direction is None:
                continue
            for top_k in cfg.top_ks:
                policy = Policy(
                    lookback, horizon, top_k, direction,
                    rebalance_every=cfg.holding_bars or horizon,
                )
                stats = simulate(open_, close, turnover, policy, cfg, start=fit_end, end=cal_end)
                record = {"policy": _policy_dict(policy), "calibration": _stats_public(stats)}
                candidates.append(record)
                if stats["rebalances"] < cfg.min_cal_rebalances or stats["ending_equity"] is None:
                    continue
                key = (
                    stats["ending_equity"],
                    -stats["traded_notional"],
                    -lookback,
                    -horizon,
                    -top_k,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = policy
    return best, {
        "mode": "fit_direction_then_calibration_equity",
        "fit_end_index": fit_end,
        "calibration_end_index": cal_end,
        "test_used_for_selection": False,
        "candidates": candidates,
    }


def latest_targets(open_: pd.DataFrame, close: pd.DataFrame, turnover: pd.DataFrame,
                   policy: Policy | None, cfg: CryptoXSConfig, as_of_ms: int) -> dict:
    """最新一根已收盘 K 线上的待执行目标。此时还没有下一根开盘，所以不成交。"""
    last = close.index[-1]
    last_ms = int(last.timestamp() * 1000)
    stale = is_stale(last_ms, as_of_ms, max_age_ms=cfg.max_staleness_ms)
    if policy is None:
        weights, reason, order = {}, "no_frozen_policy", []
    elif stale:
        weights, reason, order = {}, "stale_data", []
    else:
        relative = causal_relative_return(close, turnover, policy.lookback, cfg)
        order = ranked_symbols(relative.iloc[-1])
        weights = target_weights(order, policy, cfg.category, cfg.limits, cfg.min_names)
        reason = "pending_next_open" if weights else "no_eligible_basket"
    intents = []
    for symbol, weight in sorted(weights.items()):
        reject_account_action("set_target")
        if symbol not in close.columns:
            raise IsolationError(symbol)
        intents.append({
            "action": "set_target",
            "category": cfg.category,
            "symbol": symbol,
            "target_weight": weight,
            "signal_open_ms": last_ms,
            "client_order_id": client_order_id(cfg.category, symbol, last_ms, weight),
            "sent": False,
        })
    return {
        "as_of_ms": int(as_of_ms),
        "last_closed_open_ms": last_ms,
        "stale": stale,
        "reason": reason,
        "ranking_high_to_low": order,
        "target_weights": weights,
        "pending_intents": intents,
    }


def run_crypto_xs(open_: pd.DataFrame, close: pd.DataFrame, turnover: pd.DataFrame,
                  cfg: CryptoXSConfig, *, as_of_ms: int,
                  pinned: Policy | None = None) -> dict:
    """端到端纸面结果。数字只描述这次输入下的账本，不是实盘收益承诺。"""
    if pinned is None:
        policy, selection = select_policy(open_, close, turnover, cfg)
    else:
        policy = pinned
        selection = {
            "mode": "caller_pinned",
            "test_used_for_selection": False,
            "note": "调用方固定了规则，测试段只负责记账，不代表样本外挑选。",
        }
    n = len(close)
    cal_end = int(n * (cfg.fit_fraction + cfg.calibration_fraction))
    test = None
    if policy is not None and cal_end < n:
        test = simulate(open_, close, turnover, policy, cfg, start=cal_end, end=n)
    latest = latest_targets(open_, close, turnover, policy, cfg, as_of_ms)
    return {
        "method": "crypto_cross_section_paper_v1",
        "description": "Bybit 加密货币横截面相对强弱，纸面组合，不是单币择时。",
        "live_trading": False,
        "live_block_reason": live_block_reason({}),
        "isolation": ISOLATION_NOTE,
        "research_defaults_not_live_approved": True,
        "leverage": None,
        "live_capital_usdt": None,
        "funding_modeled": False,
        "borrow_modeled": False,
        "survivorship": SURVIVORSHIP_NOTE,
        "a_share_accuracy": A_SHARE_NOTE,
        "category": cfg.category,
        "allows_short": allows_short(cfg.category),
        "universe_source": cfg.universe_source,
        "symbols": list(cfg.symbols),
        "assumptions": {
            "clock": "UTC",
            "session": "24x7",
            "interval": "D",
            "execution": "signal at closed daily bar, fill at next bar open",
            "weighting": "equal",
            "paper_equity_usdt": cfg.paper_equity_usdt,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
            "min_history": cfg.min_history,
            "min_names": cfg.min_names,
            "min_median_turnover_usdt": cfg.min_median_turnover,
            "liquidity_window": cfg.liquidity_window,
            "max_name_weight": cfg.limits.max_name_weight,
            "max_gross": cfg.limits.max_gross,
            "max_net": cfg.limits.max_net,
            "fee_note": "fee_bps 和 slippage_bps 是研究假设，不是该账户的实际费率。",
            "paper_equity_note": "paper_equity_usdt 只是纸面记账单位，不会划转到交易所。",
        },
        "policy": None if policy is None else _policy_dict(policy),
        "selection": selection,
        "test_simulation": None if test is None else _stats_public(test),
        "latest": latest,
        "bars": int(n),
        "first_open_ms": int(close.index[0].timestamp() * 1000),
        "last_open_ms": int(close.index[-1].timestamp() * 1000),
    }


def _policy_dict(policy: Policy) -> dict:
    return {
        "lookback": policy.lookback,
        "horizon": policy.horizon,
        "top_k": policy.top_k,
        "direction": policy.direction,
        "rebalance_every": policy.rebalance_every,
        "weighting": policy.weighting,
    }


def _stats_public(stats: dict) -> dict:
    return {
        "rebalances": stats["rebalances"],
        "skipped": stats["skipped"],
        "ending_equity": stats["ending_equity"],
        "fees": stats["fees"],
        "traded_notional": stats["traded_notional"],
        "turnover": stats["turnover"],
        "note": "纸面账本输出，只对本次输入有效，不是实盘业绩。换手 = 成交额 / 纸面权益。",
    }
