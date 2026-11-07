"""Bybit 线性合约横截面实盘执行。默认不开仓。

预算是独立的 100 USDT 虚拟额度，不是钱包里的全部 USDT。
交易所杠杆不参与放大。观察到的合约杠杆不是 1 时，只允许减自己的仓。
冻结宇宙只有 LINKUSDT、AVAXUSDT、SUIUSDT。其他机器人用过的标的一律拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import hashlib
import json
import math

from .bybit_public import DAY_MS
from .crypto_xs import CryptoXSConfig, Policy, causal_relative_return, ranked_symbols

ALLOCATION_USDT = 100.0
LOSS_CUTOFF_USDT = 20.0
MAX_UNHEDGED_USDT = 20.0
MAX_NEUTRAL_IMBALANCE_USDT = 10.0
ORDER_PREFIX = "cxls"
FROZEN_UNIVERSE = ("LINKUSDT", "AVAXUSDT", "SUIUSDT")
FORBIDDEN_UNDERLYINGS = frozenset({
    "ADA", "BTC", "DOGE", "ETH", "SOL", "XAUT", "XOM", "XRP",
})
QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "PERP")
OBSERVED_ACCOUNT = {
    "as_of": "2026-11-07T22:10:05+08:00",
    "uid": "211344229",
    "account": "mainnet master UTA2",
    "margin_mode": "ISOLATED_MARGIN",
    "wallet_usdt": "121.50290183",
    "free_usdt": "121.50290183",
    "exchange_leverage": "10",
    "positions": 0,
    "open_orders": 0,
}
READINESS_WARNINGS = (
    "UID 211344229 是主账户。2026-11-07 22:10 MYT 的空仓空挂单只是快照，不能证明以后没有别的机器人使用同一 UID。",
    "逐仓模式不会在机器人之间预留 100 USDT。没有划转权限，也不能把 100 USDT 挪到独立子账户。",
    "软件里的 100 USDT 只是虚拟预算。别的机器人仍可能占用同一 UID 的可用余额。",
    "三个候选合约当时的交易所杠杆是 10 倍。本程序不会修改杠杆。杠杆不是 1 时不开新仓。",
    "亏损 20 USDT 会撤销自己的委托并只平自己的数量，然后停机。这不是成交价意义上的亏损封顶。",
    "公开行情和账户只读事实来自批准的部署机，不是本构建环境。部署前需要再做一次只读核验。",
)


class AccountReadError(RuntimeError):
    """读账户失败。这种失败不能当成空仓。"""


class AmbiguousOrderError(RuntimeError):
    """请求可能已经到达交易所，不能换一个新的 orderLinkId 重试。"""


class DuplicateLinkError(RuntimeError):
    """同一个 orderLinkId 已经存在。"""


class OrderRejected(RuntimeError):
    """交易所明确拒绝，订单没有建立。"""


def is_forbidden_symbol(symbol: str) -> bool:
    """禁投的是底层资产，所以 USDT、USDC、USD 和 PERP 合约都算。"""
    name = symbol.upper()
    for underlying in FORBIDDEN_UNDERLYINGS:
        if any(name == underlying + suffix for suffix in QUOTE_SUFFIXES):
            return True
    return False


def floor_to_step(qty: float, step: float) -> float:
    amount = Decimal(str(qty))
    size = Decimal(str(step))
    if size <= 0:
        raise ValueError("qty step 必须为正")
    units = (amount / size).to_integral_value(rounding=ROUND_DOWN)
    return float(units * size)


def format_qty(qty: float, step: float) -> str:
    size = Decimal(str(step))
    amount = Decimal(str(qty)).quantize(size, rounding=ROUND_DOWN)
    return format(amount, "f")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    qty_step: float
    min_qty: float
    min_notional: float
    status: str = "Trading"
    contract_type: str = "LinearPerpetual"
    settle_coin: str = "USDT"

    def tradable(self) -> bool:
        return (
            self.status == "Trading"
            and self.contract_type == "LinearPerpetual"
            and self.settle_coin == "USDT"
            and self.symbol in FROZEN_UNIVERSE
        )


def parse_instrument(row: dict) -> Instrument:
    lot = row.get("lotSizeFilter") or {}
    raw_notional = lot.get("minNotionalValue", row.get("minNotionalValue", "0"))
    return Instrument(
        symbol=str(row.get("symbol") or ""),
        qty_step=float(lot.get("qtyStep") or row.get("qtyStep")),
        min_qty=float(lot.get("minOrderQty") or row.get("minOrderQty")),
        min_notional=float(raw_notional or 0),
        status=str(row.get("status") or "Trading"),
        contract_type=str(row.get("contractType") or "LinearPerpetual"),
        settle_coin=str(row.get("settleCoin") or "USDT"),
    )


@dataclass(frozen=True)
class StrategyFill:
    exec_id: str
    order_link_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float


@dataclass(frozen=True)
class FundingEvent:
    event_id: str
    symbol: str
    amount: float


@dataclass(frozen=True)
class ExchangeOrder:
    order_link_id: str
    symbol: str
    side: str
    qty: float
    leaves_qty: float
    reduce_only: bool
    status: str

    @property
    def is_open(self) -> bool:
        return self.status not in {"Filled", "Cancelled", "Rejected", "Deactivated"} and self.leaves_qty > 0


@dataclass(frozen=True)
class ExchangePosition:
    symbol: str
    qty: float
    mark: float | None
    position_idx: int = 0
    leverage: float | None = None


@dataclass
class ProtectionSnapshot:
    positions: list[ExchangePosition]
    orders: list[ExchangeOrder]
    fills: list[StrategyFill]
    funding: list[FundingEvent | None]
    wallet_usdt: float | None
    free_usdt: float | None
    borrow_usdt: float
    leverage_by_symbol: dict[str, float]
    position_mode_by_symbol: dict[str, int]
    instruments: dict[str, Instrument]
    marks: dict[str, float]
    pages_complete: bool
    categories_checked: tuple[str, ...]


@dataclass
class MarketView:
    last_open_ms: int
    stale: bool
    close: object = None
    turnover: object = None


def closed_daily_view(close, turnover, now_ms: int, max_stale_ms: int = 2 * DAY_MS) -> MarketView:
    """最后一根必须已经收盘。正在形成的 UTC 日线不能参加排名。"""
    last = int(close.index[-1].timestamp() * 1000)
    closed = last + DAY_MS <= int(now_ms)
    stale = (not closed) or (int(now_ms) - (last + DAY_MS) > max_stale_ms)
    return MarketView(last, stale, close, turnover)


@dataclass(frozen=True)
class LiveConfig:
    """实盘配置。金额和宇宙是冻结的，文件不能把它们改大。"""

    live_enabled: bool
    opt_in: bool
    accept_shared_master_uid: bool
    symbols: tuple[str, ...]
    lookback: int
    min_history: int
    min_median_turnover: float
    liquidity_window: int
    ledger_path: str
    forbidden_extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.symbols) != FROZEN_UNIVERSE:
            raise ValueError("实盘宇宙已冻结为 LINKUSDT、AVAXUSDT、SUIUSDT，不能增删")
        if any(is_forbidden_symbol(symbol) for symbol in self.symbols):
            raise ValueError("冻结宇宙和禁投标的重叠")
        if self.lookback < 1 or self.min_history < 2 or self.liquidity_window < 1:
            raise ValueError("回看、历史或流动性窗口不合法")
        if self.min_median_turnover < 0:
            raise ValueError("流动性下限不能为负")

    @property
    def policy(self) -> Policy:
        return Policy(self.lookback, 1, 1, "momentum", rebalance_every=1)

    @property
    def entries_opted_in(self) -> bool:
        return bool(self.live_enabled and self.opt_in and self.accept_shared_master_uid)


def load_live_config(path: str | Path, env: dict[str, str] | None = None) -> LiveConfig:
    env = env or {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for banned in ("api_key", "api_secret", "secret", "leverage"):
        if banned in raw:
            raise ValueError(f"配置里不能写 {banned}")
    if float(raw.get("allocation_usdt", ALLOCATION_USDT)) != ALLOCATION_USDT:
        raise ValueError("预算只能是 100 USDT")
    if float(raw.get("loss_cutoff_usdt", LOSS_CUTOFF_USDT)) != LOSS_CUTOFF_USDT:
        raise ValueError("亏损线只能是 20 USDT")
    if raw.get("category", "linear") != "linear":
        raise ValueError("实盘只做 linear")
    extra_file = raw.get("forbidden_extra_file")
    extra: list[str] = [str(item).upper() for item in raw.get("forbidden_extra") or []]
    if extra_file:
        file_raw = json.loads(Path(extra_file).read_text(encoding="utf-8"))
        extra.extend(str(item).upper() for item in file_raw.get("symbols") or [])
    symbols = tuple(str(item).upper() for item in raw.get("symbols") or FROZEN_UNIVERSE)
    return LiveConfig(
        live_enabled=bool(raw.get("live_enabled", False)),
        opt_in=env.get("SUPER_TRADER_LIVE") == "1",
        accept_shared_master_uid=bool(raw.get("accept_shared_master_uid", False)),
        symbols=symbols,
        lookback=int(raw.get("lookback", 20)),
        min_history=int(raw.get("min_history", 60)),
        min_median_turnover=float(raw.get("min_median_turnover_usdt", 1_000_000)),
        liquidity_window=int(raw.get("liquidity_window", 20)),
        ledger_path=str(raw.get("ledger_path", "data/crypto_xs_live/ledger.json")),
        forbidden_extra=tuple(extra),
    )


def readiness(cfg: LiveConfig, snapshot: ProtectionSnapshot | None = None) -> dict:
    hard = []
    if not cfg.live_enabled or not cfg.opt_in:
        hard.append("缺少显式实盘开关 SUPER_TRADER_LIVE=1 和 live_enabled")
    if not cfg.accept_shared_master_uid:
        hard.append("尚未确认共享主账户上的软件预算。确认也不等于物理隔离")
    if snapshot is None:
        hard.append("这次没有读取账户，不能开仓")
    else:
        if not snapshot.pages_complete:
            hard.append("持仓或委托分页没有查完")
        if snapshot.free_usdt is None:
            hard.append("逐仓可用余额无法按公式计算，不能把空字段当成 0")
        if not leverage_is_one(snapshot, cfg.symbols):
            hard.append("候选合约杠杆不是 1 倍。本程序不会去改它")
        if snapshot.borrow_usdt > 0:
            hard.append("账户有借币，不能拿来当策略本金")
    return {
        "entries_blocked": bool(hard) or not cfg.entries_opted_in,
        "hard_blockers": hard,
        "unresolved": list(READINESS_WARNINGS),
        "universe": list(FROZEN_UNIVERSE),
        "forbidden_underlyings": sorted(FORBIDDEN_UNDERLYINGS),
        "allocation_usdt": ALLOCATION_USDT,
        "loss_cutoff_usdt": LOSS_CUTOFF_USDT,
        "wallet_is_not_budget": True,
        "isolated_margin_reserves_budget": False,
        "observed_account": OBSERVED_ACCOUNT,
    }


def leverage_is_one(snapshot: ProtectionSnapshot, symbols: tuple[str, ...]) -> bool:
    for symbol in symbols:
        value = snapshot.leverage_by_symbol.get(symbol)
        mode = snapshot.position_mode_by_symbol.get(symbol)
        if value is None or abs(value - 1.0) > 1e-9:
            return False
        if mode not in (0, None):
            return False
    return True


def opening_budget(strategy_equity: float, free_usdt: float | None, exchange_leverage: float) -> float:
    """1 倍名义预算。交易所杠杆只用来决定是否允许开仓，绝不拿来乘数量。"""
    if abs(exchange_leverage - 1.0) > 1e-9 or free_usdt is None:
        return 0.0
    if strategy_equity <= 0 or free_usdt <= 0:
        return 0.0
    return min(ALLOCATION_USDT, strategy_equity, free_usdt)


@dataclass
class Ledger:
    path: Path
    halted: bool = False
    halt_reason: str | None = None
    realized: float = 0.0
    funding: float = 0.0
    qty: dict[str, float] = field(default_factory=dict)
    avg: dict[str, float] = field(default_factory=dict)
    seen_exec: set[str] = field(default_factory=set)
    seen_funding: set[str] = field(default_factory=set)
    intents: dict[str, dict] = field(default_factory=dict)
    tracked: set[str] = field(default_factory=set)
    last_signal_ms: int | None = None
    last_wallet: float | None = None
    halt_incomplete: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        file = Path(path)
        if not file.exists():
            return cls(path=file)
        raw = json.loads(file.read_text(encoding="utf-8"))
        return cls(
            path=file,
            halted=bool(raw.get("halted", False)),
            halt_reason=raw.get("halt_reason"),
            realized=float(raw.get("realized", 0)),
            funding=float(raw.get("funding", 0)),
            qty={key: float(value) for key, value in (raw.get("qty") or {}).items()},
            avg={key: float(value) for key, value in (raw.get("avg") or {}).items()},
            seen_exec=set(raw.get("seen_exec") or []),
            seen_funding=set(raw.get("seen_funding") or []),
            intents=dict(raw.get("intents") or {}),
            tracked=set(raw.get("tracked") or []),
            last_signal_ms=raw.get("last_signal_ms"),
            last_wallet=raw.get("last_wallet"),
            halt_incomplete=list(raw.get("halt_incomplete") or []),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "realized": self.realized,
            "funding": self.funding,
            "qty": self.qty,
            "avg": self.avg,
            "seen_exec": sorted(self.seen_exec),
            "seen_funding": sorted(self.seen_funding),
            "intents": self.intents,
            "tracked": sorted(self.tracked),
            "last_signal_ms": self.last_signal_ms,
            "last_wallet": self.last_wallet,
            "halt_incomplete": self.halt_incomplete,
            "allocation_usdt": ALLOCATION_USDT,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def has_exposure(self) -> bool:
        return any(abs(qty) > 1e-12 for qty in self.qty.values())

    def unrealized(self, marks: dict[str, float]) -> float | None:
        total = 0.0
        for symbol, qty in self.qty.items():
            if abs(qty) < 1e-12:
                continue
            mark = marks.get(symbol)
            if mark is None or not math.isfinite(mark):
                return None
            total += qty * (mark - self.avg.get(symbol, 0.0))
        return total

    def pnl(self, marks: dict[str, float]) -> float | None:
        unrealized = self.unrealized(marks)
        if unrealized is None:
            return None
        return self.realized + self.funding + unrealized

    def equity(self, marks: dict[str, float]) -> float | None:
        pnl = self.pnl(marks)
        if pnl is None:
            return None
        return ALLOCATION_USDT + pnl


def apply_fill(ledger: Ledger, fill: StrategyFill) -> None:
    if not fill.exec_id or fill.exec_id in ledger.seen_exec:
        return
    if not str(fill.order_link_id).startswith(ORDER_PREFIX):
        return
    if fill.order_link_id not in ledger.intents and fill.symbol not in ledger.tracked:
        return
    ledger.seen_exec.add(fill.exec_id)
    ledger.tracked.add(fill.symbol)
    qty = float(ledger.qty.get(fill.symbol, 0.0))
    avg = float(ledger.avg.get(fill.symbol, 0.0))
    delta = fill.qty if fill.side == "Buy" else -fill.qty
    realized = -float(fill.fee)
    if qty == 0 or qty * delta > 0:
        new_qty = qty + delta
        avg = (abs(qty) * avg + abs(delta) * fill.price) / abs(new_qty)
        qty = new_qty
    else:
        closing = min(abs(qty), abs(delta))
        if qty > 0:
            realized += (fill.price - avg) * closing
        else:
            realized += (avg - fill.price) * closing
        qty = qty + delta
        if qty == 0:
            avg = 0.0
        elif (qty - delta) * qty < 0:
            avg = fill.price
    ledger.qty[fill.symbol] = qty
    ledger.avg[fill.symbol] = avg
    ledger.realized += realized


def apply_funding(ledger: Ledger, event: FundingEvent | None) -> None:
    if event is None or not event.event_id or event.event_id in ledger.seen_funding:
        return
    if event.symbol not in ledger.tracked:
        return
    ledger.seen_funding.add(event.event_id)
    ledger.funding += float(event.amount)


def link_id(symbol: str, signal_ms: int, side: str, reduce_only: bool, qty: float) -> str:
    raw = f"{ORDER_PREFIX}|{symbol}|{int(signal_ms)}|{side}|{int(reduce_only)}|{qty:.8f}"
    return ORDER_PREFIX + hashlib.sha256(raw.encode()).hexdigest()[:32]


def ours(order_link_id: str, ledger: Ledger) -> bool:
    return order_link_id.startswith(ORDER_PREFIX) and order_link_id in ledger.intents


def conflict_symbols(snapshot: ProtectionSnapshot, ledger: Ledger) -> set[str]:
    blocked = set()
    for position in snapshot.positions:
        if abs(position.qty) < 1e-12:
            continue
        owned = float(ledger.qty.get(position.symbol, 0.0))
        step = 1e-8
        instrument = snapshot.instruments.get(position.symbol)
        if instrument is not None:
            step = max(instrument.qty_step, 1e-8)
        if abs(position.qty - owned) > step or position.position_idx != 0:
            blocked.add(position.symbol)
    for order in snapshot.orders:
        if not order.is_open:
            continue
        if not ours(order.order_link_id, ledger):
            blocked.add(order.symbol)
    return blocked


def has_foreign(snapshot: ProtectionSnapshot, ledger: Ledger) -> bool:
    for position in snapshot.positions:
        if abs(position.qty) < 1e-12:
            continue
        if position.symbol not in ledger.qty or abs(position.qty - ledger.qty.get(position.symbol, 0.0)) > 1e-8:
            return True
    for order in snapshot.orders:
        if order.is_open and not ours(order.order_link_id, ledger):
            return True
    return False


def plan_basket(ranked: list[str], *, budget: float, marks: dict[str, float],
                instruments: dict[str, Instrument], blocked: set[str]) -> list[dict] | None:
    """一多一空。凑不齐、超预算或偏差过大就整篮不下。"""
    if budget <= 0:
        return None
    side_budget = budget / 2.0
    usable = [
        symbol for symbol in ranked
        if symbol in FROZEN_UNIVERSE
        and symbol not in blocked
        and not is_forbidden_symbol(symbol)
        and symbol in marks
        and symbol in instruments
        and instruments[symbol].tradable()
    ]
    if len(usable) < 2:
        return None
    long_leg = _leg(usable[0], "Buy", side_budget, marks, instruments)
    short_symbol = next(symbol for symbol in reversed(usable) if symbol != usable[0])
    short_leg = _leg(short_symbol, "Sell", side_budget, marks, instruments)
    if long_leg is None or short_leg is None:
        return None
    gross = long_leg["notional"] + short_leg["notional"]
    imbalance = abs(long_leg["notional"] - short_leg["notional"])
    if gross > budget + 1e-6 or imbalance > MAX_NEUTRAL_IMBALANCE_USDT:
        return None
    return [long_leg, short_leg]


def _leg(symbol: str, side: str, budget: float, marks: dict[str, float],
         instruments: dict[str, Instrument]) -> dict | None:
    instrument = instruments[symbol]
    mark = marks[symbol]
    if mark <= 0 or budget <= 0:
        return None
    qty = floor_to_step(budget / mark, instrument.qty_step)
    if qty < instrument.min_qty:
        return None
    notional = qty * mark
    if notional < instrument.min_notional or notional > budget + 1e-6:
        return None
    return {"symbol": symbol, "side": side, "qty": qty, "notional": notional, "reduce_only": False}


def rank_symbols(close, turnover, cfg: LiveConfig) -> list[str]:
    paper = CryptoXSConfig(
        category="linear",
        symbols=cfg.symbols,
        min_history=cfg.min_history,
        min_names=len(FROZEN_UNIVERSE),
        min_median_turnover=cfg.min_median_turnover,
        liquidity_window=cfg.liquidity_window,
        paper_equity_usdt=ALLOCATION_USDT,
    )
    frame = close[list(FROZEN_UNIVERSE)]
    scored = causal_relative_return(frame, turnover[list(FROZEN_UNIVERSE)], cfg.lookback, paper)
    return [symbol for symbol in ranked_symbols(scored.iloc[-1]) if symbol in FROZEN_UNIVERSE]


@dataclass(frozen=True)
class NewOrder:
    symbol: str
    side: str
    qty: float
    reduce_only: bool
    signal_ms: int

    @property
    def link_id(self) -> str:
        return link_id_for(self)


def link_id_for(order: NewOrder) -> str:
    return link_id(order.symbol, order.signal_ms, order.side, order.reduce_only, order.qty)


def submit_order(ledger: Ledger, broker, order: NewOrder) -> str:
    identifier = order.link_id
    current = ledger.intents.get(identifier)
    if current and current.get("status") in {"submitted", "filled"}:
        return "exists"
    ledger.intents[identifier] = {
        "symbol": order.symbol, "side": order.side, "qty": order.qty,
        "reduce_only": order.reduce_only, "signal_ms": order.signal_ms, "status": "staged",
    }
    ledger.tracked.add(order.symbol)
    ledger.save()
    try:
        broker.create(order)
    except AmbiguousOrderError:
        ledger.intents[identifier]["status"] = "ambiguous"
        ledger.save()
        return "ambiguous"
    except DuplicateLinkError:
        ledger.intents[identifier]["status"] = "submitted"
        ledger.save()
        return "duplicate"
    except OrderRejected:
        ledger.intents[identifier]["status"] = "rejected"
        ledger.save()
        return "rejected"
    ledger.intents[identifier]["status"] = "submitted"
    ledger.save()
    return "submitted"


def retry_ambiguous(ledger: Ledger, broker) -> None:
    for identifier, intent in list(ledger.intents.items()):
        if intent.get("status") != "ambiguous":
            continue
        if broker.find_order(identifier) is not None:
            intent["status"] = "submitted"
            continue
        order = NewOrder(
            intent["symbol"], intent["side"], float(intent["qty"]),
            bool(intent["reduce_only"]), int(intent["signal_ms"]),
        )
        submit_order(ledger, broker, order)


def _cancel_ours(ledger: Ledger, snapshot: ProtectionSnapshot, broker, result: dict) -> None:
    for order in snapshot.orders:
        if order.is_open and ours(order.order_link_id, ledger):
            broker.cancel(order.order_link_id, order.symbol)
            result["cancels"].append(order.order_link_id)


def _flatten_ours(ledger: Ledger, snapshot: ProtectionSnapshot, broker, result: dict,
                  blocked: set[str], signal_ms: int) -> None:
    incomplete = []
    for symbol, qty in list(ledger.qty.items()):
        if abs(qty) < 1e-12:
            continue
        if symbol in blocked or is_forbidden_symbol(symbol) or symbol not in FROZEN_UNIVERSE:
            incomplete.append(symbol)
            continue
        instrument = snapshot.instruments.get(symbol)
        if instrument is None:
            incomplete.append(symbol)
            continue
        closing = floor_to_step(abs(qty), instrument.qty_step)
        if closing < instrument.min_qty or closing * (snapshot.marks.get(symbol) or 0) < instrument.min_notional:
            incomplete.append(symbol)
            continue
        order = NewOrder(symbol, "Sell" if qty > 0 else "Buy", closing, True, signal_ms)
        result["orders"].append(order)
        submit_order(ledger, broker, order)
    ledger.halt_incomplete = incomplete


def run_cycle(cfg: LiveConfig, ledger: Ledger, broker, market, now_ms: int) -> dict:
    result = {"action": "blocked", "orders": [], "cancels": [], "blockers": [], "pnl": None, "budget": 0.0}
    report = readiness(cfg)
    if not cfg.entries_opted_in:
        result["blockers"] = report["hard_blockers"]
        ledger.save()
        return result
    try:
        snapshot = broker.fetch_protection()
    except AccountReadError as exc:
        result["blockers"].append(f"account read failed: {exc}")
        ledger.save()
        return result
    if not snapshot.pages_complete:
        result["blockers"].append("position/order pagination incomplete")
        ledger.save()
        return result
    for fill in snapshot.fills:
        apply_fill(ledger, fill)
    for event in snapshot.funding:
        apply_funding(ledger, event)
    if snapshot.wallet_usdt is not None:
        ledger.last_wallet = snapshot.wallet_usdt
    blocked = conflict_symbols(snapshot, ledger)
    foreign = has_foreign(snapshot, ledger)
    pnl = ledger.pnl(snapshot.marks)
    result["pnl"] = pnl
    signal_ms = int(getattr(market, "last_open_ms", now_ms) or now_ms)
    if pnl is not None and pnl <= -LOSS_CUTOFF_USDT:
        ledger.halted = True
        ledger.halt_reason = "loss_cutoff"
        ledger.save()
        _cancel_ours(ledger, snapshot, broker, result)
        _flatten_ours(ledger, snapshot, broker, result, blocked, signal_ms)
        result["action"] = "halted"
        ledger.save()
        return result
    if ledger.halted:
        _cancel_ours(ledger, snapshot, broker, result)
        _flatten_ours(ledger, snapshot, broker, result, blocked, signal_ms)
        result["action"] = "halted"
        ledger.save()
        return result
    retry_ambiguous(ledger, broker)
    if any(item.get("status") == "ambiguous" for item in ledger.intents.values()):
        result["action"] = "ambiguous"
        result["blockers"].append("有一笔结果不确定的委托，暂停新开仓")
        ledger.save()
        return result
    _reduce_if_needed(cfg, ledger, snapshot, broker, result, blocked, signal_ms, foreign)
    blockers = []
    if not leverage_is_one(snapshot, cfg.symbols):
        blockers.append("symbol leverage is not 1x")
    if foreign:
        blockers.append("foreign position or order on the shared account")
    if snapshot.free_usdt is None:
        blockers.append("free collateral unknown")
    if snapshot.borrow_usdt > 0:
        blockers.append("borrowed balance is not strategy capital")
    if getattr(market, "stale", True):
        blockers.append("daily bar is forming or stale")
    if pnl is None and ledger.has_exposure():
        blockers.append("mark for owned quantity is missing")
    for symbol, mode in snapshot.position_mode_by_symbol.items():
        if symbol in cfg.symbols and mode not in (0, None):
            blockers.append(f"{symbol} is not one-way mode")
    if blockers:
        result["blockers"].extend(blockers)
        result["action"] = "reduced" if result["orders"] or result["cancels"] else "monitored"
        ledger.save()
        return result
    equity = ledger.equity(snapshot.marks)
    observed = snapshot.leverage_by_symbol.get(cfg.symbols[0], 1.0)
    result["budget"] = opening_budget(equity or 0.0, snapshot.free_usdt, observed)
    if market.close is None or not _rebalance_due(ledger, market.last_open_ms, cfg.policy.rebalance_every):
        result["action"] = "monitored"
        ledger.save()
        return result
    ranked = rank_symbols(market.close, market.turnover, cfg)
    basket = plan_basket(
        ranked, budget=result["budget"], marks=snapshot.marks,
        instruments=snapshot.instruments, blocked=blocked,
    )
    if not basket:
        result["action"] = "no_trade"
        ledger.last_signal_ms = market.last_open_ms
        ledger.save()
        return result
    if ledger.has_exposure():
        _flatten_ours(ledger, snapshot, broker, result, blocked, market.last_open_ms)
        result["action"] = "reducing_before_new_basket"
        ledger.save()
        return result
    for leg in basket:
        order = NewOrder(leg["symbol"], leg["side"], leg["qty"], False, market.last_open_ms)
        result["orders"].append(order)
        if submit_order(ledger, broker, order) == "ambiguous":
            result["action"] = "ambiguous"
            ledger.last_signal_ms = market.last_open_ms
            ledger.save()
            return result
    ledger.last_signal_ms = market.last_open_ms
    result["action"] = "rebalanced"
    ledger.save()
    return result


def _rebalance_due(ledger: Ledger, last_open_ms: int, every_days: int) -> bool:
    if ledger.last_signal_ms is None:
        return True
    return int(last_open_ms) >= int(ledger.last_signal_ms) + every_days * DAY_MS


def _reduce_if_needed(cfg, ledger, snapshot, broker, result, blocked, signal_ms, foreign) -> None:
    marks = snapshot.marks
    equity = ledger.equity(marks)
    if equity is None:
        return
    cap = min(ALLOCATION_USDT, max(0.0, equity))
    owned_gross = 0.0
    long_notional = 0.0
    short_notional = 0.0
    for symbol, qty in ledger.qty.items():
        mark = marks.get(symbol)
        if mark is None or abs(qty) < 1e-12:
            continue
        notional = abs(qty) * mark
        owned_gross += notional
        if qty > 0:
            long_notional += notional
        else:
            short_notional += notional
    pending = 0.0
    for order in snapshot.orders:
        if order.is_open and ours(order.order_link_id, ledger) and not order.reduce_only:
            mark = marks.get(order.symbol)
            if mark is not None:
                pending += abs(order.leaves_qty) * mark
    if owned_gross + pending > cap + 1e-6:
        _cancel_ours(ledger, snapshot, broker, result)
    gap = abs(long_notional - short_notional)
    if gap <= MAX_UNHEDGED_USDT and owned_gross <= cap + 1e-6:
        return
    heavier_positive = long_notional >= short_notional
    for symbol, qty in list(ledger.qty.items()):
        if abs(qty) < 1e-12 or symbol in blocked or (qty > 0) != heavier_positive:
            continue
        if foreign and symbol in blocked:
            continue
        instrument = snapshot.instruments.get(symbol)
        mark = marks.get(symbol)
        if instrument is None or mark is None:
            continue
        closing = floor_to_step(abs(qty), instrument.qty_step)
        if closing < instrument.min_qty:
            continue
        order = NewOrder(symbol, "Sell" if qty > 0 else "Buy", closing, True, signal_ms)
        result["orders"].append(order)
        submit_order(ledger, broker, order)
        break


def snapshot_instruments() -> dict[str, Instrument]:
    """2026-11-07 22:10 MYT 只读核验里这三个合约的过滤器。运行时仍要重新读取。"""
    return {
        "LINKUSDT": Instrument("LINKUSDT", 0.1, 0.1, 5.0),
        "AVAXUSDT": Instrument("AVAXUSDT", 0.1, 0.1, 5.0),
        "SUIUSDT": Instrument("SUIUSDT", 10.0, 10.0, 5.0),
    }


def snapshot_marks() -> dict[str, float]:
    return {"LINKUSDT": 10.894, "AVAXUSDT": 8.271, "SUIUSDT": 0.8624}
