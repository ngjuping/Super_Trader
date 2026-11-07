"""持续监督：UTC 日线收盘后才允许再平衡，两次之间只盯自己的亏损。

这里不放宽开仓条件。没有 live_enabled、SUPER_TRADER_LIVE=1 和
accept_shared_master_uid 时直接退出。杠杆不是正好 2、有别人的仓或单、
或账户读失败时，都不发新的开仓单。
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .bybit_public import DAY_MS
from .crypto_live import (
    FROZEN_UNIVERSE, Ledger, MarketView, ProtectionSnapshot, closed_daily_view,
    run_cycle, snapshot_instruments,
)

DEFAULT_INTERVAL_S = 60
DEFAULT_MAX_FAILURES = 3


class LedgerBusy(RuntimeError):
    """账本锁已被另一个进程持有。"""


class LedgerLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise LedgerBusy("另一个 Super_Trader 进程正持有账本锁") from exc

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def status_path(cfg) -> Path:
    return Path(cfg.ledger_path).with_name("status.json")


def lock_path(cfg) -> Path:
    return Path(cfg.ledger_path).with_name("ledger.lock")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _failed(result: dict) -> bool:
    if result.get("action") != "blocked":
        return False
    text = " ".join(result.get("blockers") or [])
    return "account read failed" in text or "pagination incomplete" in text


def offline_fixture(path: str | Path):
    """测试用离线账户。不读密钥，不访问交易所。正式 unit 不得设置这个入口。"""
    import pandas as pd

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("network") or raw.get("api_key") or raw.get("api_secret"):
        raise ValueError("离线夹具不能携带密钥或网络开关")
    index = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    steps = list(range(12))
    close = pd.DataFrame({
        "AVAXUSDT": [8 * (1.01 ** i) for i in steps],
        "SUIUSDT": [0.85] * 12,
        "BCHUSDT": [250 * (0.99 ** i) for i in steps],
    }, index=index)
    turnover = pd.DataFrame(5_000_000.0, index=index, columns=list(FROZEN_UNIVERSE))
    now_ms = int(index[-1].timestamp() * 1000) + DAY_MS + 60_000
    market = closed_daily_view(close, turnover, now_ms)
    marks = {symbol: float(close[symbol].iloc[-1]) for symbol in FROZEN_UNIVERSE}
    snapshot = ProtectionSnapshot(
        positions=[], orders=[], fills=[], funding=[],
        wallet_usdt=float(raw.get("wallet_usdt", 121.50290183)),
        free_usdt=float(raw.get("free_usdt", 121.50290183)),
        borrow_usdt=0.0,
        leverage_by_symbol={symbol: float(raw.get("leverage", 2)) for symbol in FROZEN_UNIVERSE},
        position_mode_by_symbol={symbol: 0 for symbol in FROZEN_UNIVERSE},
        instruments=snapshot_instruments(),
        marks=marks,
        pages_complete=True,
        categories_checked=("linear:USDT", "linear:USDC", "inverse", "option", "spot"),
    )
    order_log = Path(raw["order_log"])

    class OfflineBroker:
        def fetch_protection(self):
            return snapshot

        def create(self, order):
            rows = []
            if order_log.exists():
                rows = json.loads(order_log.read_text(encoding="utf-8"))
            rows.append({
                "link_id": order.link_id, "symbol": order.symbol, "side": order.side,
                "qty": order.qty, "reduce_only": order.reduce_only,
            })
            order_log.write_text(json.dumps(rows), encoding="utf-8")

        def cancel(self, _link_id, _symbol):
            return None

        def find_order(self, _link_id):
            return None

    return OfflineBroker(), (lambda: market)


def run_supervisor(cfg, broker, market_fn, *, sleep_fn, now_fn, stop_fn=None,
                   interval_s: int = DEFAULT_INTERVAL_S, max_failures: int = DEFAULT_MAX_FAILURES,
                   max_loops: int | None = None) -> int:
    """返回 0 表示正常停下，1 表示连续失败后停机，2 表示开仓开关或账本锁不满足。"""
    if interval_s < 1 or max_failures < 1:
        raise ValueError("监督间隔和失败次数必须为正")
    if not cfg.entries_opted_in:
        return 2
    lock = LedgerLock(lock_path(cfg))
    try:
        lock.acquire()
    except LedgerBusy:
        return 2
    ledger = Ledger.load(cfg.ledger_path)
    failures = 0
    loops = 0
    try:
        while max_loops is None or loops < max_loops:
            if stop_fn and stop_fn():
                return 0
            loops += 1
            now_ms = int(now_fn())
            try:
                market = market_fn()
            except Exception:
                market = MarketView(0, True)
            try:
                result = run_cycle(cfg, ledger, broker, market, now_ms)
            except Exception as exc:
                failures += 1
                write_status(status_path(cfg), {
                    "mode": "supervise",
                    "exclusive_account": False,
                    "entries_blocked": True,
                    "action": "fail_closed",
                    "blockers": [type(exc).__name__],
                    "orders": 0,
                    "halted": ledger.halted,
                    "as_of_ms": now_ms,
                    "clock": "UTC",
                })
                if failures >= max_failures:
                    return 1
            else:
                if _failed(result):
                    failures += 1
                else:
                    failures = 0
                blockers = list(result.get("blockers") or [])
                write_status(status_path(cfg), {
                    "mode": "supervise",
                    "exclusive_account": False,
                    "entries_blocked": bool(blockers) or result.get("action") in {"blocked", "halted", "ambiguous"},
                    "action": result.get("action"),
                    "blockers": blockers,
                    "orders": len(result.get("orders") or []),
                    "budget": result.get("budget"),
                    "pnl": result.get("pnl"),
                    "halted": ledger.halted,
                    "as_of_ms": now_ms,
                    "clock": "UTC",
                })
                if failures >= max_failures:
                    return 1
            if max_loops is not None and loops >= max_loops:
                return 0
            if not sleep_fn(interval_s):
                return 0
        return 0
    finally:
        lock.release()
