"""持续监督：UTC 日线收盘后才允许再平衡，两次之间只盯自己的亏损。

这里不放宽开仓条件。没有 live_enabled、SUPER_TRADER_LIVE=1 和
accept_shared_master_uid 时直接退出。杠杆不是 1、有别人的仓或单、
或账户读失败时，都不发新的开仓单。
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .crypto_live import Ledger, MarketView, run_cycle

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
