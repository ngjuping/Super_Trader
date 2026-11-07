"""监督循环：不开新仓的等待也要能停在进程里，但开关和风控仍在。"""
from __future__ import annotations

import json
from pathlib import Path

from super_trader.cli import main
from super_trader.crypto_live import ExchangeOrder, Ledger, MarketView
from super_trader.crypto_supervisor import LedgerLock, lock_path, run_supervisor, status_path
from tests.test_crypto_live import FakeBroker, _config, _market, _snapshot


def _run(cfg, broker, market, loops=1, max_failures=3):
    return run_supervisor(
        cfg, broker, lambda: market, sleep_fn=lambda _seconds: True,
        now_fn=lambda: 1_700_000_000_000, max_loops=loops, interval_s=60,
        max_failures=max_failures,
    )


def test_supervisor_refuses_without_opt_in(tmp_path):
    cfg = _config(tmp_path, live_enabled=False, accept_shared_master_uid=False, opt_in=False)

    class Boom:
        def fetch_protection(self):
            raise AssertionError("不应连接账户")

    assert _run(cfg, Boom(), MarketView(0, True)) == 2


def test_ten_x_supervision_stays_up_without_opening(tmp_path):
    cfg = _config(tmp_path)
    broker = FakeBroker(_snapshot(leverage_by_symbol={
        "LINKUSDT": 10.0, "AVAXUSDT": 10.0, "SUIUSDT": 10.0,
    }))
    code = _run(cfg, broker, _market(), loops=2)
    assert code == 0 and broker.created == []
    status = json.loads(status_path(cfg).read_text(encoding="utf-8"))
    assert status["exclusive_account"] is False
    assert status["entries_blocked"] is True
    assert any("leverage" in item for item in status["blockers"])


def test_same_closed_bar_is_not_submitted_twice(tmp_path):
    cfg = _config(tmp_path)
    market = _market()
    last = {symbol: float(market.close[symbol].iloc[-1]) for symbol in market.close.columns}
    broker = FakeBroker(_snapshot(marks=last))
    assert _run(cfg, broker, market, loops=2) == 0
    assert len(broker.created) == 2
    assert {order.side for order in broker.created} == {"Buy", "Sell"}
    assert all(order.reduce_only is False for order in broker.created)
    gross = sum(order.qty * last[order.symbol] for order in broker.created)
    assert gross <= 100


def test_stale_bar_does_not_force_a_signal(tmp_path):
    cfg = _config(tmp_path)
    market = _market(now_offset=60_000)
    assert market.stale
    broker = FakeBroker(_snapshot())
    assert _run(cfg, broker, market) == 0
    assert broker.created == []


def test_second_process_cannot_take_the_ledger_lock(tmp_path):
    cfg = _config(tmp_path)
    held = LedgerLock(lock_path(cfg))
    held.acquire()
    try:
        broker = FakeBroker(_snapshot())
        assert _run(cfg, broker, _market()) == 2
        assert broker.created == []
    finally:
        held.release()


def test_account_failure_stops_closed_and_does_not_open(tmp_path):
    cfg = _config(tmp_path)
    broker = FakeBroker(_snapshot())
    broker.fail = True
    assert _run(cfg, broker, _market(), loops=3) == 1
    assert broker.created == []


def test_loss_cutoff_persists_across_loops(tmp_path):
    cfg = _config(tmp_path)
    ledger = Ledger(Path(cfg.ledger_path))
    ledger.realized = -21
    ledger.qty["LINKUSDT"] = 1.0
    ledger.avg["LINKUSDT"] = 10.0
    ledger.tracked.add("LINKUSDT")
    ledger.save()
    broker = FakeBroker(_snapshot(marks={"LINKUSDT": 10.0, "AVAXUSDT": 8.0, "SUIUSDT": 0.9}))
    assert _run(cfg, broker, _market(), loops=2) == 0
    assert broker.created
    assert all(order.reduce_only for order in broker.created)
    assert Ledger.load(cfg.ledger_path).halted


def test_foreign_order_blocks_the_supervisor_basket(tmp_path):
    cfg = _config(tmp_path)
    foreign = ExchangeOrder("xaut-old", "XAUTUSDT", "Buy", 0.01, 0.01, False, "New")
    broker = FakeBroker(_snapshot(orders=[foreign]))
    assert _run(cfg, broker, _market()) == 0
    assert broker.created == [] and broker.cancelled == []


def test_cycle_exception_fail_closed(tmp_path):
    cfg = _config(tmp_path)

    class Broken:
        def fetch_protection(self):
            raise RuntimeError("socket down")

        def create(self, _order):
            raise AssertionError("fail closed 后不能下单")

        def cancel(self, *_args):
            raise AssertionError("fail closed 后不能撤单")

        def find_order(self, _link):
            return None

    assert _run(cfg, Broken(), _market(), loops=1, max_failures=1) == 1


def test_cli_supervise_without_opt_in_does_not_touch_network(monkeypatch):
    monkeypatch.setattr(
        "super_trader.bybit_private.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")),
    )
    code = main([
        "crypto-xs-live", "--supervise", "--config", "config/crypto_xs_live.example.json",
    ])
    assert code == 2
