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


def test_unit_interpreter_is_the_python3_venv():
    unit = Path("deploy/crypto-xs-supervisor.service").read_text(encoding="utf-8")
    script = Path("deploy/install-supervisor-venv.sh").read_text(encoding="utf-8")
    assert "/usr/bin/python" not in unit
    assert "ExecStart=/opt/super-trader/.venv/bin/python -m super_trader crypto-xs-live" in unit
    assert 'ROOT="${1:-/opt/super-trader}"' in script
    assert "python3 -m venv" in script
    assert "$ROOT/.venv/bin/python" in script
    assert "SUPER_TRADER_OFFLINE_FIXTURE" not in unit


def test_process_sigterm_releases_lock_and_restart_does_not_duplicate(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    config = {
        "live_enabled": True, "category": "linear", "allocation_usdt": 100,
        "loss_cutoff_usdt": 20, "accept_shared_master_uid": True,
        "symbols": ["LINKUSDT", "AVAXUSDT", "SUIUSDT"], "lookback": 5,
        "min_history": 5, "min_median_turnover_usdt": 0, "liquidity_window": 3,
        "ledger_path": str(tmp_path / "ledger.json"),
    }
    config_path = tmp_path / "live.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    order_log = tmp_path / "orders.json"
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"order_log": str(order_log), "leverage": 1}), encoding="utf-8")
    env = os.environ.copy()
    env["SUPER_TRADER_LIVE"] = "1"
    env["SUPER_TRADER_OFFLINE_FIXTURE"] = str(fixture)
    for name in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_API_KEY_FILE", "BYBIT_API_SECRET_FILE"):
        env.pop(name, None)
    cmd = [
        sys.executable, "-m", "super_trader", "crypto-xs-live", "--supervise",
        "--config", str(config_path), "--interval", "30",
    ]

    def start():
        return subprocess.Popen(
            cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def wait_status(proc):
        status = tmp_path / "status.json"
        deadline = time.time() + 20
        while time.time() < deadline and not status.exists():
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise AssertionError(f"child exited {proc.returncode}\n{out}\n{err}")
            time.sleep(0.05)
        assert status.exists()

    def stop(proc):
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=10)
        if code != 0:
            err = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"SIGTERM exit {code}\n{err}")
        return code

    first = start()
    try:
        wait_status(first)
        rival = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env,
                               capture_output=True, text=True, timeout=20)
        assert rival.returncode == 2
        assert stop(first) == 0
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=5)

    probe = LedgerLock(tmp_path / "ledger.lock")
    probe.acquire()
    probe.release()
    created = json.loads(order_log.read_text(encoding="utf-8"))
    assert len(created) == 2
    assert {row["side"] for row in created} == {"Buy", "Sell"}
    (tmp_path / "status.json").unlink()

    second = start()
    try:
        wait_status(second)
        assert stop(second) == 0
    finally:
        if second.poll() is None:
            second.kill()
            second.wait(timeout=5)
    assert json.loads(order_log.read_text(encoding="utf-8")) == created


def test_cli_supervise_without_opt_in_does_not_touch_network(monkeypatch):
    monkeypatch.setattr(
        "super_trader.bybit_private.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")),
    )
    code = main([
        "crypto-xs-live", "--supervise", "--config", "config/crypto_xs_live.example.json",
    ])
    assert code == 2
