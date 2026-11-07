"""实盘横截面的账本、预算和共享账户保护。不连接 Bybit。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from super_trader.bybit_public import DAY_MS
from super_trader.cli import main
from super_trader.crypto_live import (
    ALLOCATION_USDT, FROZEN_UNIVERSE, LOSS_CUTOFF_USDT, AccountReadError,
    AmbiguousOrderError, ExchangeOrder, ExchangePosition, FundingEvent, Ledger,
    LiveConfig, ProtectionSnapshot, StrategyFill, apply_fill, apply_funding,
    closed_daily_view, is_forbidden_symbol, load_live_config, opening_budget,
    plan_basket, readiness, run_cycle, snapshot_instruments, snapshot_marks,
)


class FakeBroker:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.attempts = []
        self.created = []
        self.cancelled = []
        self.stored = {}
        self.fail = False
        self.ambiguous = False
        self.keep_ambiguous_order = True

    def fetch_protection(self):
        if self.fail:
            raise AccountReadError("down")
        return self.snapshot

    def create(self, order):
        self.attempts.append(order.link_id)
        if self.ambiguous:
            self.ambiguous = False
            if self.keep_ambiguous_order:
                self.stored[order.link_id] = order
            raise AmbiguousOrderError("timeout")
        self.created.append(order)
        self.stored[order.link_id] = order

    def cancel(self, link_id, symbol):
        self.cancelled.append((link_id, symbol))
        self.stored.pop(link_id, None)

    def find_order(self, link_id):
        return self.stored.get(link_id)


def _config(tmp_path: Path, **overrides) -> LiveConfig:
    values = dict(
        live_enabled=True, opt_in=True, accept_shared_master_uid=True,
        symbols=FROZEN_UNIVERSE, lookback=5, min_history=5,
        min_median_turnover=0.0, liquidity_window=3,
        ledger_path=str(tmp_path / "ledger.json"),
    )
    values.update(overrides)
    return LiveConfig(**values)


def _snapshot(**overrides) -> ProtectionSnapshot:
    marks = dict(snapshot_marks())
    marks.update(overrides.pop("marks", {}))
    values = dict(
        positions=[], orders=[], fills=[], funding=[],
        wallet_usdt=121.50290183, free_usdt=121.50290183, borrow_usdt=0.0,
        leverage_by_symbol={symbol: 1.0 for symbol in FROZEN_UNIVERSE},
        position_mode_by_symbol={symbol: 0 for symbol in FROZEN_UNIVERSE},
        instruments=snapshot_instruments(), marks=marks, pages_complete=True,
        categories_checked=("linear:USDT", "linear:USDC", "inverse", "option", "spot"),
    )
    values.update(overrides)
    return ProtectionSnapshot(**values)


def _market(n=12, now_offset=DAY_MS + 60_000):
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    steps = np.arange(n)
    close = pd.DataFrame({
        "LINKUSDT": 10 * (1.01 ** steps),
        "AVAXUSDT": np.full(n, 8.0),
        "SUIUSDT": 1 * (0.99 ** steps),
    }, index=index)
    turnover = pd.DataFrame(5_000_000.0, index=index, columns=close.columns)
    now = int(index[-1].timestamp() * 1000) + now_offset
    return closed_daily_view(close, turnover, now)


def test_forbidden_underlyings_cover_every_quoted_contract():
    for underlying in ("ADA", "BTC", "DOGE", "ETH", "SOL", "XAUT", "XOM", "XRP"):
        for suffix in ("USDT", "USDC", "USD", "PERP"):
            assert is_forbidden_symbol(underlying + suffix)
    for symbol in FROZEN_UNIVERSE:
        assert not is_forbidden_symbol(symbol)
    with pytest.raises(ValueError):
        LiveConfig(True, True, True, ("BTCUSDT", "AVAXUSDT", "SUIUSDT"), 5, 5, 0, 3, "x")


def test_budget_ignores_wallet_and_exchange_leverage():
    assert opening_budget(100, 121.50290183, 1) == 100
    assert opening_budget(150, 500, 1) == 100
    assert opening_budget(90, 121.5, 1) == 90
    assert opening_budget(100, 40, 1) == 40
    assert opening_budget(100, 121.5, 10) == 0
    assert opening_budget(100, None, 1) == 0


def test_snapshot_pair_fits_under_100_and_respects_steps():
    basket = plan_basket(
        ["LINKUSDT", "AVAXUSDT", "SUIUSDT"], budget=100,
        marks=snapshot_marks(), instruments=snapshot_instruments(), blocked=set(),
    )
    assert basket is not None
    assert [leg["side"] for leg in basket] == ["Buy", "Sell"]
    assert {leg["symbol"] for leg in basket} <= set(FROZEN_UNIVERSE)
    gross = sum(leg["notional"] for leg in basket)
    assert gross <= 100
    assert all(leg["notional"] >= 5 for leg in basket)
    link = basket[0]
    sui = basket[1]
    assert link["symbol"] == "LINKUSDT" and link["qty"] == pytest.approx(4.5)
    assert sui["symbol"] == "SUIUSDT" and sui["qty"] == pytest.approx(50)
    tiny = plan_basket(
        ["LINKUSDT", "AVAXUSDT", "SUIUSDT"], budget=9,
        marks=snapshot_marks(), instruments=snapshot_instruments(), blocked=set(),
    )
    assert tiny is None


def test_ten_x_leverage_and_unacknowledged_share_send_nothing(tmp_path):
    broker = FakeBroker(_snapshot(leverage_by_symbol={symbol: 10.0 for symbol in FROZEN_UNIVERSE}))
    result = run_cycle(_config(tmp_path), Ledger(tmp_path / "ledger.json"), broker, _market(), 0)
    assert result["orders"] == [] and broker.attempts == []
    assert any("leverage" in item for item in result["blockers"])

    boom = FakeBroker(_snapshot())

    def refuse():
        raise AssertionError("shared-account acknowledgement is still required")

    boom.fetch_protection = refuse
    blocked = run_cycle(
        _config(tmp_path, accept_shared_master_uid=False),
        Ledger(tmp_path / "other.json"), boom, _market(), 0,
    )
    assert blocked["action"] == "blocked" and blocked["orders"] == []


def test_rebalance_is_neutral_long_short_inside_virtual_budget(tmp_path):
    market = _market()
    last = {symbol: float(market.close[symbol].iloc[-1]) for symbol in FROZEN_UNIVERSE}
    broker = FakeBroker(_snapshot(marks=last, wallet_usdt=121.50290183, free_usdt=121.50290183))
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.realized = 50
    result = run_cycle(_config(tmp_path), ledger, broker, market, market.last_open_ms + DAY_MS)
    assert result["action"] == "rebalanced"
    assert result["budget"] == 100
    assert len(broker.created) == 2
    assert {order.side for order in broker.created} == {"Buy", "Sell"}
    assert all(order.reduce_only is False for order in broker.created)
    assert all(order.symbol in FROZEN_UNIVERSE for order in broker.created)
    gross = sum(order.qty * last[order.symbol] for order in broker.created)
    assert gross <= 100
    assert ledger.realized == 50


def test_foreign_order_blocks_new_risk_and_is_not_cancelled(tmp_path):
    foreign = ExchangeOrder("other-bot", "BTCUSDT", "Buy", 0.01, 0.01, False, "New")
    broker = FakeBroker(_snapshot(orders=[foreign]))
    result = run_cycle(_config(tmp_path), Ledger(tmp_path / "ledger.json"), broker, _market(), 0)
    assert broker.created == [] and broker.cancelled == []
    assert any("foreign" in item for item in result["blockers"])


def test_loss_cutoff_persists_and_closes_only_owned_quantity(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path)
    ledger.realized = -21
    ledger.qty["LINKUSDT"] = 1.0
    ledger.avg["LINKUSDT"] = 10.0
    ledger.tracked.add("LINKUSDT")
    broker = FakeBroker(_snapshot(marks={"LINKUSDT": 10.0, "AVAXUSDT": 8.0, "SUIUSDT": 0.9}))
    result = run_cycle(_config(tmp_path), ledger, broker, _market(), 0)
    assert result["action"] == "halted"
    assert broker.created and broker.created[0].reduce_only and broker.created[0].side == "Sell"
    assert broker.created[0].qty == pytest.approx(1.0)
    assert Ledger.load(path).halted
    again = FakeBroker(_snapshot(marks={"LINKUSDT": 10.0, "AVAXUSDT": 8.0, "SUIUSDT": 0.9}))
    reloaded = Ledger.load(path)
    reloaded.qty["LINKUSDT"] = 1.0
    second = run_cycle(_config(tmp_path), reloaded, again, _market(), 0)
    assert second["action"] == "halted"
    assert all(order.reduce_only for order in again.created)


def test_mixed_foreign_position_is_not_flattened(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.realized = -25
    ledger.qty["LINKUSDT"] = 1.0
    ledger.avg["LINKUSDT"] = 10.0
    ledger.tracked.add("LINKUSDT")
    position = ExchangePosition("LINKUSDT", 5.0, 10.0, 0, 1.0)
    broker = FakeBroker(_snapshot(
        positions=[position], marks={"LINKUSDT": 10.0, "AVAXUSDT": 8.0, "SUIUSDT": 0.9},
    ))
    result = run_cycle(_config(tmp_path), ledger, broker, _market(), 0)
    assert result["action"] == "halted"
    assert broker.created == []
    assert "LINKUSDT" in ledger.halt_incomplete


def test_ambiguous_submit_reuses_the_same_link_id(tmp_path):
    market = _market()
    last = {symbol: float(market.close[symbol].iloc[-1]) for symbol in FROZEN_UNIVERSE}
    broker = FakeBroker(_snapshot(marks=last))
    broker.ambiguous = True
    broker.keep_ambiguous_order = False
    ledger = Ledger(tmp_path / "ledger.json")
    first = run_cycle(_config(tmp_path), ledger, broker, market, 0)
    assert first["action"] == "ambiguous"
    assert len(broker.attempts) == 1
    second = run_cycle(_config(tmp_path), ledger, broker, market, 0)
    assert broker.attempts[0] == broker.attempts[1]
    assert len(set(broker.attempts)) == 1


def test_account_failure_and_incomplete_pages_do_not_flatten(tmp_path):
    broker = FakeBroker(_snapshot())
    broker.fail = True
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.qty["LINKUSDT"] = 1.0
    result = run_cycle(_config(tmp_path), ledger, broker, _market(), 0)
    assert result["action"] == "blocked" and broker.created == [] and broker.cancelled == []
    broker.fail = False
    broker.snapshot = _snapshot(pages_complete=False)
    again = run_cycle(_config(tmp_path), ledger, broker, _market(), 0)
    assert again["action"] == "blocked" and broker.created == []


def test_forming_bar_and_low_free_collateral_fail_closed(tmp_path):
    stale = _market(now_offset=60_000)
    assert stale.stale
    broker = FakeBroker(_snapshot())
    result = run_cycle(_config(tmp_path), Ledger(tmp_path / "ledger.json"), broker, stale, stale.last_open_ms)
    assert broker.created == []
    assert any("stale" in item or "forming" in item for item in result["blockers"])

    market = _market()
    poor = FakeBroker(_snapshot(free_usdt=9, marks=snapshot_marks()))
    # 排名用行情，下单用快照价格；9 USDT 一半后低于最小名义，整篮取消。
    blocked = run_cycle(_config(tmp_path), Ledger(tmp_path / "poor.json"), poor, market, 0)
    assert poor.created == []
    assert blocked["action"] == "no_trade"
    assert blocked["budget"] == 9


def test_fees_funding_and_wallet_deposit_are_not_confused(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    owned = "cxls" + "a" * 32
    ledger.intents[owned] = {"status": "submitted"}
    ledger.tracked.add("LINKUSDT")
    fill = StrategyFill("e1", owned, "LINKUSDT", "Buy", 1.0, 10.0, 0.2)
    apply_fill(ledger, fill)
    apply_fill(ledger, fill)
    assert ledger.qty["LINKUSDT"] == pytest.approx(1.0)
    assert ledger.realized == pytest.approx(-0.2)
    apply_funding(ledger, FundingEvent("f1", "LINKUSDT", -0.5))
    apply_funding(ledger, FundingEvent("f2", "BTCUSDT", -3))
    assert ledger.funding == pytest.approx(-0.5)
    assert ledger.pnl({"LINKUSDT": 9.0}) == pytest.approx(-0.2 - 0.5 - 1.0)
    ledger.last_wallet = 5000
    assert ledger.pnl({"LINKUSDT": 9.0}) == pytest.approx(-1.7)


def test_unhedged_owned_long_is_reduced_without_opening(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.qty["LINKUSDT"] = 3.0
    ledger.avg["LINKUSDT"] = 10.0
    ledger.tracked.add("LINKUSDT")
    broker = FakeBroker(_snapshot(marks={"LINKUSDT": 10.0, "AVAXUSDT": 8.0, "SUIUSDT": 0.9}))
    result = run_cycle(_config(tmp_path), ledger, broker, _market(), 0)
    assert any(order.reduce_only and order.symbol == "LINKUSDT" for order in broker.created)
    assert all(order.reduce_only for order in broker.created)
    assert result["action"] in {"reduced", "monitored", "reducing_before_new_basket"}


def test_example_preflight_stays_closed_and_has_no_secrets(monkeypatch):
    monkeypatch.setattr(
        "super_trader.bybit_private.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")),
    )
    code = main(["crypto-xs-live", "--config", "config/crypto_xs_live.example.json", "--preflight"])
    assert code == 2
    text = Path("config/crypto_xs_live.example.json").read_text(encoding="utf-8")
    assert "api_key" not in text and "secret" not in text
    report = readiness(load_live_config("config/crypto_xs_live.example.json", {}))
    assert report["wallet_is_not_budget"] is True
    assert report["isolated_margin_reserves_budget"] is False
    assert report["allocation_usdt"] == ALLOCATION_USDT
    assert LOSS_CUTOFF_USDT == 20
    assert any("主账户" in item for item in report["unresolved"])
