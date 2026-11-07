"""加密货币横截面纸面路径。测试核对的是规则和账本，不是一条声称能赚钱的回测。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from super_trader.bybit_public import DAY_MS
from super_trader.cli import main
from super_trader.crypto_xs import (
    ISOLATION_NOTE, REQUIRED_LIVE_CONFIRMATIONS, RESEARCH_UNIVERSE_USDT,
    CryptoXSConfig, IsolationError, LiveTradingDisabled, PaperBook, PaperLimits,
    Policy, apply_exposure_limits, apply_targets, causal_relative_return,
    client_order_id,     fit_direction, forward_relative, latest_targets, live_block_reason,
    prepare_panels, ranked_symbols, refuse_live, reject_account_action,
    research_config, run_crypto_xs, select_policy, simulate, target_weights,
)


def _cfg(category="spot", n_symbols=4, **overrides) -> CryptoXSConfig:
    symbols = tuple(overrides.pop("symbols", tuple(f"C{i:02d}USDT" for i in range(n_symbols))))
    params = dict(
        category=category,
        symbols=symbols,
        paper_equity_usdt=10_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_history=5,
        min_names=2,
        min_median_turnover=0.0,
        liquidity_window=3,
        fit_fraction=0.50,
        calibration_fraction=0.25,
        lookbacks=(5,),
        horizons=(1,),
        top_ks=(1,),
        min_fit_days=10,
        min_cal_rebalances=1,
        limits=PaperLimits(max_name_weight=0.5, max_gross=1.0, max_net=1.0),
    )
    params.update(overrides)
    return CryptoXSConfig(**params)


def _constant_market(n=20, symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"),
                     close=100.0, open_=110.0, turnover=5_000_000.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = list(symbols)
    close_df = pd.DataFrame(close, index=idx, columns=cols)
    open_df = pd.DataFrame(open_, index=idx, columns=cols)
    turn = pd.DataFrame(turnover, index=idx, columns=cols)
    return open_df, close_df, turn


def _momentum_market(n=360, n_symbols=8, seed=3, phi=0.9, shock=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    cols = [f"C{i:02d}USDT" for i in range(n_symbols)]
    price = np.full(n_symbols, 100.0)
    prev = np.zeros(n_symbols)
    closes, opens = [], []
    for _ in range(n):
        ret = phi * prev + rng.normal(0, shock, n_symbols)
        opens.append(price * 1.001)
        price = price * (1 + ret)
        closes.append(price.copy())
        prev = ret
    close_df = pd.DataFrame(closes, index=idx, columns=cols)
    open_df = pd.DataFrame(opens, index=idx, columns=cols)
    turn = pd.DataFrame(5_000_000.0, index=idx, columns=cols)
    return open_df, close_df, turn


def test_research_defaults_are_paper_only_and_xaut_is_absent():
    cfg = research_config("linear")
    assert cfg.universe_source == "research_default"
    assert cfg.category == "linear"
    assert "XAUTUSDT" not in cfg.symbols
    assert "SOLUSDT" in RESEARCH_UNIVERSE_USDT
    assert "leverage" not in CryptoXSConfig.__dataclass_fields__
    with pytest.raises(ValueError):
        research_config("inverse")
    with pytest.raises(ValueError):
        _cfg(fee_bps=-1)
    with pytest.raises(ValueError):
        CryptoXSConfig(category="swap", symbols=("BTCUSDT", "ETHUSDT"))


def test_alignment_keeps_weekends_and_does_not_backfill_listings():
    idx = pd.date_range("2024-01-05", periods=4, freq="D", tz="UTC")  # 周五起，含周六
    assert int((idx.day_of_week >= 5).sum()) > 0
    btc = pd.DataFrame({
        "open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0, 4.0],
        "volume": 1.0, "turnover": 1.0,
    }, index=idx)
    eth = btc.iloc[2:].copy()
    eth["close"] = [30.0, 40.0]
    _open, close, _turn = prepare_panels({"ETHUSDT": eth, "BTCUSDT": btc})
    assert len(close) == 4
    assert np.isnan(close.loc[idx[0], "ETHUSDT"])
    assert np.isnan(close.loc[idx[1], "ETHUSDT"])
    assert close.loc[idx[2], "ETHUSDT"] == 30


def test_ranking_tie_break_is_symbol_order_and_limits_block_shorts_on_spot():
    assert ranked_symbols(pd.Series({"MMM": 1.0, "AAA": 1.0, "ZZZ": 0.2})) == ["AAA", "MMM", "ZZZ"]
    again = ranked_symbols(pd.Series({"ZZZ": 0.2, "AAA": 1.0, "MMM": 1.0}))
    assert again == ["AAA", "MMM", "ZZZ"]
    order = [f"C{i:02d}USDT" for i in range(8)]
    spot = target_weights(order, Policy(5, 1, 2, "momentum", 1), "spot", PaperLimits(), 6)
    perp = target_weights(order, Policy(5, 1, 2, "momentum", 1), "linear", PaperLimits(), 6)
    inverse = target_weights(order, Policy(5, 1, 2, "reversal", 1), "inverse", PaperLimits(), 6)
    assert spot and all(weight > 0 for weight in spot.values())
    assert sum(spot.values()) <= 1
    assert any(weight < 0 for weight in perp.values())
    assert abs(sum(perp.values())) < 1e-9
    assert abs(sum(abs(weight) for weight in perp.values()) - 1) < 1e-9
    assert any(weight < 0 for weight in inverse.values())
    capped = apply_exposure_limits({"AAAUSDT": 1.0}, PaperLimits(0.25, 1, 1), allow_short=False)
    assert capped["AAAUSDT"] == 0.25


def test_causal_score_ignores_future_prices_and_keeps_large_crypto_moves():
    _open, close, turn = _momentum_market(n=40, n_symbols=4, seed=1)
    cfg = _cfg(symbols=tuple(close.columns), min_names=2)
    base = causal_relative_return(close, turn, 5, cfg)
    changed = close.copy()
    changed.iloc[-5:] = changed.iloc[-5:] * 3
    later = causal_relative_return(changed, turn, 5, cfg)
    pd.testing.assert_frame_equal(base.iloc[:-5], later.iloc[:-5])
    crashed = close.copy()
    crashed.iloc[-1, 0] = crashed.iloc[-2, 0] * 0.6
    scored = causal_relative_return(crashed, turn, 5, cfg)
    assert np.isfinite(scored.iloc[-1, 0])


def test_liquidity_and_history_remove_a_name_without_deleting_the_day():
    _open, close, turn = _constant_market(n=12)
    turn = turn.copy()
    turn["DDDUSDT"] = 0.0
    cfg = _cfg(symbols=tuple(close.columns), min_median_turnover=100.0, min_history=5, liquidity_window=3)
    scored = causal_relative_return(close, turn, 5, cfg)
    last = ranked_symbols(scored.iloc[-1])
    assert "DDDUSDT" not in last
    assert len(last) >= 2


def test_next_open_fill_fee_and_idempotent_replay():
    open_, close, turn = _constant_market(n=12, close=100.0, open_=110.0)
    cfg = _cfg(symbols=tuple(close.columns), fee_bps=0.0, slippage_bps=0.0)
    policy = Policy(5, 1, 1, "momentum", rebalance_every=20)
    stats = simulate(open_, close, turn, policy, cfg, start=0, end=len(close))
    assert stats["rebalances"] == 1
    qty = stats["qty"]["AAAUSDT"]
    assert qty == pytest.approx(5_000 / 110)
    assert qty != pytest.approx(5_000 / 100)
    assert stats["ending_equity"] == pytest.approx(5_000 + qty * 100)
    assert stats["traded_notional"] == pytest.approx(5_000)
    assert stats["turnover"] == pytest.approx(0.5)

    book = PaperBook(cash=1_000.0, qty={}, filled_ids=set())
    weights = {"AAAUSDT": 0.5, "BBBUSDT": 0.5}
    opens = {"AAAUSDT": 10.0, "BBBUSDT": 20.0}
    kwargs = dict(
        decision_equity=1_000.0, fee_bps=10.0, slippage_bps=0.0, category="spot",
        signal_ms=DAY_MS, allowed_symbols=("AAAUSDT", "BBBUSDT"),
    )
    first = apply_targets(book, weights, opens, **kwargs)
    qty_after = dict(book.qty)
    fees_after = book.fees
    second = apply_targets(book, weights, opens, **kwargs)
    assert first["applied"] and second["reason"] == "duplicate"
    assert book.qty == qty_after and book.fees == fees_after
    assert book.cash >= -1e-6
    assert all(qty >= -1e-8 for qty in book.qty.values())
    assert fees_after > 0


def test_spot_target_outside_universe_is_rejected_before_the_fail_line():
    book = PaperBook(cash=1_000.0, qty={}, filled_ids=set())
    with pytest.raises(IsolationError):
        apply_targets(
            book, {"XAUTUSDT": 1.0}, {"XAUTUSDT": 10.0},
            decision_equity=1_000.0, fee_bps=0.0, slippage_bps=0.0, category="spot",
            signal_ms=1, allowed_symbols=("BTCUSDT",),
        )
    assert book.cash == 1_000.0 and not book.qty


def test_perp_long_short_is_unlevered_and_spot_cannot_borrow():
    book = PaperBook(cash=1_000.0, qty={}, filled_ids=set())
    apply_targets(
        book, {"AAAUSDT": 0.5, "BBBUSDT": -0.5},
        {"AAAUSDT": 10.0, "BBBUSDT": 20.0},
        decision_equity=1_000.0, fee_bps=0.0, slippage_bps=0.0, category="linear",
        signal_ms=5, allowed_symbols=("AAAUSDT", "BBBUSDT"),
    )
    assert book.qty["AAAUSDT"] == pytest.approx(50)
    assert book.qty["BBBUSDT"] == pytest.approx(-25)
    assert book.equity({"AAAUSDT": 10.0, "BBBUSDT": 20.0}) == pytest.approx(1_000)

    spot = PaperBook(cash=1_000.0, qty={}, filled_ids=set())
    apply_targets(
        spot, {"AAAUSDT": 0.5, "BBBUSDT": -0.5},
        {"AAAUSDT": 10.0, "BBBUSDT": 20.0},
        decision_equity=1_000.0, fee_bps=0.0, slippage_bps=0.0, category="spot",
        signal_ms=5, allowed_symbols=("AAAUSDT", "BBBUSDT"),
    )
    assert spot.qty.get("BBBUSDT", 0.0) == 0.0
    assert spot.cash >= -1e-6


def test_account_wide_actions_and_live_transport_stay_closed():
    for action in ("cancel_all", "cancelAll", "close-all-positions"):
        with pytest.raises(IsolationError):
            reject_account_action(action)
    missing = live_block_reason({})
    for name in REQUIRED_LIVE_CONFIRMATIONS:
        assert name in missing
    confirmed = live_block_reason({name: True for name in REQUIRED_LIVE_CONFIRMATIONS})
    assert "没有私有下单通道" in confirmed
    with pytest.raises(LiveTradingDisabled):
        refuse_live({name: True for name in REQUIRED_LIVE_CONFIRMATIONS})


def test_selection_does_not_read_the_test_window_and_does_not_flip_on_it():
    open_, close, turn = _momentum_market()
    cfg = _cfg(
        category="spot",
        symbols=tuple(close.columns),
        lookbacks=(5, 20),
        horizons=(1, 5),
        top_ks=(1, 2),
        min_history=30,
        min_names=6,
        min_fit_days=20,
        min_cal_rebalances=2,
        fee_bps=5.0,
        slippage_bps=1.0,
    )
    policy, _info = select_policy(open_, close, turn, cfg)
    assert policy is not None
    assert policy.direction == "momentum"
    cal_end = int(len(close) * 0.75)
    open2, close2, turn2 = open_.copy(), close.copy(), turn.copy()
    open2.iloc[cal_end:] = np.nan
    close2.iloc[cal_end:] = np.nan
    wiped, _info2 = select_policy(open2, close2, turn2, cfg)
    assert wiped == policy

    as_of = int(close.index[-1].timestamp() * 1000) + DAY_MS + 60_000
    report = run_crypto_xs(open_, close, turn, cfg, as_of_ms=as_of)
    close_up = close.copy()
    close_up.iloc[cal_end:] *= 1.3
    changed = run_crypto_xs(open_, close_up, turn, cfg, as_of_ms=as_of)
    assert changed["policy"] == report["policy"]
    assert changed["selection"]["test_used_for_selection"] is False
    assert report["test_simulation"]["ending_equity"] != changed["test_simulation"]["ending_equity"]
    assert "50.38%" in report["a_share_accuracy"]
    assert report["leverage"] is None and report["live_capital_usdt"] is None
    assert report["live_trading"] is False
    relative = causal_relative_return(close, turn, policy.lookback, cfg)
    future = forward_relative(close, policy.horizon, cfg.min_names)
    assert fit_direction(relative, future, int(len(close) * 0.5), policy.horizon, cfg) == "momentum"


def test_stale_bar_produces_no_pending_order_and_fresh_ids_repeat():
    open_, close, turn = _constant_market(n=12)
    cfg = _cfg(symbols=tuple(close.columns))
    policy = Policy(5, 1, 1, "momentum", 1)
    last_ms = int(close.index[-1].timestamp() * 1000)
    stale = latest_targets(open_, close, turn, policy, cfg, last_ms + DAY_MS + cfg.max_staleness_ms + 1)
    assert stale["stale"] and stale["pending_intents"] == []
    fresh_ms = last_ms + DAY_MS + 60_000
    first = latest_targets(open_, close, turn, policy, cfg, fresh_ms)
    second = latest_targets(open_, close, turn, policy, cfg, fresh_ms)
    assert first["reason"] == "pending_next_open"
    assert first["pending_intents"]
    assert all(item["sent"] is False for item in first["pending_intents"])
    assert [item["client_order_id"] for item in first["pending_intents"]] == [
        item["client_order_id"] for item in second["pending_intents"]
    ]
    assert client_order_id("spot", "AAAUSDT", last_ms, 0.5) == client_order_id("spot", "AAAUSDT", last_ms, 0.5)


def test_missing_next_open_skips_the_whole_basket():
    open_, close, turn = _constant_market(n=12, close=100.0, open_=110.0)
    open_.loc[open_.index[6], "AAAUSDT"] = np.nan
    cfg = _cfg(symbols=tuple(close.columns))
    stats = simulate(open_, close, turn, Policy(5, 1, 1, "momentum", 20), cfg, start=0, end=len(close))
    assert stats["rebalances"] == 0
    assert stats["skipped"] == 1
    assert stats["qty"] == {}


def test_holding_period_changes_rebalance_count():
    open_, close, turn = _constant_market(n=40)
    cfg = _cfg(symbols=tuple(close.columns))
    daily = simulate(open_, close, turn, Policy(5, 1, 1, "momentum", 1), cfg, start=0, end=len(close))
    held = simulate(open_, close, turn, Policy(5, 1, 1, "momentum", 5), cfg, start=0, end=len(close))
    assert held["rebalances"] < daily["rebalances"]
    assert held["rebalances"] == 7


def test_cli_paper_report_and_live_refusal(tmp_path: Path):
    open_, close, turn = _constant_market(n=16)
    payload = {"as_of_ms": int(close.index[-1].timestamp() * 1000) + DAY_MS + 60_000, "symbols": {}}
    for symbol in close.columns:
        bars = []
        for ts, opened, closed, traded in zip(open_.index, open_[symbol], close[symbol], turn[symbol]):
            bars.append({
                "start_ms": int(ts.timestamp() * 1000),
                "open": float(opened), "high": float(closed), "low": float(closed),
                "close": float(closed), "volume": 1.0, "turnover": float(traded),
            })
        payload["symbols"][symbol] = bars
    src = tmp_path / "bars.json"
    out = tmp_path / "paper.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    symbols = ",".join(close.columns)
    code = main([
        "crypto-xs", "--category", "spot", "--symbols", symbols, "--bars-json", str(src),
        "--min-history", "5", "--min-names", "2", "--min-turnover", "0", "--liquidity-window", "3",
        "--lookback", "5", "--horizon", "1", "--top-k", "1", "--direction", "momentum",
        "--fee-bps", "0", "--slippage-bps", "0", "--report", str(out),
    ])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "横截面" in report["description"]
    assert report["selection"]["mode"] == "caller_pinned"
    assert report["isolation"] == ISOLATION_NOTE
    assert report["latest"]["pending_intents"]
    assert all(item["sent"] is False for item in report["latest"]["pending_intents"])

    live = main([
        "crypto-xs", "--category", "linear", "--live",
        "--confirm-universe", "--confirm-category", "--confirm-funding",
        "--confirm-leverage", "--confirm-loss-limit",
    ])
    assert live == 2
    with pytest.raises(SystemExit):
        main(["crypto-xs"])
