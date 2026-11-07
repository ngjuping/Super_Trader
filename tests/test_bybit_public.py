"""Bybit 公开行情适配器：类别必须写明，只保留已收盘日线，不携带密钥。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from super_trader.bybit_public import (
    DAY_MS, BybitPublicError, assert_public_path, bars_to_frame, default_getter,
    drop_unclosed, fetch_klines, frames_from_payload, is_stale, parse_kline_row,
    parse_launch_ms, require_category,
)

DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def _bar(start: int, close: float = 1.0) -> dict:
    return {
        "start_ms": start, "open": close, "high": close, "low": close,
        "close": close, "volume": 1.0, "turnover": 2_000_000.0,
    }


def _row(start: int, close: str = "1") -> list[str]:
    return [str(start), close, close, close, close, "1", "2000000"]


def test_category_has_no_default_and_private_paths_are_refused():
    with pytest.raises(ValueError):
        require_category("")
    with pytest.raises(BybitPublicError):
        assert_public_path("/v5/order/create")
    with pytest.raises(BybitPublicError):
        assert_public_path("/v5/position/list")


def test_parse_official_kline_order_and_drop_the_forming_bar():
    parsed = parse_kline_row(_row(DAY0, "10"))
    assert parsed["open"] == 10 and parsed["turnover"] == 2_000_000
    forming = DAY0 + DAY_MS
    closed = drop_unclosed([_bar(DAY0, 10), _bar(forming, 11)], forming + 1_000)
    assert [bar["start_ms"] for bar in closed] == [DAY0]
    assert is_stale(DAY0, forming + 3 * DAY_MS, max_age_ms=2 * DAY_MS)
    assert not is_stale(DAY0, forming + 60_000, max_age_ms=2 * DAY_MS)


def test_pagination_sends_category_and_stops_at_closed_bars():
    calls = []

    def getter(path, params):
        calls.append((path, dict(params)))
        end = params.get("end")
        if end is None:
            batch = [_row(DAY0 + 2 * DAY_MS), _row(DAY0 + DAY_MS)]
        elif end == str(DAY0 + DAY_MS - 1):
            batch = [_row(DAY0)]
        else:
            batch = []
        return {"retCode": 0, "result": {"category": "linear", "symbol": "BTCUSDT", "list": batch}}

    bars = fetch_klines(
        "linear", "BTCUSDT", as_of_ms=DAY0 + 3 * DAY_MS, getter=getter, limit=2,
    )
    assert [bar["start_ms"] for bar in bars] == [DAY0, DAY0 + DAY_MS, DAY0 + 2 * DAY_MS]
    assert calls[0][0] == "/v5/market/kline"
    assert all(params["category"] == "linear" for _, params in calls)
    assert all("api_key" not in params for _, params in calls)


def test_response_category_must_match_the_request():
    def getter(path, params):
        return {"retCode": 0, "result": {"category": "spot", "list": []}}

    with pytest.raises(BybitPublicError):
        fetch_klines("linear", "BTCUSDT", as_of_ms=DAY0 + DAY_MS, getter=getter)


def test_launch_time_drops_earlier_bars_without_backfill():
    payload = {"retCode": 0, "result": {"category": "spot", "list": [
        {"symbol": "ETHUSDT", "launchTime": str(DAY0 + DAY_MS)}
    ]}}
    assert parse_launch_ms(payload, "spot", "ETHUSDT") == DAY0 + DAY_MS
    frame = bars_to_frame([_bar(DAY0, 5), _bar(DAY0 + DAY_MS, 9)], launch_ms=DAY0 + DAY_MS)
    assert len(frame) == 1
    assert frame["close"].iloc[0] == 9
    assert frame.index[0] == pd.Timestamp(DAY0 + DAY_MS, unit="ms", tz="UTC")


def test_frames_payload_rejects_leverage_and_live_capital():
    with pytest.raises(ValueError):
        frames_from_payload({"leverage": 5, "symbols": {"BTCUSDT": []}})
    with pytest.raises(ValueError):
        frames_from_payload({"live_capital_usdt": 100, "symbols": {"BTCUSDT": []}})


def test_default_getter_is_get_without_api_key(monkeypatch):
    captured = {}

    class Resp:
        def read(self):
            return json.dumps({"retCode": 0, "result": {"category": "spot", "list": []}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = {key.lower(): value for key, value in req.header_items()}
        captured["method"] = req.get_method()
        return Resp()

    monkeypatch.setattr("super_trader.bybit_public.urllib.request.urlopen", fake_urlopen)
    default_getter()("/v5/market/kline", {"category": "spot", "symbol": "BTCUSDT", "interval": "D"})
    assert captured["method"] == "GET"
    assert "category=spot" in captured["url"]
    assert "api.bybit.com" in captured["url"]
    assert not any("api" in key and "key" in key for key in captured["headers"])
    assert not any(str(value).lower().startswith("x-bapi") for value in captured["headers"].values())
