"""私有客户端只签允许的路径，分页不完整就失败。不访问交易所。"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from super_trader.bybit_private import (
    AccountReadError, BybitPrivateClient, Credentials, OrderRejected, PrivatePathError,
    _parse_usdt_wallet, redact, sign_v5,
)
from super_trader.crypto_live import snapshot_instruments


def _client(transport, logs=None):
    return BybitPrivateClient(
        Credentials("test-key", "test-secret"),
        transport,
        clock=lambda: 1_700_000_000_000,
        log=(logs.append if logs is not None else None),
    )


def test_signature_matches_an_independent_hmac_and_logs_are_redacted():
    payload = '{"category":"linear","symbol":"LINKUSDT"}'
    expected = hmac.new(
        b"test-secret",
        b"1700000000000test-key5000" + payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    assert sign_v5("test-secret", "1700000000000", "test-key", "5000", payload) == expected
    seen = []

    def transport(method, url, headers, body):
        seen.append((method, url, headers, body))
        return 200, b'{"retCode":0,"result":{}}'

    logs = []
    client = _client(transport, logs)
    client.request("POST", "/v5/order/cancel", body={"category": "linear", "symbol": "LINKUSDT"})
    assert "test-secret" not in logs[0]
    assert "test-key" not in logs[0]
    assert seen[0][2]["X-BAPI-SIGN"] == sign_v5(
        "test-secret", "1700000000000", "test-key", "5000",
        seen[0][3].decode(),
    )


def test_mutation_paths_are_rejected_before_the_network():
    def transport(*args):
        raise AssertionError("network")

    client = _client(transport)
    for path in ("/v5/order/cancel-all", "/v5/position/set-leverage", "/v5/asset/transfer/inter-transfer"):
        with pytest.raises(PrivatePathError):
            client.request("POST", path, body={})


def test_blank_available_balance_is_not_zero_and_transfer_permission_is_refused():
    free, wallet, _borrow = _parse_usdt_wallet({"result": {"list": [{"coin": [{
        "coin": "USDT", "walletBalance": "121.50290183", "totalPositionIM": "",
        "totalOrderIM": "0", "locked": "0", "bonus": "0",
    }]}]}})
    assert free is None and wallet == pytest.approx(121.50290183)
    free, wallet, borrow = _parse_usdt_wallet({"result": {"list": [{"coin": [{
        "coin": "USDT", "walletBalance": "121.50290183", "totalPositionIM": "0",
        "totalOrderIM": "0", "locked": "0", "bonus": "0", "borrowAmount": "0",
    }]}]}})
    assert free == pytest.approx(121.50290183) and borrow == 0
    with pytest.raises(OrderRejected):
        from super_trader.bybit_private import _assert_permissions
        _assert_permissions({"result": {"permissions": {
            "ContractTrade": ["Order", "Position"],
            "Wallet": ["AccountTransfer"],
        }}})


def test_unfinished_pagination_fails_closed_instead_of_looking_empty():
    def transport(method, url, headers, body):
        if "/v5/position/list" in url:
            return 200, json.dumps({
                "retCode": 0, "result": {"list": [], "nextPageCursor": "still-going"},
            }).encode()
        return 200, b'{"retCode":0,"result":{"list":[]}}'

    client = _client(transport)
    with pytest.raises(AccountReadError):
        client.fetch_protection()


def test_instrument_rows_parse_the_checked_filters():
    row = {
        "symbol": "SUIUSDT", "status": "Trading", "contractType": "LinearPerpetual",
        "settleCoin": "USDT",
        "lotSizeFilter": {"qtyStep": "10", "minOrderQty": "10", "minNotionalValue": "5"},
    }
    from super_trader.crypto_live import parse_instrument
    parsed = parse_instrument(row)
    assert parsed.qty_step == 10 and parsed.min_notional == 5
    assert snapshot_instruments()["SUIUSDT"].min_qty == 10
