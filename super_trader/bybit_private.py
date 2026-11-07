"""Bybit v5 私有接口的窄客户端。

只允许行情、仓位、委托、成交、资金费和钱包的读取，以及按 orderLinkId
创建或撤销单笔订单。没有全账户撤单、没有调杠杆、没有划转、没有切换保证金模式。
密钥只放在请求头里，日志里会打掉。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .bybit_public import OFFICIAL_BASE
from .crypto_live import (
    AccountReadError, AmbiguousOrderError, DuplicateLinkError, ExchangeOrder,
    ExchangePosition, FundingEvent, Instrument, OrderRejected, ProtectionSnapshot,
    StrategyFill, parse_instrument,
)

ALLOWED = {
    ("GET", "/v5/market/kline"),
    ("GET", "/v5/market/instruments-info"),
    ("GET", "/v5/market/tickers"),
    ("GET", "/v5/position/list"),
    ("GET", "/v5/order/realtime"),
    ("GET", "/v5/execution/list"),
    ("GET", "/v5/account/transaction-log"),
    ("GET", "/v5/account/wallet-balance"),
    ("GET", "/v5/user/query-api"),
    ("POST", "/v5/order/create"),
    ("POST", "/v5/order/cancel"),
}
ORDER_FILTERS = ("Order", "StopOrder", "tpslOrder")
# 共享 UID 上，这些类别都要查完才算“当前没有别人的仓/单”。
PROTECTION_QUERIES = (
    ("linear", "USDT"),
    ("linear", "USDC"),
    ("inverse", None),
    ("option", None),
    ("spot", None),
)


class PrivatePathError(PermissionError):
    """调用方试图走被禁止的私有路径。"""


@dataclass
class Credentials:
    api_key: str
    api_secret: str

    def __repr__(self) -> str:
        return "Credentials(present=True)"

    def __str__(self) -> str:
        return "Credentials(present=True)"


def redact(text: str, secrets: tuple[str, ...]) -> str:
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[redacted]")
    return out


def sign_v5(secret: str, timestamp: str, api_key: str, recv_window: str, payload: str) -> str:
    message = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def load_credentials(env: dict[str, str]) -> Credentials:
    """从环境变量或文件路径读密钥。缺了就失败，不打印文件内容。"""
    key = env.get("BYBIT_API_KEY", "").strip()
    secret = env.get("BYBIT_API_SECRET", "").strip()
    key_file = env.get("BYBIT_API_KEY_FILE", "").strip()
    secret_file = env.get("BYBIT_API_SECRET_FILE", "").strip()
    if key_file:
        key = _read_secret_file(key_file)
    if secret_file:
        secret = _read_secret_file(secret_file)
    if not key or not secret:
        raise AccountReadError("BYBIT_API_KEY 或 BYBIT_API_SECRET（或对应 FILE）未设置")
    return Credentials(key, secret)


def _read_secret_file(path: str) -> str:
    from pathlib import Path
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AccountReadError("密钥文件无法读取") from exc


class BybitPrivateClient:
    def __init__(self, credentials: Credentials, transport, clock, *,
                 base_url: str = OFFICIAL_BASE, recv_window: str = "5000", log=None):
        self.credentials = credentials
        self.transport = transport
        self.clock = clock
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.log = log or (lambda message: None)

    def request(self, method: str, path: str, params: dict | None = None,
                body: dict | None = None) -> dict:
        if (method, path) not in ALLOWED:
            raise PrivatePathError(f"拒绝私有路径 {method} {path}")
        params = {key: value for key, value in (params or {}).items() if value is not None}
        query = urllib.parse.urlencode(sorted(params.items()))
        payload = json.dumps(body, separators=(",", ":")) if body is not None else query
        timestamp = str(int(self.clock()))
        signature = sign_v5(
            self.credentials.api_secret, timestamp, self.credentials.api_key,
            self.recv_window, payload,
        )
        headers = {
            "X-BAPI-API-KEY": self.credentials.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type": "application/json",
        }
        url = self.base_url + path + (("?" + query) if query and body is None else "")
        raw = payload.encode() if body is not None else None
        self.log(redact(
            f"{method} {path}",
            (self.credentials.api_key, self.credentials.api_secret, signature),
        ))
        try:
            status, content = self.transport(method, url, headers, raw)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            raise AmbiguousOrderError("私有请求没有收到可判定的结果") from exc
        if status in {502, 503, 504} or not content:
            raise AmbiguousOrderError(f"私有请求结果不确定: HTTP {status}")
        try:
            parsed = json.loads(content.decode())
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AmbiguousOrderError("私有响应无法解析") from exc
        ret = int(parsed.get("retCode", -1))
        if ret != 0:
            message = str(parsed.get("retMsg") or "")
            duplicate = "orderlinkid" in message.lower() and "duplicate" in message.lower()
            if duplicate or ret == 110072:
                raise DuplicateLinkError(message or "duplicate orderLinkId")
            if status >= 500:
                raise AmbiguousOrderError(message or "server error")
            raise OrderRejected(message or f"retCode {ret}")
        return parsed

    def _pages(self, path: str, params: dict, limit: int = 50, max_pages: int = 20) -> list:
        rows: list = []
        cursor = None
        for _ in range(max_pages):
            query = dict(params)
            query["limit"] = str(limit)
            if cursor:
                query["cursor"] = cursor
            payload = self.request("GET", path, query)
            result = payload.get("result") or {}
            rows.extend(result.get("list") or [])
            cursor = result.get("nextPageCursor") or ""
            if not cursor:
                return rows
        raise AccountReadError(f"{path} 分页没有结束，拒绝把账户看成空的")

    def fetch_protection(self) -> ProtectionSnapshot:
        positions: list[ExchangePosition] = []
        orders: list[ExchangeOrder] = []
        categories: list[str] = []
        leverage: dict[str, float] = {}
        modes: dict[str, int] = {}
        try:
            for category, settle in PROTECTION_QUERIES:
                categories.append(category if settle is None else f"{category}:{settle}")
                base = {"category": category}
                if settle:
                    base["settleCoin"] = settle
                if category != "spot":
                    for item in self._pages("/v5/position/list", base):
                        position = _parse_position(item)
                        positions.append(position)
                        if position.symbol and position.leverage is not None:
                            leverage[position.symbol] = position.leverage
                        if position.symbol:
                            modes[position.symbol] = position.position_idx
                filters = ORDER_FILTERS if category not in {"option", "spot"} else ("Order",)
                for order_filter in filters:
                    query = dict(base)
                    if category not in {"spot"}:
                        query["orderFilter"] = order_filter
                    for item in self._pages("/v5/order/realtime", query):
                        orders.append(_parse_order(item))
            wallet = self.request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            free, wallet_usdt, borrow = _parse_usdt_wallet(wallet)
            api = self.request("GET", "/v5/user/query-api")
            _assert_permissions(api)
            fills = [
                _parse_fill(item)
                for item in self._pages("/v5/execution/list", {"category": "linear", "settleCoin": "USDT"})
            ]
            funding = []
            for item in self._pages("/v5/account/transaction-log", {"accountType": "UNIFIED", "category": "linear"}):
                event = _parse_funding(item)
                if event is not None and event.event_id:
                    funding.append(event)
            instruments = {}
            marks = {}
            from .crypto_live import FROZEN_UNIVERSE
            for symbol in FROZEN_UNIVERSE:
                info = self.request("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": symbol})
                for row in (info.get("result") or {}).get("list") or []:
                    inst = parse_instrument(row)
                    instruments[inst.symbol] = inst
                tickers = self.request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})
                for row in (tickers.get("result") or {}).get("list") or []:
                    price = row.get("markPrice") or row.get("lastPrice")
                    if price not in (None, ""):
                        marks[str(row.get("symbol") or symbol)] = float(price)
        except (OrderRejected, DuplicateLinkError, PrivatePathError, AmbiguousOrderError) as exc:
            raise AccountReadError(str(exc)) from exc
        return ProtectionSnapshot(
            positions=positions, orders=orders, fills=fills, funding=funding,
            wallet_usdt=wallet_usdt, free_usdt=free, borrow_usdt=borrow,
            leverage_by_symbol=leverage, position_mode_by_symbol=modes,
            instruments=instruments, marks=marks, pages_complete=True,
            categories_checked=tuple(categories),
        )


def _parse_position(row: dict) -> ExchangePosition:
    side = str(row.get("side") or "")
    size = float(row.get("size") or 0)
    qty = size if side == "Buy" else (-size if side == "Sell" else 0.0)
    if size == 0:
        qty = 0.0
    return ExchangePosition(
        symbol=str(row.get("symbol") or ""),
        qty=qty,
        mark=_float_or_none(row.get("markPrice")),
        position_idx=int(row.get("positionIdx") or 0),
        leverage=_float_or_none(row.get("leverage")),
    )


def _parse_order(row: dict) -> ExchangeOrder:
    return ExchangeOrder(
        order_link_id=str(row.get("orderLinkId") or ""),
        symbol=str(row.get("symbol") or ""),
        side=str(row.get("side") or ""),
        qty=float(row.get("qty") or 0),
        leaves_qty=float(row.get("leavesQty") or 0),
        reduce_only=bool(row.get("reduceOnly")),
        status=str(row.get("orderStatus") or ""),
    )


def _parse_fill(row: dict) -> StrategyFill:
    return StrategyFill(
        exec_id=str(row.get("execId") or ""),
        order_link_id=str(row.get("orderLinkId") or ""),
        symbol=str(row.get("symbol") or ""),
        side=str(row.get("side") or ""),
        qty=float(row.get("execQty") or 0),
        price=float(row.get("execPrice") or 0),
        fee=float(row.get("execFee") or 0),
    )


def _parse_funding(row: dict) -> FundingEvent | None:
    kind = str(row.get("type") or "")
    if kind not in {"SETTLEMENT", "FUNDING"}:
        return None
    amount = row.get("funding")
    if amount in (None, ""):
        amount = row.get("change")
    if amount in (None, ""):
        return None
    return FundingEvent(
        event_id=str(row.get("id") or row.get("tradeId") or ""),
        symbol=str(row.get("symbol") or ""),
        amount=float(amount),
    )


def _parse_usdt_wallet(payload: dict) -> tuple[float | None, float | None, float]:
    accounts = (payload.get("result") or {}).get("list") or []
    for account in accounts:
        for coin in account.get("coin") or []:
            if coin.get("coin") != "USDT":
                continue
            wallet = _optional_float(coin.get("walletBalance"))
            parts = [
                _optional_float(coin.get("totalPositionIM")),
                _optional_float(coin.get("totalOrderIM")),
                _optional_float(coin.get("locked")),
                _optional_float(coin.get("bonus")),
            ]
            borrow = _optional_float(coin.get("borrowAmount")) or 0.0
            if wallet is None or any(part is None for part in parts):
                return None, wallet, borrow
            free = wallet - parts[0] - parts[1] - parts[2] - parts[3]
            return free, wallet, borrow
    return None, None, 0.0


def _assert_permissions(payload: dict) -> None:
    result = payload.get("result") or {}
    permissions = result.get("permissions") or {}
    wallet = set(permissions.get("Wallet") or [])
    contract = set(permissions.get("ContractTrade") or [])
    if wallet & {"AccountTransfer", "Withdraw", "SubMemberTransfer"}:
        raise OrderRejected("密钥带有划转或提现权限，拒绝使用")
    if "Order" not in contract or "Position" not in contract:
        raise OrderRejected("密钥没有合约下单和持仓权限")


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def urllib_transport(timeout: float = 20.0):
    def transport(method: str, url: str, headers: dict, body: bytes | None):
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
    return transport
