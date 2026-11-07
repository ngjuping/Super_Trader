"""Bybit v5 公开行情。只读 /v5/market，不接收密钥，也不提供下单、撤单或查持仓。

官方主机默认是 https://api.bybit.com 。category 必须由调用方写明
（spot / linear / inverse），这里不替任何一笔请求挑选现货或合约。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

OFFICIAL_BASE = "https://api.bybit.com"
CATEGORIES = ("spot", "linear", "inverse")
PERP_CATEGORIES = ("linear", "inverse")
DAY_MS = 86_400_000
INTERVAL_MS = {"D": DAY_MS}
# 日线研究只接受 2010 之后的时间戳，避免坏数据把日历轴拉到 1970 年。
EARLIEST_MS = 1_262_304_000_000
MARKET_PATHS = ("/v5/market/kline", "/v5/market/instruments-info")


class BybitPublicError(Exception):
    """公开行情请求或解析失败。不包含任何私有交易动作。"""


def require_category(category: str) -> str:
    """产品类别没有默认值。现货和永续必须由调用方写出来。"""
    if category not in CATEGORIES:
        raise ValueError(
            "Bybit category 必须显式指定为 spot、linear 或 inverse，没有默认值"
        )
    return category


def allows_short(category: str) -> bool:
    """做空只在永续（linear / inverse）上有意义。现货不借券。"""
    require_category(category)
    return category in PERP_CATEGORIES


def assert_public_path(path: str) -> None:
    if path not in MARKET_PATHS:
        raise BybitPublicError(f"公开适配器拒绝非行情路径: {path}")


def _reject_secret_params(params: dict) -> None:
    banned = {"api_key", "api-key", "apikey", "sign", "secret", "api_secret"}
    found = [str(k) for k in params if str(k).lower() in banned]
    if found:
        raise BybitPublicError("公开适配器拒绝在请求里携带密钥或签名")


def parse_kline_row(row: list) -> dict:
    """Bybit v5 K 线： [startTime, open, high, low, close, volume, turnover]。"""
    if not isinstance(row, (list, tuple)) or len(row) < 7:
        raise BybitPublicError("K 线字段不足，期望 start/open/high/low/close/volume/turnover")
    try:
        bar = {
            "start_ms": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "turnover": float(row[6]),
        }
    except (TypeError, ValueError) as exc:
        raise BybitPublicError("K 线字段无法解析为数字") from exc
    if bar["start_ms"] < EARLIEST_MS:
        raise BybitPublicError("K 线时间早于研究日历下限")
    if min(bar["open"], bar["high"], bar["low"], bar["close"]) <= 0:
        raise BybitPublicError("K 线价格必须为正")
    return bar


def drop_unclosed(bars: list[dict], as_of_ms: int, bar_ms: int = DAY_MS) -> list[dict]:
    """只保留已经走完的 bar。start + 周期 > as_of 的是正在形成的 K 线。"""
    closed: dict[int, dict] = {}
    for bar in bars:
        if int(bar["start_ms"]) + int(bar_ms) <= int(as_of_ms):
            closed[int(bar["start_ms"])] = bar
    return [closed[key] for key in sorted(closed)]


def is_stale(last_start_ms: int, as_of_ms: int, *, bar_ms: int = DAY_MS,
             max_age_ms: int) -> bool:
    """最后一根已收盘 K 线的收盘时刻距 as_of 过远，则行情过期。"""
    closed_at = int(last_start_ms) + int(bar_ms)
    return int(as_of_ms) - closed_at > int(max_age_ms)


def default_getter(base_url: str = OFFICIAL_BASE, timeout: float = 20.0):
    """返回只发 GET 的公开行情函数。请求头里没有 API key。"""

    def getter(path: str, params: dict) -> dict:
        assert_public_path(path)
        _reject_secret_params(params)
        query = urllib.parse.urlencode({k: params[k] for k in params})
        url = base_url.rstrip("/") + path + "?" + query
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "super-trader-crypto-xs"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise BybitPublicError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BybitPublicError(f"公开行情请求失败: {exc.reason}") from exc
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise BybitPublicError("公开行情返回的不是 JSON") from exc
        if not isinstance(payload, dict):
            raise BybitPublicError("公开行情返回结构异常")
        return payload

    return getter


def _ok_payload(payload: dict, category: str) -> dict:
    if int(payload.get("retCode", -1)) != 0:
        raise BybitPublicError(str(payload.get("retMsg") or "Bybit retCode 非 0"))
    result = payload.get("result") or {}
    if not isinstance(result, dict):
        raise BybitPublicError("Bybit result 不是对象")
    got = result.get("category")
    if got and got != category:
        raise BybitPublicError(f"返回的 category={got} 与请求的 {category} 不一致")
    return result


def fetch_klines(category: str, symbol: str, *, interval: str = "D",
                 start_ms: int | None = None, end_ms: int | None = None,
                 as_of_ms: int, getter, limit: int = 1000, max_pages: int = 20) -> list[dict]:
    """按时间向前翻页，拉取日线并丢掉未收盘的最后一根。"""
    require_category(category)
    if interval not in INTERVAL_MS:
        raise ValueError("加密货币横截面研究路径只使用日线（interval=D）已收盘 K 线")
    if not symbol or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for ch in symbol):
        raise ValueError(f"非法交易对代码: {symbol}")
    bar_ms = INTERVAL_MS[interval]
    page_limit = max(1, min(int(limit), 1000))
    rows: list[dict] = []
    seen: set[int] = set()
    cursor_end = end_ms
    for _ in range(max_pages):
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": str(page_limit),
        }
        if start_ms is not None:
            params["start"] = str(int(start_ms))
        if cursor_end is not None:
            params["end"] = str(int(cursor_end))
        result = _ok_payload(getter("/v5/market/kline", params), category)
        batch = result.get("list") or []
        if not batch:
            break
        starts: list[int] = []
        for item in batch:
            bar = parse_kline_row(item)
            starts.append(bar["start_ms"])
            if start_ms is not None and bar["start_ms"] < int(start_ms):
                continue
            if bar["start_ms"] in seen:
                continue
            seen.add(bar["start_ms"])
            rows.append(bar)
        oldest = min(starts)
        if len(batch) < page_limit:
            break
        next_end = oldest - 1
        if cursor_end is not None and next_end >= int(cursor_end):
            break
        if start_ms is not None and oldest <= int(start_ms):
            break
        cursor_end = next_end
    else:
        raise BybitPublicError(f"{symbol} K 线翻页超过 {max_pages} 页")
    return drop_unclosed(rows, as_of_ms, bar_ms)


def parse_launch_ms(payload: dict, category: str, symbol: str) -> int | None:
    result = _ok_payload(payload, category)
    rows = result.get("list") or []
    if not rows:
        return None
    row = rows[0]
    if row.get("symbol") and row["symbol"] != symbol:
        raise BybitPublicError(f"instruments-info 返回了 {row.get('symbol')} 而不是 {symbol}")
    launch = row.get("launchTime")
    if launch in (None, "", "0", 0):
        return None
    return int(launch)


def fetch_launch_ms(category: str, symbol: str, getter) -> int | None:
    require_category(category)
    payload = getter("/v5/market/instruments-info", {"category": category, "symbol": symbol})
    return parse_launch_ms(payload, category, symbol)


def bars_to_frame(bars: list[dict], launch_ms: int | None = None) -> pd.DataFrame:
    """转成 UTC 日线表。上市日之前的 bar 直接丢掉，不用后价回填。"""
    columns = ["open", "high", "low", "close", "volume", "turnover"]
    kept = bars
    if launch_ms is not None:
        kept = [bar for bar in bars if int(bar["start_ms"]) >= int(launch_ms)]
    if not kept:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], tz="UTC")
        return empty
    frame = pd.DataFrame(kept).sort_values("start_ms").drop_duplicates("start_ms", keep="last")
    frame.index = pd.to_datetime(frame["start_ms"].to_numpy(), unit="ms", utc=True)
    frame.index.name = "open_time"
    return frame[columns].sort_index()


def frames_from_payload(payload: dict) -> dict[str, pd.DataFrame]:
    """从本地 JSON 组装多币日线。拒绝杠杆和实盘资金字段，避免把它们当成纸面默认。"""
    if not isinstance(payload, dict):
        raise ValueError("K 线 JSON 必须是对象")
    if "leverage" in payload or "live_capital_usdt" in payload:
        raise ValueError("不接受 leverage 或 live_capital_usdt；纸面路径没有这些设置")
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        raise ValueError("JSON 需要 symbols 对象")
    frames: dict[str, pd.DataFrame] = {}
    for symbol, body in symbols.items():
        if isinstance(body, dict) and "bars" in body:
            launch = body.get("launch_ms")
            bars = body["bars"]
        elif isinstance(body, list):
            launch = None
            bars = body
        else:
            raise ValueError(f"{symbol} 的 K 线格式无法识别")
        frames[str(symbol)] = bars_to_frame(list(bars), launch_ms=launch)
    return frames
