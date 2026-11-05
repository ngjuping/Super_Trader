"""数据层：A 股日线 CSV → 标准化 DataFrame（带本地 parquet 缓存）。

数据源约定（见 D:\\A股k线数据\\_元数据\\说明.txt）：
  D:\\A股k线数据\\多周期\\日线\\{个股|ETF|指数}\\代码_名称.csv
  date,open,close,high,low,volume,amount,amplitude,pct_chg,change,turnover
  * 不复权原始行情，UTF-8 with BOM，按时间升序。
  * 指数为价格指数，无分红除权；个股/ETF 为不复权，除权日会出现价格跳变，
    由 `artifact_mask` 识别并在评估中剔除，避免把分红缺口当成下跌。

本模块只做「读入 + 校验 + 标准化 + 缓存」，不做任何特征计算。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ("个股", "ETF", "指数")

# 涨跌停幅度（A 股现行规则，用于识别异常跳空与「买不进/卖不出」）
_LIMIT_BY_PREFIX = {
    "688": 0.20,  # 科创板
    "689": 0.20,
    "300": 0.20,  # 创业板
    "301": 0.20,
    "83": 0.30,   # 北交所
    "87": 0.30,
    "88": 0.30,
    "43": 0.30,
    "92": 0.30,
}

REQUIRED_COLUMNS = (
    "date", "open", "high", "low", "close",
    "volume", "amount", "pct_chg", "turnover",
)


@dataclass(frozen=True)
class SymbolInfo:
    """一个标的的静态信息。"""

    code: str
    name: str
    category: str
    path: str
    board: str
    price_limit: float  # 0 表示无涨跌停（指数）

    @property
    def key(self) -> str:
        return f"{self.code}_{self.name}"

    @property
    def is_stock(self) -> bool:
        return self.category == "个股"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "board": self.board,
            "price_limit": self.price_limit,
        }


def classify_board(code: str, category: str) -> tuple[str, float]:
    """返回 (板块名, 涨跌停幅度)。"""
    if category == "指数":
        return "指数", 0.0
    if category == "ETF":
        return "ETF", 0.10
    for prefix, limit in _LIMIT_BY_PREFIX.items():
        if code.startswith(prefix):
            if limit == 0.30:
                return "北交所", limit
            if code.startswith("688") or code.startswith("689"):
                return "科创板", limit
            return "创业板", limit
    return "主板", 0.10


class KlineStore:
    """目录扫描 + 单标的读取 + 缓存。"""

    def __init__(self, root: str | Path, cache_dir: str | Path | None = None):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog: pd.DataFrame | None = None

    # ------------------------------------------------------------------ 目录
    def catalog(self, refresh: bool = False) -> pd.DataFrame:
        """全部标的的清单（只读文件名，不解析内容，很快）。"""
        if self._catalog is not None and not refresh:
            return self._catalog

        rows: list[dict] = []
        for cat in CATEGORIES:
            d = self.root / cat
            if not d.exists():
                continue
            for p in sorted(d.glob("*.csv")):
                stem = p.stem
                if "_" not in stem:
                    continue
                code, name = stem.split("_", 1)
                board, limit = classify_board(code, cat)
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "category": cat,
                        "path": str(p),
                        "board": board,
                        "price_limit": limit,
                        "size": p.stat().st_size,
                    }
                )
        self._catalog = pd.DataFrame(rows)
        return self._catalog

    def find(self, code: str, category: str | None = None) -> SymbolInfo | None:
        cat = self.catalog()
        hit = cat[cat["code"] == code]
        if category:
            hit = hit[hit["category"] == category]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return SymbolInfo(
            code=r["code"], name=r["name"], category=r["category"],
            path=r["path"], board=r["board"], price_limit=float(r["price_limit"]),
        )

    def resolve(self, code_or_key: str, category: str | None = None) -> SymbolInfo | None:
        """支持 6 位代码 / 代码_名称 两种写法。"""
        tok = code_or_key.split("_")[0]
        return self.find(tok, category)

    # ------------------------------------------------------------------ 读取
    def load(self, info: SymbolInfo | str, use_cache: bool = True) -> pd.DataFrame:
        if isinstance(info, str):
            found = self.resolve(info)
            if found is None:
                raise FileNotFoundError(f"标的不存在: {info}")
            info = found
        src = Path(info.path)
        if use_cache and self.cache_dir is not None:
            cached = self._load_cache(info, src)
            if cached is not None:
                return cached
        df = self._read_csv(src, info)
        if use_cache and self.cache_dir is not None:
            self._write_cache(info, src, df)
        return df

    @staticmethod
    def _read_csv(src: Path, info: SymbolInfo) -> pd.DataFrame:
        df = pd.read_csv(src, encoding="utf-8-sig")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{src.name} 缺列: {missing}")
        df = df[list(REQUIRED_COLUMNS)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in REQUIRED_COLUMNS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        df = df[df["close"] > 0]
        df = df[df["high"] >= df["low"]]
        df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
        df = df.reset_index(drop=True)
        df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
        df["amount"] = df["amount"].fillna(0.0).clip(lower=0.0)
        df.attrs["symbol"] = info.as_dict()
        return df

    # ------------------------------------------------------------------ 缓存
    def _cache_paths(self, info: SymbolInfo) -> tuple[Path, Path]:
        base = self.cache_dir / info.category  # type: ignore[operator]
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{info.code}_{info.name}"
        return base / f"{stem}.parquet", base / f"{stem}.meta.json"

    def _load_cache(self, info: SymbolInfo, src: Path) -> pd.DataFrame | None:
        pq, meta_p = self._cache_paths(info)
        if not pq.exists() or not meta_p.exists():
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        st = src.stat()
        if meta.get("mtime") != st.st_mtime or meta.get("size") != st.st_size:
            return None
        try:
            df = pd.read_parquet(pq)
        except (OSError, ValueError):
            return None
        if "date" not in df.columns or df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.attrs["symbol"] = info.as_dict()
        return df

    def _write_cache(self, info: SymbolInfo, src: Path, df: pd.DataFrame) -> None:
        pq, meta_p = self._cache_paths(info)
        st = src.stat()
        try:
            df.to_parquet(pq, index=False)
            meta_p.write_text(
                json.dumps({"mtime": st.st_mtime, "size": st.st_size}, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, ValueError):
            # 缓存失败不影响主流程
            pass


# ---------------------------------------------------------------------- 工具
def artifact_mask(df: pd.DataFrame, price_limit: float) -> np.ndarray:
    """识别「不可能由正常交易产生」的价格跳变（除权除息、送转、复牌跳空等）。

    逻辑：主板上单日涨跌幅不可能超过 ±10%，若 |open/prev_close-1| 或
    |close/prev_close-1| 超过 涨跌停+1.2 个百分点，则视为数据层面的价格重塑事件
    （不复权数据里的分红/送转缺口最典型）。这些 bar 参与特征计算会污染动量类特征，
    含这些 bar 的前瞻窗口也不能用于统计方向命中率。
    """
    prev_close = df["close"].shift(1).to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=bool)
    if price_limit <= 0 or len(df) == 0:
        return out
    thr = price_limit + 0.012
    with np.errstate(invalid="ignore", divide="ignore"):
        jump_o = np.abs(o / prev_close - 1.0)
        jump_c = np.abs(c / prev_close - 1.0)
    out[1:] = (jump_o[1:] > thr) | (jump_c[1:] > thr)
    return out


def suspension_mask(df: pd.DataFrame, max_gap_days: int = 15) -> np.ndarray:
    """相邻 K 线间隔超过 max_gap_days 个自然日，视为停牌/长假后的第一根。"""
    dates = pd.to_datetime(df["date"])
    gap = dates.diff().dt.days.to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=bool)
    out[1:] = np.nan_to_num(gap[1:], nan=0.0) > max_gap_days
    return out


def quality_report(df: pd.DataFrame, info: SymbolInfo) -> dict:
    """单标的体检：行数、时间范围、重复、缺口、异常跳变。"""
    dates = pd.to_datetime(df["date"])
    art = artifact_mask(df, info.price_limit)
    susp = suspension_mask(df)
    return {
        "code": info.code,
        "name": info.name,
        "category": info.category,
        "bars": int(len(df)),
        "start": str(dates.iloc[0].date()) if len(df) else None,
        "end": str(dates.iloc[-1].date()) if len(df) else None,
        "artifact_bars": int(art.sum()),
        "suspension_gaps": int(susp.sum()),
        "zero_volume_bars": int((df["volume"] <= 0).sum()),
    }
