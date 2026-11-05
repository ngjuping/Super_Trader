"""全局配置：默认值 + config.json 覆盖 + 环境变量覆盖。

设计原则：默认值就能跑（指向本机 A 股日线数据目录），配置文件只用于改路径/门槛。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path(r"D:\A股k线数据\多周期\日线")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- 数据 ----
    data_root: str = str(DEFAULT_DATA_ROOT)
    cache_dir: str = str(PROJECT_ROOT / "data" / "cache")
    report_dir: str = str(PROJECT_ROOT / "reports")
    timeframe: str = "日线"

    # ---- 评估 ----
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    min_bars: int = 1000          # 单标的最少 K 线根数（约 4 年）
    warmup_bars: int = 120        # 特征预热，前 N 根不参与评估
    train_fraction: float = 0.6   # 前 60% 训练，后 40% 留出（按时间切，不重叠）
    min_train_samples: int = 60   # 训练段最少触发次数
    min_holdout_samples: int = 30 # 留出段最少触发次数
    train_p_max: float = 0.10     # 训练段入选 p 值上限（单侧，对基准涨率）
    holdout_p_max: float = 0.05   # 留出段确认 p 值上限
    fdr_q: float = 0.10           # Benjamini-Hochberg 假发现率上限
    overlap_adjust: bool = True   # 用 n/h 折算独立样本量，抵抗重叠窗口带来的 p 值虚高

    # ---- 输出 ----
    scan_name: str = "mainstream_daily"

    # ---- 运行期（不写盘） ----
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def data_path(self) -> Path:
        return Path(self.data_root)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def report_path(self) -> Path:
        return Path(self.report_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """读取 config.json（若存在），再用环境变量覆盖。"""
        cfg = cls()

        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.append(PROJECT_ROOT / "config.json")

        for p in candidates:
            if p and p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                cfg = cfg.merged(raw)
                break

        # 环境变量覆盖（便于 CI / 换机器）
        env_map = {
            "SUPER_TRADER_DATA_ROOT": "data_root",
            "SUPER_TRADER_CACHE_DIR": "cache_dir",
            "SUPER_TRADER_REPORT_DIR": "report_dir",
        }
        updates: dict[str, Any] = {}
        for env, key in env_map.items():
            if os.environ.get(env):
                updates[key] = os.environ[env]
        if updates:
            cfg = cfg.merged(updates)
        return cfg

    def merged(self, raw: dict[str, Any]) -> "Config":
        """返回覆盖了 raw 的新配置；未知键收进 extra，不静默丢弃。"""
        known = {f.name: f for f in fields(self)}
        kwargs: dict[str, Any] = {}
        extra = dict(self.extra)
        for k, v in (raw or {}).items():
            if k in known and k != "extra":
                kwargs[k] = v
            else:
                extra[k] = v
        base = self.to_dict()
        base.pop("extra", None)
        base.update(kwargs)
        base["extra"] = extra
        return Config(**base)

    def ensure_dirs(self) -> None:
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.report_path.mkdir(parents=True, exist_ok=True)
