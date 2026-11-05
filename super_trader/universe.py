"""主流标的池：先做「大、久、有代表性」的标的，再做全市场。

为什么先主流
------------
1) 流动性好、滑点小，规律才有可执行意义；
2) 历史长，才有足够的独立样本来做显著性检验；
3) 指数不被个股事件（停牌、并购、退市）污染，适合验证「规律本身是否存在」。

池子按用途分三层：
  宽基指数  → 判断规律在「市场层面」是否成立（无个股噪声、无分红干扰）
  行业指数  → 判断规律是否有行业适用性（同一规则在不同行业的表现差异）
  ETF/龙头股 → 判断规律能否落到可交易标的（含个股除权噪声，单独标注）
"""
from __future__ import annotations

from dataclasses import dataclass

ROCKET = "🚀"


@dataclass(frozen=True)
class PoolMember:
    code: str
    tag: str          # 宽基 / 行业 / ETF / 龙头


WHY = {
    "宽基": "市场整体方向，无个股噪声",
    "行业": "行业层面的方向性",
    "ETF": "可交易、T+0 不可但成本低",
    "龙头": "可交易个股，需注意除权",
}

# ------------------------------------------------------------------ 手工主流池
MAINSTREAM: tuple[PoolMember, ...] = (
    # ---- 宽基指数 ----
    PoolMember("000001", "宽基"),   # 上证指数
    PoolMember("399001", "宽基"),   # 深证成指
    PoolMember("399006", "宽基"),   # 创业板指
    PoolMember("000688", "宽基"),   # 科创50
    PoolMember("000300", "宽基"),   # 沪深300
    PoolMember("000905", "宽基"),   # 中证500
    PoolMember("000852", "宽基"),   # 中证1000
    PoolMember("000016", "宽基"),   # 上证50
    PoolMember("399303", "宽基"),   # 国证2000
    PoolMember("399005", "宽基"),   # 中小100
    # ---- 行业指数 ----
    PoolMember("399975", "行业"),   # 证券公司
    PoolMember("399986", "行业"),   # 中证银行
    PoolMember("399997", "行业"),   # 中证白酒
    PoolMember("399396", "行业"),   # 国证食品
    PoolMember("399417", "行业"),   # 新能源车
    PoolMember("399967", "行业"),   # 中证军工
    PoolMember("000819", "行业"),   # 有色金属
    PoolMember("399998", "行业"),   # 中证煤炭
    PoolMember("000813", "行业"),   # 细分化工
    PoolMember("931775", "行业"),   # 房地产
    PoolMember("000949", "行业"),   # 中证农业
    PoolMember("399440", "行业"),   # 国证钢铁
    PoolMember("931079", "行业"),   # 5G通信
    PoolMember("000922", "行业"),   # 中证红利
    # ---- 主流 ETF ----
    PoolMember("510300", "ETF"),    # 沪深300ETF
    PoolMember("510500", "ETF"),    # 中证500ETF
    PoolMember("510050", "ETF"),    # 上证50ETF
    PoolMember("159915", "ETF"),    # 创业板ETF
    PoolMember("588000", "ETF"),    # 科创50ETF
    PoolMember("512100", "ETF"),    # 中证1000ETF
    PoolMember("512880", "ETF"),    # 证券ETF
    PoolMember("518880", "ETF"),    # 黄金ETF
    PoolMember("513050", "ETF"),    # 中概互联网ETF
    PoolMember("512690", "ETF"),    # 酒ETF
    PoolMember("512480", "ETF"),    # 半导体ETF
    PoolMember("159949", "ETF"),    # 创业板50ETF
    PoolMember("512660", "ETF"),    # 军工ETF
    PoolMember("515790", "ETF"),    # 光伏ETF
    # ---- 龙头个股 ----
    PoolMember("600519", "龙头"),   # 贵州茅台
    PoolMember("300750", "龙头"),   # 宁德时代
    PoolMember("601318", "龙头"),   # 中国平安
    PoolMember("600036", "龙头"),   # 招商银行
    PoolMember("000858", "龙头"),   # 五粮液
    PoolMember("002594", "龙头"),   # 比亚迪
    PoolMember("601899", "龙头"),   # 紫金矿业
    PoolMember("600900", "龙头"),   # 长江电力
    PoolMember("601012", "龙头"),   # 隆基绿能
    PoolMember("000333", "龙头"),   # 美的集团
    PoolMember("600030", "龙头"),   # 中信证券
    PoolMember("601088", "龙头"),   # 中国神华
    PoolMember("600276", "龙头"),   # 恒瑞医药
    PoolMember("000651", "龙头"),   # 格力电器
    PoolMember("002415", "龙头"),   # 海康威视
    PoolMember("601166", "龙头"),   # 兴业银行
    PoolMember("600887", "龙头"),   # 伊利股份
    PoolMember("601988", "龙头"),   # 中国银行
    PoolMember("000001", "龙头"),   # 平安银行（与上证指数同代码，靠 category 区分）
    PoolMember("300059", "龙头"),   # 东方财富
    PoolMember("002304", "龙头"),   # 洋河股份
    PoolMember("600028", "龙头"),   # 中国石化
    PoolMember("601857", "龙头"),   # 中国石油
    PoolMember("601668", "龙头"),   # 中国建筑
)

POOLS = {
    "mainstream": MAINSTREAM,
    "宽基": tuple(m for m in MAINSTREAM if m.tag == "宽基"),
    "行业": tuple(m for m in MAINSTREAM if m.tag == "行业"),
    "ETF": tuple(m for m in MAINSTREAM if m.tag == "ETF"),
    "龙头": tuple(m for m in MAINSTREAM if m.tag == "龙头"),
}


def category_for(member: PoolMember) -> str:
    """主流池里 6 位代码在指数/个股两个目录都可能存在，用 tag 定死分类。"""
    if member.tag in ("宽基", "行业"):
        return "指数"
    if member.tag == "ETF":
        return "ETF"
    return "个股"


def resolve_pool(store, name: str = "mainstream") -> list[tuple]:
    """返回 [(SymbolInfo, tag)]，按池顺序；找不到的标的明确报出而不是静默跳过。"""
    members = POOLS.get(name)
    if members is None:
        raise KeyError(f"未知标的池: {name}（可选: {', '.join(POOLS)}）")
    out, missing = [], []
    for m in members:
        info = store.find(m.code, category_for(m))
        if info is None:
            missing.append(f"{m.code}({m.tag})")
            continue
        out.append((info, m.tag))
    if missing:
        raise FileNotFoundError(f"以下标的数据缺失: {', '.join(missing)}")
    return out


def auto_liquid(store, top: int = 40, min_amount: float = 3e8) -> list[tuple]:
    """按最新成交额挑「最有代表性、真的有人在交易」的标的（需要元数据清单）。"""
    meta = store.root.parent.parent / "_元数据" / "标的清单.csv"
    if not meta.exists():
        return resolve_pool(store, "mainstream")
    import pandas as pd

    df = pd.read_csv(meta, dtype={"代码": str})
    df = df[df["成交额"] >= min_amount]
    df = df[df["分类"].isin(["指数", "ETF"])]
    df = df.sort_values("成交额", ascending=False).head(top * 3)
    out, seen = [], set()
    for _, row in df.iterrows():
        cat = row["分类"]
        info = store.find(row["代码"], cat)
        if info is None:
            continue
        if info.key in seen:
            continue
        seen.add(info.key)
        tag = "宽基" if cat == "指数" else "ETF"
        out.append((info, tag))
        if len(out) >= top:
            break
    return out
