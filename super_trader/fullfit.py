"""全历史拟合：不分验证/测试集，近期权重高，逐标的挑规律直到准确率过线。

口径（按用户要求）：
* 全部历史样本参与拟合，不做切分，也不留出验证段。
* 样本权重 = 0.5 ** (距今天数 / half_life_days)，默认半衰期一年：
  最近一年权重最高，越久远衰减越快。
* 每个标的在 105 条规则 × 5 个持有期中搜索，用加权准确率排序；
  单条规则不达标就换规则，仍不够就叠加规则做投票集成，直到达标或候选用尽。
* 最终如果集成仍然低于 50%，就把方向整体反向（等价于把规律换成它的反面），
  保证每个标的的加权准确率 ≥ 50%。

必须知道：以上全部是**内样本**结果。全历史拟合意味着没有任何未参与拟合的数据，
所以「准确率」衡量的是拟合优度，不是未来预测能力；反向约定同样是内样本决定的。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .config import Config
from .data import KlineStore
from .evaluate import PreparedSymbol, prepare_symbol
from .rules import RULES


EXCLUDED = {'month_start_long', 'month_end_long', 'pre_holiday_long'}


@dataclass(frozen=True)
class FullFitConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    half_life_days: float = 365.0   # 权重半衰期：一年
    min_signals: int = 80           # 单条规则至少触发多少次才入选
    target_accuracy: float = 0.55   # 达到这个加权准确率就停止加规则
    max_rules: int = 5              # 集成最多几条规则
    candidate_pool: int = 10        # 进入集成的候选规则数
    min_coverage: float = 0.10      # 集成至少覆盖 10% 的样本，避免只剩个别样本的假高分
    weight_floor: float = 0.10      # 权重下限：老数据仍计入，避免近期两三条样本垄断加权准确率
    min_eff_n: float = 30.0         # 有效样本量 (Σw)²/Σw² 下限

    def __post_init__(self):
        if self.min_signals < 1:
            raise ValueError('min_signals必须≥1')
        if not 0 < self.target_accuracy <= 1:
            raise ValueError('target_accuracy必须在(0,1]')
        if self.max_rules < 1 or self.candidate_pool < 1:
            raise ValueError('max_rules/candidate_pool必须≥1')
        if not 0 <= self.weight_floor < 1:
            raise ValueError('weight_floor必须在[0,1)')
        if not 0 <= self.min_coverage < 1:
            raise ValueError('min_coverage必须在[0,1)')
        if self.half_life_days <= 0:
            raise ValueError('half_life_days必须为正')


def _labels(p: PreparedSymbol, horizon: int):
    """全历史样本（保留收平=0），只剔除除权跳变污染的前瞻窗口。"""
    idx = np.arange(p.start_idx, max(p.start_idx, p.n - horizon))
    close = np.asarray(p.fs['close'], dtype=float)
    ret = close[idx + horizon] / close[idx] - 1.0
    keep = np.isfinite(ret) & ~p.art_strict[idx]
    keep &= p.art_cum[idx + horizon + 1] == p.art_cum[idx + 1]
    idx, ret = idx[keep], ret[keep]
    return idx, np.sign(ret).astype(np.int8)


def _weights(p: PreparedSymbol, idx: np.ndarray, half_life_days: float,
             floor: float = 0.10) -> np.ndarray:
    last = p.dates[-1]
    age_days = np.asarray((last - p.dates[idx]).days, dtype=float)
    decay = np.power(0.5, age_days / max(half_life_days, 1.0))
    return floor + (1.0 - floor) * decay


def _eff_n(w: np.ndarray) -> float:
    """有效样本量：权重越不均衡，独立信息越少。"""
    total = float(np.sum(w))
    if total <= 0:
        return 0.0
    return total * total / float(np.sum(w * w))


def _wacc(pred: np.ndarray, actual: np.ndarray, w: np.ndarray) -> tuple[float, int]:
    """加权准确率；只统计发出预测的样本。返回 (准确率, 预测次数)。"""
    m = pred != 0
    n = int(m.sum())
    if n == 0:
        return float('nan'), 0
    hit = (pred[m] == actual[m]).astype(float)
    return float(np.average(hit, weights=w[m])), n


def _vote(stack: np.ndarray, invert: bool = False) -> np.ndarray:
    """等权投票：净票数>0 看涨，<0 看跌，0 弃权。"""
    score = stack.astype(np.int32).sum(axis=0)
    out = np.zeros(stack.shape[1], dtype=np.int8)
    out[score > 0] = 1
    out[score < 0] = -1
    return -out if invert else out


def fit_symbol(p: PreparedSymbol, config: FullFitConfig) -> dict:
    best = None
    for horizon in config.horizons:
        idx, actual = _labels(p, horizon)
        if idx.size < config.min_signals:
            continue
        w = _weights(p, idx, config.half_life_days, config.weight_floor)
        signals = {k: np.asarray(v, dtype=np.int8)[idx] for k, v in p.sigs.items()}
        scored = []
        for key, s in signals.items():
            if key.split('__')[-1] in EXCLUDED:
                continue
            acc, n = _wacc(s, actual, w)
            if n >= config.min_signals and np.isfinite(acc):
                if _eff_n(w[s != 0]) < config.min_eff_n:
                    continue
                scored.append((acc, n, key))
        if not scored:
            continue
        scored.sort(key=lambda x: (-x[0], -x[1]))
        pool = [k for _, _, k in scored[:config.candidate_pool]]

        # 贪心叠加：先取最好的一条，不达标就换/叠加下一条
        floor = max(config.min_signals, int(config.min_coverage * idx.size))
        chosen: list[str] = []
        cur_acc, cur_n = None, 0
        for key in pool:
            trial = chosen + [key]
            pred = _vote(np.vstack([signals[k] for k in trial]))
            acc, n = _wacc(pred, actual, w)
            if n < floor or _eff_n(w[pred != 0]) < config.min_eff_n:
                continue
            if not chosen or acc > (cur_acc or 0):
                chosen, cur_acc, cur_n = trial, acc, n
            if cur_acc is not None and cur_acc >= config.target_accuracy:
                break
            if len(chosen) >= config.max_rules:
                break
        if not chosen:
            continue

        pred = _vote(np.vstack([signals[k] for k in chosen]))
        acc, n = _wacc(pred, actual, w)
        inverted = acc < 0.5
        if inverted:
            pred = _vote(np.vstack([signals[k] for k in chosen]), invert=True)
            acc, n = _wacc(pred, actual, w)

        # 最近一年子集（仍然在内样本里，只用于展示近期表现）
        recent = (p.dates[-1] - p.dates[idx]).days <= 365
        recent_acc, recent_n = (float('nan'), 0)
        if recent.any():
            m = (pred != 0) & recent
            if m.any():
                recent_acc = float(np.mean(pred[m] == actual[m]))
                recent_n = int(m.sum())

        plain_acc = (float(np.mean(pred[pred != 0] == actual[pred != 0]))
                     if n else float('nan'))
        row = {
            'category': p.info.category, 'code': p.info.code, 'name': p.info.name,
            'horizon': horizon, 'rules': chosen, 'inverted': bool(inverted),
            'weighted_accuracy': acc, 'plain_accuracy': plain_acc,
            'eff_n': _eff_n(w[pred != 0]),
            'predictions': n, 'eligible': int(idx.size),
            'coverage': n / float(idx.size) if idx.size else None,
            'up': int((pred == 1).sum()), 'down': int((pred == -1).sum()),
            'flat_actual': int((actual == 0).sum()),
            'recent_accuracy': recent_acc, 'recent_predictions': recent_n,
            'start': str(p.dates[idx[0]].date()), 'end': str(p.dates[idx[-1]].date()),
            'unfiltered_accuracy': (float(np.mean(pred == actual)) if idx.size else None),
            'always_up_accuracy': float(np.mean(actual == 1)) if idx.size else None,
        }
        # 每个标的保留加权准确率最高的持有期
        if best is None or (np.isfinite(acc) and acc > (best['weighted_accuracy'] or 0)):
            best = row
    return best


def run_fullfit(cfg: Config | None = None, pool: str = 'mainstream',
                config: FullFitConfig | None = None, limit: int | None = None) -> dict:
    from .scan import eval_config
    from .universe import resolve_pool

    cfg = cfg or Config.load()
    config = config or FullFitConfig()
    cfg.ensure_dirs()
    store = KlineStore(cfg.data_path, cfg.cache_path)
    members = resolve_pool(store, pool)
    if limit is not None:
        members = members[:limit]
    rows = []
    for i, (info, _tag) in enumerate(members, 1):
        df = store.load(info)
        if len(df) < cfg.min_bars:
            continue
        p = prepare_symbol(df, info, eval_config(cfg))
        if p is None:
            continue
        row = fit_symbol(p, config)
        if row:
            rows.append(row)
        print(f'[{i}/{len(members)}] {info.code} {info.name}'
              + (f"  {row['weighted_accuracy']*100:.2f}% n={row['predictions']}" if row else '  —'),
              flush=True)

    rows.sort(key=lambda r: (-(r['weighted_accuracy'] or 0), -r['predictions']))
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
    total_n = sum(r['predictions'] for r in rows)
    total_hit = sum(r['plain_accuracy'] * r['predictions'] for r in rows
                    if np.isfinite(r['plain_accuracy'] or float('nan')))
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'method': 'fullfit_recency_weighted_v1',
        'config': asdict(config), 'pool': pool, 'excluded_rules': sorted(EXCLUDED),
        'scope': 'in_sample_full_history_no_split',
        'totals': {
            'symbols': len(rows), 'predictions': total_n,
            'weighted_accuracy': (float(np.mean([r['weighted_accuracy'] for r in rows
                                                 if np.isfinite(r['weighted_accuracy'])]))
                                  if rows else None),
            'plain_accuracy': (total_hit / total_n) if total_n else None,
            'median_accuracy': float(np.median([r['weighted_accuracy'] for r in rows
                                                if np.isfinite(r['weighted_accuracy'])])) if rows else None,
            'inverted': int(sum(r['inverted'] for r in rows)),
            'below_50': int(sum(1 for r in rows if (r['weighted_accuracy'] or 0) < .5)),
        },
        'records': rows,
    }
    out = cfg.report_path / 'fullfit'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fullfit.json').write_text(
        json.dumps(_clean(payload), ensure_ascii=False, allow_nan=False, indent=2),
        encoding='utf-8', newline='\n')
    return payload


def _clean(obj):
    """NaN/Inf 不能写进 JSON，统一转成 None。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
