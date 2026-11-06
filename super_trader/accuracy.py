"""准确率优先的选择性分类：拟合规则→校准中性阈值→冻结后的时间外评估。

不使用收益、p值、FDR作为入选门槛。置信评分仅为规则投票分数，不是校准概率。
历史已被项目多次查看，测试结果为后段回顾评估，不冒充未来实盘验证。
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

# 现有实现依赖后续观察到的日期/不完整月份；准确率模式保守排除。
EXCLUDED_RULE_SUFFIXES = {'month_start_long', 'month_end_long', 'pre_holiday_long'}


@dataclass(frozen=True)
class AccuracyConfig:
    fit_fraction: float = .50
    calibration_fraction: float = .25
    min_fit_signals: int = 60
    min_calibration_predictions: int = 30
    top_ks: tuple[int, ...] = (1, 3, 5, 10)
    thresholds: tuple[float, ...] = (.50, .55, .60, .65, .70, .75, .80, .85, .90)

    def __post_init__(self):
        if not (0 < self.fit_fraction < 1 and 0 < self.calibration_fraction < 1
                and self.fit_fraction + self.calibration_fraction < 1):
            raise ValueError('时间切分比例无效')
        if min(self.min_fit_signals, self.min_calibration_predictions) < 1:
            raise ValueError('最小样本数必须为正数')
        if not self.top_ks or any(k < 1 for k in self.top_ks):
            raise ValueError('top_ks必须为正整数')
        if not self.thresholds or any(not .5 <= t <= 1 for t in self.thresholds):
            raise ValueError('中性阈值必须在[0.5,1]之间')


def labeled_samples(p: PreparedSymbol, horizon: int):
    """保留收平标签0，不能只删除预测难命中的收平样本。"""
    idx = np.arange(p.start_idx, max(p.start_idx, p.n - horizon))
    close = p.fs['close']
    ret = close[idx + horizon] / close[idx] - 1
    good = np.isfinite(ret) & ~p.art_strict[idx]
    good &= p.art_cum[idx + horizon + 1] == p.art_cum[idx + 1]
    idx, ret = idx[good], ret[good]
    return idx, np.sign(ret).astype(np.int8)


def split_masks(idx, horizon: int, fit_end: int, calibration_end: int):
    return (
        idx + horizon < fit_end,
        (idx >= fit_end) & (idx + horizon < calibration_end),
        idx >= calibration_end,
    )


def summarize(prediction, actual, baseline_direction: int = 1):
    prediction, actual = np.asarray(prediction), np.asarray(actual)
    active = prediction != 0
    n = int(active.sum())
    eligible = int(actual.size)
    correct = int((prediction[active] == actual[active]).sum())
    up_acc = float(np.mean(actual == 1)) if eligible else None
    dn_acc = float(np.mean(actual == -1)) if eligible else None
    return {
        'eligible': eligible, 'predictions': n, 'correct': correct,
        'accuracy': correct / n if n else None,
        'coverage': n / eligible if eligible else 0.0,
        'up_predictions': int((prediction == 1).sum()),
        'down_predictions': int((prediction == -1).sum()),
        'neutral': eligible - n, 'flat_actual': int((actual == 0).sum()),
        'always_up_accuracy': up_acc, 'always_down_accuracy': dn_acc,
        'majority_baseline': float(np.mean(actual == baseline_direction)) if eligible else None,
        'selected_baseline': float(np.mean(actual[active] == baseline_direction)) if n else None,
    }


def fit_experts(signals: dict, actual, mask, config: AccuracyConfig):
    experts = []
    for key in sorted(signals):
        if key.split('__')[-1] in EXCLUDED_RULE_SUFFIXES:
            continue
        s = np.asarray(signals[key])
        m = mask & (s != 0)
        n = int(m.sum())
        if n < config.min_fit_signals:
            continue
        correct = int((s[m] == actual[m]).sum())
        confidence = (correct + 1) / (n + 2)  # 小样本平滑，只是评分
        if confidence > .5:
            experts.append({'rule': key, 'confidence': confidence, 'n_fit': n})
    return sorted(experts, key=lambda x: (-x['confidence'], -x['n_fit'], x['rule']))


def vote_score(signals, experts, size):
    numerator, denominator = np.zeros(size), np.zeros(size)
    for e in experts:
        s = np.asarray(signals[e['rule']])
        active = s != 0
        weight = 2 * e['confidence'] - 1
        # 每条规则的方向与历史准确评分共同决定看涨评分。
        numerator += active * weight * (.5 + s * (e['confidence'] - .5))
        denominator += active * weight
    score = np.full(size, .5)
    np.divide(numerator, denominator, out=score, where=denominator > 0)
    return score, denominator > 0


def apply_threshold(score, available, threshold):
    pred = np.zeros(len(score), dtype=np.int8)
    pred[available & (score > .5) & (score >= threshold)] = 1
    pred[available & (score < .5) & ((1 - score) >= threshold)] = -1
    return pred


def select_model(signals, actual, fit_mask, calibration_mask, config: AccuracyConfig):
    experts = fit_experts(signals, actual, fit_mask, config)
    best, best_key = None, None
    used = set()
    for requested_k in config.top_ks:
        chosen = experts[:requested_k]
        if not chosen or len(chosen) in used:
            continue
        used.add(len(chosen))
        score, available = vote_score(signals, chosen, len(actual))
        for threshold in config.thresholds:
            pred = apply_threshold(score, available, threshold)
            result = summarize(pred[calibration_mask], actual[calibration_mask])
            if result['predictions'] < config.min_calibration_predictions:
                continue
            # 主目标为准确率；同分优先更多预测、更少规则、更低阈值。
            key = (result['accuracy'], result['predictions'], -len(chosen), -threshold)
            if best_key is None or key > best_key:
                best_key = key
                best = {'rules': [e['rule'] for e in chosen], 'experts': chosen,
                        'k': len(chosen), 'threshold': threshold}
    return best


def evaluate_accuracy(p: PreparedSymbol, horizon: int, config: AccuracyConfig | None = None):
    config = config or AccuracyConfig()
    idx, actual = labeled_samples(p, horizon)
    span = p.n - p.start_idx
    fit_end = p.start_idx + int(span * config.fit_fraction)
    cal_end = p.start_idx + int(span * (config.fit_fraction + config.calibration_fraction))
    fit, cal, test = split_masks(idx, horizon, fit_end, cal_end)
    signals = {key: values[idx] for key, values in p.sigs.items()}
    model = select_model(signals, actual, fit, cal, config)
    majority = 1 if int((actual[fit] == 1).sum()) >= int((actual[fit] == -1).sum()) else -1
    prediction = np.zeros(len(idx), dtype=np.int8)
    unfiltered_prediction = prediction.copy()
    if model:
        score, available = vote_score(signals, model['experts'], len(idx))
        prediction = apply_threshold(score, available, model['threshold'])
        unfiltered_prediction = apply_threshold(score, available, .5)
        latest_signals = {key: np.asarray([values[-1]]) for key, values in p.sigs.items()}
        last_score, last_available = vote_score(latest_signals, model['experts'], 1)
        latest_direction = int(apply_threshold(last_score, last_available, model['threshold'])[0])
    else:
        last_score, latest_direction = np.asarray([.5]), 0
    # 未知未来价格无需参与今日预测；最新异常bar仍强制中性。
    if p.art_strict[-1]:
        latest_direction = 0
    def dates(m):
        positions = idx[m]
        return (str(p.dates[positions[0]].date()), str(p.dates[positions[-1]].date())) if positions.size else (None, None)
    fs, fe = dates(fit)
    cs, ce = dates(cal)
    ts, te = dates(test)
    row = {
        'code': p.info.code, 'name': p.info.name, 'category': p.info.category,
        'horizon': horizon, 'status': 'selected' if model else 'neutral_no_calibrated_model',
        'model': model, 'fit_start': fs, 'fit_end': fe, 'cal_start': cs, 'cal_end': ce,
        'test_start': ts, 'test_end': te,
        'calibration': summarize(prediction[cal], actual[cal], majority),
        'test': summarize(prediction[test], actual[test], majority),
        'test_without_abstention': summarize(unfiltered_prediction[test], actual[test], majority),
        'latest': {'date': str(p.dates[-1].date()), 'direction': latest_direction,
                   'score': float(max(last_score[0], 1 - last_score[0])) if model else None},
    }
    observations = [
        {'category': p.info.category, 'code': p.info.code, 'name': p.info.name,
         'horizon': horizon, 'date': str(p.dates[i].date()),
         'prediction': int(pred), 'actual': int(label)}
        for i, pred, label in zip(idx[test], prediction[test], actual[test])]
    return row, observations


def aggregate(rows):
    eligible = sum(r['test']['eligible'] for r in rows)
    n = sum(r['test']['predictions'] for r in rows)
    correct = sum(r['test']['correct'] for r in rows)
    unfiltered_n = sum(r.get('test_without_abstention', r['test'])['predictions'] for r in rows)
    unfiltered_correct = sum(r.get('test_without_abstention', r['test'])['correct'] for r in rows)
    return {'unfiltered_predictions': unfiltered_n, 'unfiltered_correct': unfiltered_correct,
            'unfiltered_accuracy': unfiltered_correct / unfiltered_n if unfiltered_n else None,
            'eligible': eligible, 'predictions': n, 'correct': correct,
            'accuracy': correct / n if n else None, 'coverage': n / eligible if eligible else 0.0,
            'symbols': len({(r['category'], r['code']) for r in rows}),
            'models': sum(r['model'] is not None for r in rows)}


def run_accuracy(cfg: Config, pool='mainstream', config: AccuracyConfig | None = None, limit=None):
    from .scan import eval_config
    from .universe import resolve_pool
    import pandas as pd
    config = config or AccuracyConfig()
    cfg.ensure_dirs()
    store = KlineStore(cfg.data_path, cfg.cache_path)
    members = resolve_pool(store, pool)
    if limit is not None:
        members = members[:limit]
    rows, observations = [], []
    for number, (info, _tag) in enumerate(members, 1):
        df = store.load(info)
        if len(df) < cfg.min_bars:
            continue
        p = prepare_symbol(df, info, eval_config(cfg))
        if p is None:
            continue
        for h in cfg.horizons:
            row, obs = evaluate_accuracy(p, h, config)
            rows.append(row)
            observations.extend(obs)
        print(f'[{number}/{len(members)}] {info.category} {info.code} {info.name}', flush=True)
    payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
               'method': 'accuracy_abstention_v1', 'config': asdict(config),
               'pool': pool, 'excluded_rules': sorted(EXCLUDED_RULE_SUFFIXES),
               'totals': aggregate(rows), 'records': rows,
               'by_horizon': [{'horizon': h, **aggregate([r for r in rows if r['horizon'] == h])}
                              for h in cfg.horizons]}
    out = cfg.report_path / 'accuracy'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'accuracy.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8', newline='\n')
    pd.DataFrame(observations).to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    latest = [{k: r[k] for k in ['category', 'code', 'name', 'horizon', 'latest']} for r in rows]
    (out / 'today_signals.json').write_text(json.dumps(latest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8', newline='\n')
    from .accuracy_report import build_accuracy_report
    build_accuracy_report(payload, out)
    print(json.dumps(payload['totals'], ensure_ascii=False))
    return payload
