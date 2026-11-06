from __future__ import annotations

import json
import numpy as np
import pytest

from super_trader.accuracy import (
    AccuracyConfig, aggregate, apply_threshold, evaluate_accuracy, labeled_samples,
    select_model, split_masks, summarize, vote_score,
)
from super_trader.data import SymbolInfo
from super_trader.evaluate import EvalConfig, prepare_symbol
from tests.conftest import make_ohlcv


def test_neutral_is_not_counted_correct_and_flats_are_misses():
    m = summarize([1, -1, 0, 1], [1, 0, -1, -1])
    assert m['predictions'] == 3 and m['correct'] == 1
    assert m['accuracy'] == pytest.approx(1/3)
    assert m['coverage'] == .75 and m['flat_actual'] == 1
    assert summarize([0, 0], [1, -1])['accuracy'] is None


def test_raise_threshold_abstains_not_flips_direction():
    score = np.array([.9, .6, .5, .3, .1])
    available = np.ones(5, dtype=bool)
    a = apply_threshold(score, available, .5)
    b = apply_threshold(score, available, .8)
    assert a.tolist() == [1, 1, 0, -1, -1]
    assert b.tolist() == [1, 0, 0, 0, -1]
    assert np.all(b[b != 0] == a[b != 0])


def test_no_active_rules_is_neutral_even_at_half_threshold():
    score, available = vote_score({'r': np.zeros(3)}, [{'rule':'r','confidence':.9}], 3)
    assert not apply_threshold(score, available, .5).any()


def test_conflicting_equal_experts_abstain():
    signals = {'a':np.ones(4), 'b':-np.ones(4)}
    experts = [{'rule':'a','confidence':.8}, {'rule':'b','confidence':.8}]
    score, available = vote_score(signals, experts, 4)
    np.testing.assert_allclose(score, .5)
    assert not apply_threshold(score, available, .5).any()


def test_purged_splits_use_label_endpoint():
    idx = np.array([0, 10, 45, 49, 50, 60, 70, 74, 75, 90])
    fit, cal, test = split_masks(idx, 5, 50, 75)
    assert np.all(idx[fit] + 5 < 50)
    assert np.all(idx[cal] >= 50) and np.all(idx[cal] + 5 < 75)
    assert np.all(idx[test] >= 75)
    assert not np.any(fit & cal) and not np.any(cal & test)


def _selection_problem():
    actual = np.tile(np.array([1, -1]), 300)
    good = actual.copy()
    good[np.arange(600) % 10 == 0] *= -1
    fit, cal = np.arange(600) < 300, (np.arange(600) >= 300) & (np.arange(600) < 450)
    return actual, {'good':good, 'bad':-good}, fit, cal


def test_test_outcomes_do_not_change_model_or_neutral_threshold():
    actual, signals, fit, cal = _selection_problem()
    cfg = AccuracyConfig()
    a = select_model(signals, actual, fit, cal, cfg)
    changed = actual.copy()
    changed[450:] *= -1
    b = select_model(signals, changed, fit, cal, cfg)
    assert a == b and a is not None
    assert a['rules'] == ['good']


def test_insufficient_calibration_predictions_disables_model():
    actual, signals, fit, cal = _selection_problem()
    assert select_model(signals, actual, fit, cal,
                        AccuracyConfig(min_calibration_predictions=151)) is None


def test_exact_calibration_support_boundary():
    actual, signals, fit, _cal = _selection_problem()
    cal = (np.arange(600) >= 300) & (np.arange(600) < 330)
    assert select_model(signals, actual, fit, cal, AccuracyConfig()) is not None
    cal[329] = False
    assert select_model(signals, actual, fit, cal, AccuracyConfig()) is None


def test_latest_artifact_bar_forces_neutral():
    df = make_ohlcv(900, seed=5)
    info = SymbolInfo(code='T', name='test', category='指数', path='x', board='指数', price_limit=0.)
    p = prepare_symbol(df, info, EvalConfig())
    p.art_strict[-1] = True
    row, _ = evaluate_accuracy(p, 1)
    assert row['latest']['direction'] == 0
    assert 'test_without_abstention' in row


def test_flats_are_retained_by_accuracy_labels():
    df = make_ohlcv(900, seed=5)
    info = SymbolInfo(code='T', name='test', category='指数', path='x', board='指数', price_limit=0.)
    # At least one known flat outcome in eligible bars.
    df.loc[300, 'close'] = df.loc[299, 'close']
    p = prepare_symbol(df, info, EvalConfig())
    idx, actual = labeled_samples(p, 1)
    assert actual[np.where(idx == 299)[0][0]] == 0
    row, observations = evaluate_accuracy(p, 1)
    assert row['test']['eligible'] == len(observations)
    assert row['latest']['direction'] in [-1, 0, 1]
    json.dumps(row, allow_nan=False)


def test_aggregation_is_prediction_weighted_not_mean_accuracy():
    rows = [{'category':'指数','code':str(i),'model':{},'test':test}
            for i,test in enumerate([
                {'eligible':100,'predictions':100,'correct':60},
                {'eligible':100,'predictions':10,'correct':10}])]
    total = aggregate(rows)
    assert total['accuracy'] == pytest.approx(70/110)
    assert total['coverage'] == .55


def test_accuracy_report_handles_missing_predictions(tmp_path):
    from super_trader.accuracy_report import build_accuracy_report
    paths = build_accuracy_report({'totals':{'predictions':0,'eligible':100,'accuracy':None,'coverage':0},
                                   'records':[], 'by_horizon':[]}, tmp_path)
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize('kwargs', [{'min_fit_signals':0}, {'thresholds':(.4,)}, {'fit_fraction':.9}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AccuracyConfig(**kwargs)
