'use strict';

// Run from the repository root: node --test tests/workbench_state.test.cjs
// Exercise production helpers, not a copied implementation. Only the snapshot
// initializer and the DOM/event bootstrap are replaced in the in-memory source.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const APP_PATH = path.resolve(__dirname, '../web/workbench/app.js');
const STORAGE_KEY = 'super-trader.workbench.v3';
const DATA_INITIALIZER = "  const D = JSON.parse(document.getElementById('workbench-data').textContent);";
const EVENT_BOOTSTRAP = "  document.addEventListener('click',";
const plain = value => JSON.parse(JSON.stringify(value));

function tinySnapshot() {
  const row = (code, name, horizon, accuracy, coverage, direction) => ({
    category: '个股', code, name, horizon, rules: ['fixture_rule'],
    weighted_accuracy: accuracy, plain_accuracy: accuracy,
    predictions: Math.round(1000 * coverage), eligible: 1000, coverage,
    recent_accuracy: accuracy, latest_direction: direction, base_up_rate: 0.51,
  });
  return {
    snapshot: '2026-11-05', generated_at: 'fixture-only',
    totals: { weighted_accuracy: 0.7 },
    mainstream: ['个股|601088', '个股|600900', '个股|000858', '个股|300750'],
    records: [
      row('601088', '中国神华', 1, 0.72, 0.35, 1),
      row('600900', '长江电力', 3, 0.68, 0.10, 1),
      row('000858', '五粮液', 5, 0.80, 0.099, 0),
      row('300750', '宁德时代', 10, 0.66, 0.30, -1),
    ],
  };
}

function loadWorkbench({ saved, rawSaved, snapshot = tinySnapshot() } = {}) {
  let source = fs.readFileSync(APP_PATH, 'utf8');
  assert.equal(source.split(DATA_INITIALIZER).length - 1, 1,
    'The production snapshot initializer changed; update this test adapter explicitly.');
  const bootstrapAt = source.indexOf(EVENT_BOOTSTRAP);
  assert.ok(bootstrapAt > 0, 'The production event bootstrap marker must exist.');
  assert.equal(source.indexOf(EVENT_BOOTSTRAP, bootstrapAt + 1), -1,
    'The production event bootstrap marker must be unambiguous.');
  source = source.slice(0, bootstrapAt).replace(DATA_INITIALIZER,
    '  const D = globalThis.__snapshot;');
  source += `
  globalThis.__helpers = {
    defaultState, validateState, validatedTask, cfgSnapshot, calcVotes,
    computeResearch, effectiveQuantRows, currentAssessment, isStale,
    fixtures: FIXTURES,
    getState: () => state,
    setState: value => { state = value; },
    getStartupWarning: () => startupWarning,
  };
})();
`;
  const storage = new Map();
  if (rawSaved !== undefined) storage.set(STORAGE_KEY, rawSaved);
  else if (saved !== undefined) storage.set(STORAGE_KEY, JSON.stringify(saved));
  const context = vm.createContext({
    __snapshot: plain(snapshot),
    localStorage: {
      getItem: key => storage.has(key) ? storage.get(key) : null,
      setItem: (key, value) => storage.set(key, String(value)),
    },
    // Deliberately no document, window, timers, fetch, or DOM substitute.
  });
  vm.runInContext(source, context, { filename: APP_PATH, timeout: 1000 });
  return { api: context.__helpers, context, storage };
}

function freezeDeep(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) freezeDeep(child);
  }
  return value;
}

function completedTask(api, id = 'SIM-FIXTURE') {
  const config = api.cfgSnapshot();
  return {
    id, title: '固定快照模拟研究', status: 'completed',
    created: '2026/11/7 09:00:00', finished: '2026/11/7 09:00:03',
    config, result: api.computeResearch(config),
  };
}

test('pure helpers load without DOM, network, timers, or report files', () => {
  const { api, context } = loadWorkbench();
  assert.equal(vm.runInContext('typeof document', context), 'undefined');
  assert.equal(vm.runInContext('typeof window', context), 'undefined');
  assert.equal(vm.runInContext('typeof fetch', context), 'undefined');
  assert.equal(api.getStartupWarning(), '');
  assert.equal(api.getState().experts.length, 7);
});

test('persisted configuration survives JSON/localStorage startup roundtrip', () => {
  const { api } = loadWorkbench();
  const state = plain(api.defaultState());
  state.revision = 9;
  state.lead = 'pa';
  state.experts[0].name = '价值研究员';
  state.experts[0].desc = '研究简介'.repeat(750); // 3,000 characters must not become 2,000.
  state.experts[0].basis = '核实来源\n检查反例\n证据不足则等待';
  state.experts[0].flaw = '风险边界'.repeat(750);
  state.experts[0].weight = 1.7;
  state.experts[0].params.minDividend = 4.5;
  state.experts[0].params.maxPE = 18;
  state.experts[4].enabled = false;
  state.team.mode = 'debate';
  state.team.threshold = 71;
  state.research.pool = 'watchlist';
  state.research.minCoverage = 15;
  state.watchlist = ['个股|601088'];
  state.messages = [{ role: 'user', text: '保留这条本地对话', time: '09:00' }];
  state.activity = [{ text: '保存本机配置', time: '2026/11/7 09:00' }];
  const restored = loadWorkbench({ saved: state }).api;
  assert.equal(restored.getStartupWarning(), '');
  assert.deepEqual(plain(restored.getState()), state);
});

const invalidParams = [
  ['numeric string', 'maxPE', '25'],
  ['non-numeric string', 'maxPE', 'not-a-number'],
  ['below PE minimum', 'maxPE', 0],
  ['above PE maximum', 'maxPE', 201],
  ['invalid PB', 'maxPB', 0],
  ['negative dividend', 'minDividend', -1],
  ['excessive dividend', 'minDividend', 21],
  ['out-of-range ROE', 'minROE', 101],
  ['fractional breakout', 'breakout', 20.5],
  ['excessive breakout', 'breakout', 121],
  ['insufficient risk/reward', 'rr', 0.4],
  ['insufficient accuracy', 'minAccuracy', 49],
  ['unknown strictness', 'strictness', 'wild'],
  ['NaN', 'rr', Number.NaN],
  ['Infinity', 'maxPE', Number.POSITIVE_INFINITY],
  ['null', 'breakout', null],
  ['boolean', 'minAccuracy', true],
];
for (const [label, key, value] of invalidParams) {
  test(`invalid expert parameter is rejected: ${label}`, () => {
    const { api } = loadWorkbench();
    const state = api.defaultState();
    state.experts[0].params[key] = value;
    assert.throws(() => api.validateState(state), /专家/);
  });
}

test('all numeric parameter boundaries and allowed strictness values are valid', () => {
  const { api } = loadWorkbench();
  const bounds = {
    minDividend: [0, 20], maxPE: [1, 200], maxPB: [0.1, 50],
    minROE: [-100, 100], breakout: [5, 120], rr: [0.5, 10], minAccuracy: [50, 100],
  };
  for (const side of [0, 1]) {
    for (const strictness of ['strict', 'balanced', 'loose']) {
      const state = api.defaultState();
      for (const [key, range] of Object.entries(bounds)) state.experts[0].params[key] = range[side];
      state.experts[0].params.strictness = strictness;
      const restored = api.validateState(state);
      assert.deepEqual(plain(restored.experts[0].params), plain(state.experts[0].params));
    }
  }
});

test('invalid saved JSON or invalid expert parameters fall back safely', () => {
  const invalidJSON = loadWorkbench({ rawSaved: '{not valid JSON' }).api;
  assert.match(invalidJSON.getStartupWarning(), /默认配置/);
  assert.equal(invalidJSON.getState().experts.length, 7);
  const { api } = loadWorkbench();
  const bad = api.defaultState();
  bad.experts[0].params.rr = 'NaN';
  const invalidConfiguration = loadWorkbench({ saved: bad }).api;
  assert.match(invalidConfiguration.getStartupWarning(), /默认配置/);
  assert.equal(invalidConfiguration.getState().experts[0].params.rr, 2);
});

test('malformed lastResult is dropped without discarding valid configuration', () => {
  const { api } = loadWorkbench();
  const validTask = plain(completedTask(api));
  const malformed = [
    {},
    { config: {}, result: {} },
    { ...validTask, result: {} },
    { ...validTask, result: { ...validTask.result, assessments: null } },
    { ...validTask, result: { ...validTask.result, counts: [4, 3, 4, 2, 1] } },
    { ...validTask, result: { ...validTask.result, codes: ['does-not-exist'] } },
  ];
  for (const lastResult of malformed) {
    const state = api.defaultState();
    state.experts[0].name = '有效配置不应丢失';
    state.lastResult = lastResult;
    const restored = loadWorkbench({ saved: state }).api;
    assert.equal(restored.getStartupWarning(), '');
    assert.equal(restored.getState().lastResult, null);
    assert.equal(restored.getState().experts[0].name, '有效配置不应丢失');
    assert.doesNotThrow(() => restored.currentAssessment(restored.fixtures[0]));
  }
});

for (const kind of ['buffett', 'cigar', 'neutral', 'pa', 'growth', 'board', 'hotmoney']) {
  test(`copied ${kind} strategy retains its vote after config validation`, () => {
    const { api } = loadWorkbench();
    const state = api.defaultState();
    const original = plain(state.experts.find(expert => expert.id === kind));
    const duplicate = { ...plain(original), id: `custom-copy-${kind}`, name: `${original.name}副本` };
    state.experts = [original, duplicate];
    state.lead = original.id;
    const config = api.validateState(state);
    assert.equal(config.experts[1].strategyKind, kind);
    for (const fixture of api.fixtures) {
      const { votes } = api.calcVotes(fixture, config);
      assert.equal(votes.length, 2);
      assert.notEqual(votes[0].id, votes[1].id);
      assert.equal(votes[0].vote, votes[1].vote);
      assert.equal(votes[0].why, votes[1].why);
      assert.equal(votes[0].evaluable, votes[1].evaluable);
    }
  });
}

test('missing tools or decision basis cannot bypass disabled abstention', () => {
  const { api } = loadWorkbench();
  for (const missing of ['tools', 'basis', 'quant-tool']) {
    const state = api.defaultState();
    const expert = state.experts.find(e => e.id === (missing === 'quant-tool' ? 'neutral' : 'buffett'));
    state.experts = [expert];
    state.lead = expert.id;
    state.team.abstain = false;
    state.team.evidence = true;
    if (missing === 'tools') expert.tools = [];
    if (missing === 'basis') expert.basis = '   ';
    if (missing === 'quant-tool') expert.tools = ['kline'];
    const result = api.calcVotes(api.fixtures[0], state);
    assert.equal(result.votes[0].evaluable, false, missing);
    assert.equal(result.votes[0].vote, 0, missing);
    assert.equal(result.share, 0, missing);
    assert.equal(result.admitted, false, missing);
  }
});

test('evidence gate blocks admission even when other support exceeds the threshold', () => {
  const { api } = loadWorkbench();
  const state = api.defaultState();
  state.experts = state.experts.filter(e => ['buffett', 'growth'].includes(e.id));
  state.experts.find(e => e.id === 'growth').tools = [];
  state.team.threshold = 1;
  state.team.abstain = false;
  state.team.evidence = true;
  const result = api.calcVotes(api.fixtures[0], state);
  assert.ok(result.share > state.team.threshold / 100);
  assert.equal(result.admitted, false);
  assert.equal(result.votes.find(v => v.id === 'growth').vote, 0);
});

test('running watchlist snapshot is detached from subsequent workspace changes', () => {
  const { api } = loadWorkbench();
  const state = api.defaultState();
  state.research.pool = 'watchlist';
  state.watchlist = ['个股|601088'];
  api.setState(state);
  const snapshot = api.cfgSnapshot();
  const originalConfig = plain(snapshot);
  const firstResult = plain(api.computeResearch(snapshot));
  assert.equal(firstResult.counts[0], 1);
  assert.deepEqual(plain(snapshot.watchlist), ['个股|601088']);
  assert.notEqual(snapshot.watchlist, state.watchlist);
  state.watchlist.splice(0, 1, '个股|600900');
  state.research.minDividend = 20;
  state.experts[0].enabled = false;
  state.team.threshold = 100;
  freezeDeep(snapshot);
  assert.deepEqual(plain(api.computeResearch(snapshot)), firstResult);
  assert.deepEqual(plain(snapshot), originalConfig);
});

test('minimum coverage clamps to 10% and the boundary is enforced in snapshot rows', () => {
  const { api } = loadWorkbench();
  for (const raw of [-10, 0, 9.9, 10]) {
    const state = api.defaultState();
    state.research.minCoverage = raw;
    state.research.minAccuracy = 50;
    state.research.pool = 'all';
    const restored = api.validateState(state);
    assert.equal(restored.research.minCoverage, 10);
    api.setState(restored);
    const codes = plain(api.effectiveQuantRows().map(row => row.code));
    assert.ok(codes.includes('600900'), 'Exactly 10% coverage remains eligible.');
    assert.ok(!codes.includes('000858'), '9.9% coverage must never pass the hard floor.');
  }
  const high = api.defaultState();
  high.research.minCoverage = 1000;
  assert.equal(api.validateState(high).research.minCoverage, 100);
});

test('completed task survives persisted roundtrip with config, votes and evidence metadata', () => {
  const { api } = loadWorkbench();
  const state = api.defaultState();
  state.revision = 6;
  state.research.pool = 'watchlist';
  state.watchlist = ['个股|601088', '个股|600900'];
  api.setState(state);
  const task = plain(completedTask(api));
  assert.deepEqual(task.result.counts, [2, 2, 2, 2, 1]);
  assert.ok(task.result.assessments.length > 0);
  state.tasks = [task];
  state.lastResult = task;
  const restored = loadWorkbench({ saved: state }).api;
  assert.equal(restored.getStartupWarning(), '');
  assert.equal(restored.getState().tasks.length, 1);
  assert.deepEqual(plain(restored.getState().tasks[0]), task);
  assert.deepEqual(plain(restored.getState().lastResult), task);
  assert.equal(restored.isStale(), false);
  assert.deepEqual(plain(restored.computeResearch(restored.getState().tasks[0].config)), task.result);
});

test('persisted task retains false evidence eligibility rather than inventing eligibility', () => {
  const { api } = loadWorkbench();
  const state = api.defaultState();
  state.experts = state.experts.filter(expert => expert.id === 'buffett');
  state.experts[0].tools = [];
  state.team.abstain = false;
  state.team.evidence = true;
  api.setState(state);
  const task = plain(completedTask(api, 'SIM-NO-EVIDENCE'));
  assert.ok(task.result.assessments.length > 0);
  for (const assessment of task.result.assessments) {
    assert.equal(assessment.votes[0].evaluable, false);
    assert.equal(assessment.votes[0].vote, 0);
    assert.equal(assessment.admitted, false);
  }
  state.tasks = [task];
  state.lastResult = task;
  const restored = loadWorkbench({ saved: state }).api;
  assert.equal(restored.getState().tasks.length, 1);
  assert.deepEqual(plain(restored.getState().tasks[0]), task);
  assert.deepEqual(plain(restored.getState().lastResult), task);
});

test('bad task records are isolated while valid completed and cancelled records survive', () => {
  const { api } = loadWorkbench();
  const state = api.defaultState();
  const valid = plain(completedTask(api));
  const cancelled = {
    id: 'SIM-CANCELLED', title: '已取消任务', status: 'cancelled',
    created: '2026/11/7 09:00:00', finished: '', config: plain(api.cfgSnapshot()),
    result: { counts: [], codes: [], admitted: [], assessments: [] },
  };
  const malformed = plain(valid);
  malformed.id = 'SIM-BROKEN';
  malformed.result.assessments[0].votes[1].id = malformed.result.assessments[0].votes[0].id;
  state.tasks = [{ id: 'SIM-EMPTY', status: 'completed', config: {}, result: {} }, malformed, valid, cancelled];
  const restored = api.validateState(state);
  assert.deepEqual(plain(restored.tasks.map(task => task.id)), ['SIM-FIXTURE', 'SIM-CANCELLED']);
  assert.equal(restored.tasks[1].status, 'cancelled');
  assert.deepEqual(plain(restored.tasks[1].result), cancelled.result);
});
