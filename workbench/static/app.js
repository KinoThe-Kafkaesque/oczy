'use strict';

let evidence;
const $ = id => document.getElementById(id);
const primaryConditions = ['diversity', 'control', 'initial', 'no_context', 'chat_diversity', 'text_diversity', 'direct_oracle'];
const names = {
  diversity: 'Learned · varied examples', control: 'Learned · earlier control', initial: 'Untrained state',
  zeroed: 'Zeroed state', no_context: 'No learned state', chat_diversity: 'Retrieved conversation examples',
  text_diversity: 'Retrieved example text', direct_oracle: 'Explicitly supplied rule', complete_table: 'Complete rule table',
  chat_control: 'Conversation · control examples', text_control: 'Text · control examples',
  zlib_diversity: 'Compressed example text', zlib_control: 'Compressed control text', joint_restored: 'Earlier joint learner'
};

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function tableRow(parent, values) {
  const row = node('tr');
  for (const value of values) row.append(value instanceof Node ? value : node('td', String(value)));
  parent.append(row);
}

function score(value) {
  return node('td', value ? 'Correct' : 'Incorrect', value ? 'pass' : 'fail');
}

function renderTrial(rows, target, mode) {
  $('trial-rows').replaceChildren();
  $('target').textContent = target;
  $('trial-mode').textContent = mode;
  for (const row of rows) {
    const title = node('td', names[row.condition] || row.condition);
    if (row.condition === 'direct_oracle') title.append(node('span', 'Rule selected outside the model', 'dim'));
    tableRow($('trial-rows'), [title, node('td', JSON.stringify(row.generated)), score(row.correct)]);
  }
}

function showSaved() {
  if (!evidence) return;
  const client = $('client').value, word = $('word').value, seed = Number($('seed').value);
  const rows = evidence.rows.filter(row =>
    (evidence.current_study || row.stage === 2) &&
    (!evidence.current_study || primaryConditions.includes(row.condition)) &&
    row.input === word && row.clients?.length === 1 && row.clients[0] === client &&
    (row.seed === seed || row.seed === null));
  const order = ['diversity', 'control', 'joint_restored', 'initial', 'zeroed', 'no_context', 'chat_diversity', 'text_diversity', 'direct_oracle', 'complete_table'];
  const rank = condition => order.includes(condition) ? order.indexOf(condition) : 100;
  rows.sort((a, b) => rank(a.condition) - rank(b.condition));
  if (!rows.length) {
    renderTrial([], '—', 'No saved trial');
    $('trial-status').textContent = 'No registered result for this word. Choose a recorded word or run an exploratory live comparison.';
    return;
  }
  renderTrial(rows, rows[0].target, 'Saved · ' + rows[0].split);
  $('trial-status').textContent = 'Recorded output, including whitespace, from ' +
    (evidence.current_study ? 'scoped-diversity DEV-v3.' : 'context-preservation DEV-v2.');
}

function renderCapabilities() {
  const selected = evidence.rows.filter(row => row.split === 'confirmation' &&
    row.condition === (evidence.current_study ? 'diversity' : 'joint_restored'));
  const affix = selected.filter(row => ['amber', 'cobalt'].includes(row.category));
  const neutral = selected.filter(row => ['silver', 'quartz'].includes(row.category));
  const count = rows => `${rows.filter(row => row.correct).length}/${rows.length}`;
  const direct = evidence.language.find(row => row.condition === 'direct_operation' && row.split === 'confirmation');
  const compositionCorrect = evidence.composition.reduce((sum, row) => sum + row.correct, 0);
  const compositionTotal = evidence.composition.reduce((sum, row) => sum + row.total, 0);
  const strictActions = evidence.action_rows.filter(row => row.correct).length;
  const storage = evidence.current_study ? evidence.study.storage.find(row => row.arm === 'diversity') : evidence.compression.at(-1);
  const exampleBytes = storage.example_utf8_bytes ?? storage.active_example_bytes;
  const zlibBytes = storage.zlib_bytes ?? storage.zlib_file_bytes;
  const capabilities = [
    ['Language interface', `${direct.correct}/${direct.total} new-word direct operations; decoder parity ${evidence.decoder_qualified ? 'qualified' : 'not qualified'}.`,
      'The operation is supplied externally; this is not learned selection.'],
    ['Selective application', `${count(affix)} fresh suffix cases; ${count(neutral)} copy cases`,
      evidence.current_study && evidence.study.admission ? 'Acquisition prerequisite passed; preservation is separate.' : 'Not validated across all three seeds.'],
    ['Accumulating knowledge', 'Earlier sequential learner lost all previously correct amber cases after learning cobalt.',
      'The new study trains rules jointly; sequential accumulation remains unproved.'],
    ['Correcting knowledge', evidence.current_study ? evidence.study.preservation_status : 'Blocked by acquisition prerequisite.',
      'Unrelated knowledge must survive the correction.'],
    ['Composition', `Earlier learned composition: ${compositionCorrect}/${compositionTotal}.`,
      'Individual rule prerequisites failed; composition is not isolated.'],
    ['Useful compression', `Numeric state: 28,800 bytes; example text: ${exampleBytes} bytes; zlib: ${zlibBytes} bytes.`,
      'Behavior in less storage than examples is not demonstrated for learned state.'],
    ['Reliable action', 'Tool calls and real files are scored separately from final text.',
      `Earlier strict end-to-end action success: ${strictActions}/${evidence.action_rows.length}.`]
  ];
  for (const [capability, current, limit] of capabilities) {
    const heading = node('th', capability);
    heading.scope = 'row';
    const cells = [heading];
    for (const [label, value] of [['Current evidence', current], ['Limit', limit]]) {
      const cell = node('td');
      const mobileLabel = node('span', label, 'capability-label');
      mobileLabel.setAttribute('aria-hidden', 'true');
      cell.append(mobileLabel, document.createTextNode(value));
      cells.push(cell);
    }
    tableRow($('capabilities'), cells);
  }
  for (const row of evidence.study.counts || []) {
    tableRow($('counts'), [row.split, row.condition, row.seed ?? '—', row.category, `${row.correct}/${row.total}`]);
  }
}

function renderPairedRegressions(paired) {
  if (!paired.length) return;
  const total = paired.reduce((sum, row) => sum + row.total, 0);
  const before = paired.reduce((sum, row) => sum + row.control_correct, 0);
  const after = paired.reduce((sum, row) => sum + row.diversity_correct, 0);
  $('regression-results').append(node('p', `On the unchanged earlier prompts: varied teaching ${after}/${total}, earlier control ${before}/${total}. These aggregate scores include neutral copying and composition; inspect each context below.`));
  const details = node('details'), wrap = node('div', undefined, 'table-wrap');
  const table = node('table'), head = node('thead'), body = node('tbody'), header = node('tr');
  for (const label of ['Seed', 'Context', 'Earlier control', 'Varied teaching', 'Lost / gained']) header.append(node('th', label));
  head.append(header);
  for (const row of paired) {
    tableRow(body, [row.seed, row.category, `${row.control_correct}/${row.total}`, `${row.diversity_correct}/${row.total}`, `${row.lost.length} / ${row.gained.length}`]);
  }
  table.append(head, body);
  wrap.append(table);
  details.append(node('summary', 'Compare candidate regressions by context and seed'), wrap);
  $('regression-results').append(details);
}

function renderRegression() {
  const regression = evidence.regression;
  if (!regression) return;
  const replay = regression.verdicts || {};
  const exact = Object.keys(replay).length > 0 && Object.values(replay).every(row => row.exact_replay);
  $('regression-status').textContent = exact
    ? 'Historical outputs reproduced exactly. This verifies reproducibility, including earlier failures.'
    : 'At least one historical output changed. Inspect the replay record before relying on prior claims.';
  $('regression-status').className = exact ? 'pass' : 'fail';
  const list = node('ul');
  for (const [name, row] of Object.entries(replay)) {
    list.append(node('li', `${name.replaceAll('_', ' ')}: ${row.rows} rows · ${row.exact_replay ? 'exact replay' : 'changed'}`));
  }
  $('regression-results').append(list);
  if (regression.candidate_summary) $('regression-results').append(node('p', regression.candidate_summary));
  renderPairedRegressions(regression.paired || []);
}

function actionRows() {
  return evidence.candidate_actions || evidence.action_rows;
}

function renderAction() {
  const row = actionRows()[Number($('action-case').value)];
  $('action-verdict').replaceChildren();
  const fields = [['tool_sequence_exact', 'Tool sequence'], ['filesystem_exact', 'Actual files'],
    ['final_exact', 'Final answer'], ['execution_error_free', 'Execution without error']];
  for (const [key, label] of fields) {
    $('action-verdict').append(node('span', `${label}: ${row[key] ? 'correct' : 'failed'}`, row[key] ? 'pass' : 'fail'));
  }
  $('transcript').textContent = JSON.stringify({transcript: row.transcript, final: row.final, error: row.error}, null, 2);
  $('files').textContent = JSON.stringify(row.actual_files, null, 2);
}

async function init() {
  try {
    const response = await fetch('/api/evidence');
    if (!response.ok) throw new Error('Could not load saved evidence. Restart the local workbench.');
    evidence = await response.json();
    $('study-state').textContent = evidence.study_state;
    $('gate-status').textContent = evidence.current_study
      ? (evidence.study.admission ? 'Acquisition gate passed. Inspect correction evidence below.' : 'Acquisition gate failed. Correction remains blocked.')
      : 'The next experiment is running. Earlier evidence remains inspectable.';
    if (evidence.study.execution_recovery) {
      $('execution-info').hidden = false;
      $('recovery-note').textContent = evidence.study.execution_recovery.provenance_limit;
    }
    $('as-of').textContent = 'As of ' + evidence.as_of;
    $('footer-date').textContent = evidence.as_of;
    $('word').value = evidence.current_study ? 'berry' : 'lime';
    $('word').setAttribute('list', 'saved-word-options');
    for (const word of [...new Set(evidence.rows.map(row => row.input))].sort()) {
      const option = node('option');
      option.value = word;
      $('saved-word-options').append(option);
    }
    $('live').disabled = !evidence.live_available;
    if (!evidence.live_available) $('live').title = 'The released states or configured runtime/model are unavailable';
    renderCapabilities();
    renderRegression();
    for (const wrap of document.querySelectorAll('.table-wrap')) wrap.tabIndex = 0;
    actionRows().forEach((row, index) => {
      const option = node('option', `${row.condition} · seed ${row.seed ?? 'none'} · ${row.case}`);
      option.value = String(index);
      $('action-case').append(option);
    });
    renderAction();
    for (const source of evidence.sources) {
      const item = node('div', source.path, 'source');
      item.append(node('code', source.sha256));
      $('sources').append(item);
    }
    showSaved();
  } catch (error) {
    $('study-state').textContent = 'Evidence unavailable';
    $('gate-status').textContent = error.message;
    $('live').disabled = true;
  }
}

async function runLive(event) {
  event.preventDefault();
  if (!evidence) return;
  for (const id of ['live', 'saved', 'client', 'word', 'seed']) $(id).disabled = true;
  renderTrial([], '—', 'Live · running');
  $('trial-status').textContent = 'Running seven offline comparisons. The first trial also loads and verifies the model; allow up to three minutes.';
  try {
    const response = await fetch('/api/trial', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Oczy-Token': evidence.csrf},
      body: JSON.stringify({client: $('client').value, input: $('word').value, seed: Number($('seed').value)})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Trial failed');
    renderTrial(result.rows, result.target, 'Live · exploratory');
    $('trial-status').textContent = 'Completed with zero optimizer updates. This trial is outside the frozen evaluation record.';
  } catch (error) {
    $('trial-status').textContent = error.message;
    $('trial-mode').textContent = 'Live · error';
  } finally {
    for (const id of ['saved', 'client', 'word', 'seed']) $(id).disabled = false;
    $('live').disabled = !evidence.live_available;
  }
}

$('saved').addEventListener('click', showSaved);
$('action-case').addEventListener('change', renderAction);
$('trial-form').addEventListener('submit', runLive);
init();
