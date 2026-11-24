'use strict';

const offline = window.DSH_OFFLINE && typeof window.DSH_OFFLINE === 'object' ? window.DSH_OFFLINE : null;
const initialView = new URLSearchParams(location.search).get('view');
const state = {
  catalog: null, view: ['analysis', 'sft', 'rollout', 'training'].includes(initialView) ? initialView : offline?.analysis ? 'analysis' : 'sft', query: '', checkpoint: 'all', sample: null,
  left: null, right: null, metric: 'turns', diff: false, episode: 'loaded',
  requestSerial: 0, detailCache: new Map(), expanded: new Set(),
};
const $ = selector => document.querySelector(selector);
const content = $('#content');
const number = value => value === null || value === undefined || value === '' ? '—' : Number(value).toLocaleString('en-US');
const idOf = value => String(value.id ?? value.sample_id ?? value.run ?? '');
const callNumber = (call, index = 0) => call.index ?? call.call ?? call.id ?? index + 1;
const checkpointOf = run => run.checkpoint ?? null;
const checkpointLabel = value => value === 0 || value === 'base' ? '原始 9B' : value == null ? '阶段未知' : `Step ${value}`;
const categoryNames = {assistant_only:'正文决策',crop_related:'裁图相关',image_read:'读取图像',other_tool:'其他工具',xml_related:'XML 相关'};
const effectNames = {native_xml_changed:'XML 实物改变',code_file_changed:'脚本改变',candidate_render_changed:'渲染产物改变',image_files_changed:'图像文件改变',no_case_file_change:'本步无文件变化',next_snapshot_unavailable:'后继快照缺失',case_files_deleted:'删除文件',other_case_file_changed:'其他文件改变',runtime_assets_changed:'运行资源改变'};
function exposureBand(row) {
  if (!row.current_curriculum) return 'historical';
  if (row.first_seen_step == null) return 'unobserved';
  const start=Math.floor((Number(row.first_seen_step)-1)/50)*50+1;
  return `${start}-${Math.min(start+49,Number(state.catalog?.meta?.training_steps)||218)}`;
}
function exposureLabel(key) {return key==='historical'?'历史导出':key==='unobserved'?'尚无已训记录':`Update ${key}`;}
const roleNames = {assistant: '模型', user: '输入', system: '系统', tool: '工具结果', generator: 'Generator', reviewer: 'Reviewer'};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function append(parent, ...children) {
  for (const child of children) if (child !== null && child !== undefined) parent.append(child);
  return parent;
}
function button(label, callback, className = 'button') {
  const node = el('button', className, label);
  node.type = 'button';
  node.addEventListener('click', callback);
  return node;
}
function badge(text, tone = '') { return el('span', `badge ${tone}`, text); }
function clear(node) { node.replaceChildren(); return node; }
function stringify(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}
function plainContent(value) {
  if (typeof value === 'string') return value;
  if (!value) return '';
  if (!Array.isArray(value)) return stringify(value);
  return value.map(part => {
    if (typeof part === 'string') return part;
    if (part.type === 'text' || part.type === 'input_text') return part.text ?? '';
    if (part.type?.includes('image')) return '[图像附件]';
    return stringify(part);
  }).join('\n');
}
function copied(value) { return JSON.parse(JSON.stringify(value)); }
function hasOwn(object, key) { return object != null && Object.prototype.hasOwnProperty.call(object, key); }
function offlineDetail(value) {
  const data = copied(value);
  const messages = values => values.map(message => {
    if (!Number.isInteger(message)) return message;
    if (!hasOwn(offline.message_pool, message)) throw new Error(`离线包未包含上下文消息 #${message}。`);
    return copied(offline.message_pool[message]);
  });
  for (const side of [data.original, data.rewritten]) if (Array.isArray(side?.messages)) side.messages = messages(side.messages);
  if (Array.isArray(data.request_messages)) data.request_messages = messages(data.request_messages);
  return data;
}
async function request(path) {
  if (offline) {
    const url = new URL(path, 'https://offline.invalid');
    if (url.pathname === '/api/catalog') {
      if (!offline.catalog) throw new Error('离线包未包含轨迹索引。');
      const catalog = copied(offline.catalog);
      catalog.sft = (catalog.sft || []).filter(row => hasOwn(offline.sft, idOf(row)));
      return catalog;
    }
    if (url.pathname === '/api/sft') {
      const id = url.searchParams.get('id');
      if (!hasOwn(offline.sft, id)) throw new Error(`此 SFT 配对未打包：${id}。离线文件不会连接远端。`);
      return offlineDetail(offline.sft[id]);
    }
    if (url.pathname === '/api/sft-index') {
      const episode = url.searchParams.get('episode');
      const rows = (offline.catalog?.sft || []).filter(row => row.current_curriculum && row.episode === episode && hasOwn(offline.sft, idOf(row)));
      if (!rows.length) throw new Error(`此 episode 的决策未打包：${episode}。`);
      return {sft: copied(rows)};
    }
    if (url.pathname === '/api/call') {
      const run = url.searchParams.get('run'), call = url.searchParams.get('call');
      if (!hasOwn(offline.calls?.[run], call)) throw new Error(`此调用的详情未打包：${run} / ${call}。现有摘要仍可浏览。`);
      return offlineDetail(offline.calls[run][call]);
    }
    throw new Error('此资源未打包到离线文件。');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, {signal: controller.signal, cache: 'no-store'});
    if (!response.ok) {
      const body = await response.text();
      let message = body;
      try { message = JSON.parse(body).error || JSON.parse(body).message || body; } catch (_) {}
      throw new Error(`${response.status} · ${String(message).slice(0, 600)}`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('读取超时。远端原始记录可能暂不可用，可以稍后重试。');
    throw error;
  } finally { clearTimeout(timer); }
}
function resolveImage(url) {
  if (typeof url !== 'string') return '';
  if (url.startsWith('data:image/')) return url;
  if (!url.startsWith('/api/')) return '';
  if (!offline) return url;
  const embedded = offline.images?.[url];
  return typeof embedded === 'string' && embedded.startsWith('data:image/') ? embedded : '';
}
function imageDialog(url, title) {
  const source = resolveImage(url);
  if (!source) return;
  const previous = document.activeElement;
  const dialog = el('dialog', 'image-dialog');
  dialog.setAttribute('aria-label', title);
  const close = () => { if (dialog.open && typeof dialog.close === 'function') dialog.close(); else dialog.remove(); };
  const toolbar = el('div', 'image-dialog-toolbar');
  append(toolbar, el('strong', '', title), button('关闭 ×', close));
  const image = el('img'); image.src = source; image.alt = title;
  append(dialog, toolbar, image);
  dialog.addEventListener('close', () => {dialog.remove(); previous?.focus();}, {once:true});
  dialog.addEventListener('click', event => {if (event.target === dialog) close();});
  document.body.append(dialog);
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else {dialog.setAttribute('open', ''); toolbar.querySelector('button').focus();}
}
function snapshotTime(value) {
  if (value == null || value === '') return '未记录';
  const time = new Date(typeof value === 'number' ? value * 1000 : value);
  return Number.isNaN(time.getTime()) ? '未记录' : `${time.toLocaleString('zh-CN', {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',timeZone:'Asia/Shanghai'})} UTC+8`;
}
function renderOfflineHeader() {
  if (!offline) return;
  document.body.classList.add('offline-mode');
  document.title = 'Trajectory Lab · 离线快照';
  $('#refresh').textContent = '重置视图 ↺';
  const header = $('.app-header');
  let summary = $('#offline-summary');
  if (!summary) {summary = el('div', 'offline-summary'); summary.id = 'offline-summary'; header.after(summary);}
  clear(summary);
  const coverage = offline.coverage || {}, catalog = state.catalog || {};
  const pairs = Object.keys(offline.sft || {}).length;
  const current = (catalog.sft || []).filter(row => row.current_curriculum).length;
  const historical = (catalog.sft || []).filter(row => !row.current_curriculum).length;
  const counts = `SFT 配对 ${number(coverage.sft_pairs ?? pairs)}（当前 ${number(coverage.current_pairs ?? current)} / 历史 ${number(coverage.historical_pairs ?? historical)}）`;
  const calls = [];
  if (coverage.captured_calls != null) calls.push(`调用捕获 ${number(coverage.captured_calls)} 份`);
  if (coverage.unique_captures != null) calls.push(`独立捕获 ${number(coverage.unique_captures)} 份`);
  calls.push(`已打包调用详情 ${number(coverage.full_call_outputs)} 份`, `仅摘要 ${number(coverage.summary_calls)} 份`, `带输入上下文 ${number(coverage.request_contexts)} 份`);
  if (coverage.incomplete_streams != null) calls.push(`流未完整结束 ${number(coverage.incomplete_streams)} 份`);
  if (coverage.loss_points != null) calls.push(`loss ${number(coverage.loss_points)} 点`);
  append(summary, badge('离线快照'), el('span', '', `导出 ${snapshotTime(offline.exported_at)}`), el('span', 'offline-coverage', `${counts} · ${calls.join(' · ')}`));
  const dataset = catalog.meta?.dataset;
  if (dataset?.total_decisions != null) summary.append(el('span', 'offline-source-note', `源数据总量 ${number(dataset.total_decisions)} 决策 / ${number(dataset.total_episodes)} episodes；本文件只含上述已打包内容。捕获副本不计为新增实验。`));
}
function notice(text, tone = '') { return el('div', `notice ${tone}`, text); }
function loading(text = '正在读取真实轨迹…') { return append(el('div', 'loading-state'), el('span', 'spinner'), document.createTextNode(text)); }
function metric(label, value, subtitle = '') {
  return append(el('div', 'metric'), el('div', 'metric-label', label), el('div', 'metric-value', value), el('div', 'metric-sub', subtitle));
}
function sourceDetails(value) {
  if (!value) return null;
  const detail = el('details', 'provenance');
  append(detail, el('summary', '', '来源与核对信息'), el('pre', '', stringify(value)));
  return detail;
}
function pageHeading(title, subtitle, extra) {
  return append(el('div', 'page-top'), append(el('div', 'page-heading'), el('h2', '', title), el('p', '', subtitle)), extra);
}
function sampleTitle(sample) { return sample.title || sample.category || sample.sample_id || sample.id; }
function runLabel(run) { return run.label || `${checkpointLabel(checkpointOf(run))} · ${run.id}`; }
function policyLabel(policy) { return policy == null ? '未记录' : typeof policy === 'string' ? policy : policy.kind || policy.state_prefix || '详见来源'; }
function toolTone(tool) {
  return tool.result_status === 'failed' || (tool.result_exit_codes || []).some(x => Number(x) !== 0) ? 'error' : 'tool';
}
function callTone(call) {
  if (call.complete_stream === false || call.reasoning_loop || ['error', 'failed'].includes(call.status)) return 'error';
  if ((call.tools || []).some(tool => toolTone(tool) === 'error')) return 'error';
  return call.role === 'reviewer' ? 'reviewer' : 'generator';
}

function renderSidebar() {
  const isSft = state.view === 'sft';
  $('#list-title').textContent = isSft ? '监督样本' : '运行记录';
  $('#list-note').textContent = isSft ? '同一决策的原始轨迹与实际重写版本' : '按 checkpoint 与原图来源浏览';
  const records = isSft ? (state.catalog.sft || []).filter(row => state.episode !== 'historical' || !row.current_curriculum) : (state.catalog.runs || []);
  $('#episode-field').hidden = !isSft;
  document.querySelector('label[for="checkpoint-filter"]').textContent=isSft?'首次监督':'训练阶段';
  const filter = clear($('#checkpoint-filter'));
  append(filter, new Option('全部阶段', 'all'));
  const steps = [...new Set(records.map(row => isSft ? exposureBand(row) : String(checkpointOf(row) ?? 'unknown')))];
  steps.sort((a, b) => Number(a) - Number(b)).forEach(step => {
    filter.add(new Option(isSft ? exposureLabel(step) : step === 'unknown' ? '未知阶段' : checkpointLabel(step), step));
  });
  if (!steps.includes(state.checkpoint)) state.checkpoint = 'all';
  filter.value = state.checkpoint;
  const list = clear($('#record-list'));
  const filtered = records.filter(row => {
    const step = isSft ? exposureBand(row) : String(checkpointOf(row) ?? 'unknown');
    return (state.checkpoint === 'all' || step === state.checkpoint)
      && JSON.stringify([row.id, row.title, row.category, row.label, row.episode, row.case_id]).toLowerCase().includes(state.query.toLowerCase());
  });
  for (const row of filtered) {
    const active = isSft ? state.sample === idOf(row) : state.left === idOf(row);
    const node = button('', () => {
      if (isSft) { state.sample = idOf(row); showSft(); }
      else { state.left = idOf(row); chooseCompatibleRight(); showRollout(); }
      renderSidebar();
    }, `record${active ? ' selected' : ''}`);
    if (active) node.setAttribute('aria-current', 'true');
    const title = isSft ? sampleTitle(row) : checkpointLabel(checkpointOf(row));
    append(node, append(el('div', 'record-head'), el('span', 'record-title', title), badge(isSft ? categoryNames[row.category] || row.category || 'SFT' : row.calls?.length ? `${row.calls.length} 次调用` : row.tool_count != null ? `${row.tool_count} 工具` : '汇总')));
    append(node, el('div', 'record-sub', isSft ? `${row.episode || row.id} · 阶段 ${row.stage ?? '—'}${row.first_seen_step != null ? ` · 首次 U${row.first_seen_step}` : ''}` : row.label || row.id));
    const meta = isSft
      ? [row.current_curriculum ? `目标 ${number(row.rewritten_tokens)} tokens · ${row.split || 'split未记录'}` : `${number(row.original_tokens)} → ${number(row.rewritten_tokens)} 输入 tokens`, row.current_curriculum ? '当前冻结清单' : '历史导出']
      : [row.case_id || (row.reference_sha256 || '').slice(0, 12) || '原图未知', statusLabel(row)];
    append(node, append(el('div', 'record-meta'), el('span', '', meta[0]), el('span', '', meta[1])));
    if (!isSft && row.calls?.length) {
      const mini = el('div', 'record-mini');
      row.calls.forEach(call => {
        const mark = el('span'); mark.style.flex = '1';
        mark.style.background = callTone(call) === 'error' ? 'var(--red)' : (call.tools || []).length ? 'var(--orange)' : 'var(--purple)';
        mini.append(mark);
      });
      node.append(mini);
    }
    list.append(node);
  }
  if (!filtered.length) list.append(el('p', 'list-empty', '没有符合条件的记录'));
}
function statusLabel(run) {
  if (typeof run.status === 'string') return ({aggregate_only:'仅有汇总审计',captured:'已捕获',incomplete:'未完成',HOST_DETECTED_REASONING_LOOP:'宿主检测到重复'}[run.status] || run.status);
  return run.loop ? '检测到重复' : run.render ? '有渲染记录' : '已归档';
}
function tokenDifference(a, b) {
  if (a == null || b == null || !Number.isFinite(Number(a)) || !Number.isFinite(Number(b)) || Number(a) === 0) return '—';
  const change = (Number(b) - Number(a)) / Number(a) * 100;
  return `${change > 0 ? '+' : ''}${change.toFixed(1)}%`;
}
function textBox(text, className = '', otherText = null, side = '') {
  const value = plainContent(text);
  const node = el('div', `text-body ${className}${value ? '' : ' empty'}`);
  if (!value) { node.textContent = '该字段在此记录中未提供'; return node; }
  if (!state.diff || otherText == null) { node.textContent = value; return node; }
  const other = plainContent(otherText);
  if (other === value) { node.textContent = value; return node; }
  const tokenPattern = /\s+|[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu;
  const remaining = new Map();
  for (const token of other.match(tokenPattern) || []) remaining.set(token, (remaining.get(token) || 0) + 1);
  for (const token of value.match(tokenPattern) || []) {
    const count = remaining.get(token) || 0;
    if (count > 0 || /^\s+$/.test(token)) { node.append(document.createTextNode(token)); remaining.set(token, Math.max(count - 1, 0)); }
    else node.append(el('mark', side === 'original' ? 'diff-removed' : 'diff-added', token));
  }
  return node;
}
function detailGroup(label, value, className = '', open = false, otherText = null, side = '') {
  const group = el('details', 'details-group'); group.open = open;
  const length = plainContent(value).length;
  append(group, append(el('summary'), document.createTextNode(label), el('span', 'length', `${number(length)} 字符`)), textBox(value, className, otherText, side));
  return group;
}
function toolDescription(tool) { return tool.function?.name || tool.name || '工具调用'; }
function toolArguments(tool) { return tool.function?.arguments ?? tool.arguments ?? tool.argument_head; }
function sidePane(title, side, subtitle) {
  const pane = el('section', 'pane');
  append(pane, append(el('div', `pane-head ${side}`), append(el('div', 'pane-head-title'), el('span', 'pane-bar'), append(el('div'), el('div', 'pane-title', title), el('p', '', subtitle)))));
  const body = el('div', 'pane-body'); pane.append(body);
  return {pane, body};
}
function renderMessages(messages, target, counterparts = [], side = '') {
  if (!messages?.length) { target.append(el('div', 'placeholder', '此记录没有展开的上下文消息')); return; }
  messages.forEach((message, i) => {
    const role = message.role || 'unknown';
    const peer = counterparts[i]?.role === role ? counterparts[i] : {};
    const body = el('div', 'message-content');
    const label = el('div', 'role-label', `${i + 1} · ${roleNames[role] || role}`);
    if (message.tool_call_id) label.append(el('span', 'call-id', message.tool_call_id));
    body.append(label);
    if (message.reasoning || message.reasoning_content) body.append(detailGroup('Thinking', message.reasoning || message.reasoning_content, 'reasoning', false, peer.reasoning || peer.reasoning_content, side));
    if (message.content !== null && message.content !== undefined) body.append(textBox(message.content, role === 'tool' ? 'code' : '', peer.content, side));
    for (const tool of message.tool_calls || []) body.append(detailGroup(toolDescription(tool), toolArguments(tool), 'code', false));
    if (!message.content && !message.reasoning && !message.tool_calls?.length) body.append(el('div', 'small-note', '该消息未包含可显示文本'));
    const row = el('div', 'dialogue-row'); row.id = `message-${side}-${i}`;
    target.append(append(row, el('div', `role-dot ${role}`, {assistant: 'A', tool: 'T', user: 'U', system: 'S'}[role] || '·'), body));
  });
}

function messageTimeline(messages, side, completion) {
  const root = el('div', 'timeline message-timeline');
  root.append(el('div', 'small-note', '输入上下文 → 本次监督输出'));
  root.append(append(el('div', 'message-phase-labels'), el('span'), el('span', '', `${number(messages.length)} 条输入消息`), el('span', 'output-phase', '目标输出')));
  const jumpToOutput = () => {
    const target = document.getElementById(`target-${side}`);
    if (!target) return;
    const thinking = target.querySelector('details'); if (thinking) thinking.open = true;
    target.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block:'center'});
  };
  const tools = completion.tool_calls || [];
  for (const [lane, label] of [['input', '输入'], ['reasoning', '模型'], ['tool', '工具']]) {
    const track = el('div', 'lane-track');
    track.style.gap = messages.length > 80 ? '0' : messages.length > 35 ? '1px' : '2px';
    messages.forEach((message, index) => {
      const active = lane === 'input' ? ['system', 'user'].includes(message.role) : lane === 'reasoning' ? message.role === 'assistant' : message.role === 'tool' || message.tool_calls?.length > 0;
      const point = button('', () => {
        const row = document.getElementById(`message-${side}-${index}`);
        if (!row) return;
        let parent = row.parentElement;
        while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
        row.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block:'center'});
      }, `lane-cell${active ? ` lane-${lane}` : ''}`);
      point.style.flex = '1'; point.disabled = !active;
      point.title = `消息 ${index+1} · ${roleNames[message.role] || message.role}`;
      point.setAttribute('aria-label', point.title); track.append(point);
    });
    let endpoint = el('span', 'message-endpoint empty-endpoint', '—');
    if (lane === 'reasoning') {
      endpoint = button('Assistant 输出', jumpToOutput, 'message-endpoint output-endpoint');
      endpoint.title = '本次监督目标 · Thinking、正文与工具调用参数';
      endpoint.setAttribute('aria-label', `${side === 'original' ? '原始' : '重写'}轨迹 · 跳到本次监督输出`);
    } else if (lane === 'tool' && tools.length) {
      endpoint = button(tools.map(toolDescription).join(' / '), jumpToOutput, 'message-endpoint tool-endpoint');
      endpoint.title = '模型输出中的工具调用参数；工具尚未执行';
      endpoint.setAttribute('aria-label', `${side === 'original' ? '原始' : '重写'}轨迹 · 输出中的工具调用`);
    } else if (lane === 'tool' && completion.tool_calls_available === false) {
      endpoint = el('span', 'message-endpoint missing-endpoint', '未导出');
      endpoint.title = '此历史导出未保留目标工具字段，不能据此判断有无调用';
    }
    root.append(append(el('div','timeline-lane'),el('span','lane-name',label),track,endpoint));
  }
  return root;
}

async function showSft() {
  const serial = ++state.requestSerial;
  const rows = state.catalog.sft || [];
  const visibleRows = state.episode === 'historical' ? rows.filter(item => !item.current_curriculum) : rows;
  const row = visibleRows.find(item => idOf(item) === state.sample) || visibleRows[0];
  clear(content);
  if (!row) { content.append(notice('还没有可浏览的 SFT 重写配对记录。')); return; }
  state.sample = idOf(row);
  renderSidebar();
  const actions = el('div', 'page-actions');
  const diffControl = el('label', 'check'); const input = el('input'); input.type = 'checkbox'; input.checked = state.diff;
  input.addEventListener('change', () => {state.diff = input.checked; showSft();});
  append(diffControl, input, document.createTextNode('突出新增 / 减少的词')); actions.append(diffControl);
  content.append(pageHeading(sampleTitle(row), `${row.episode || row.id} · 阶段 ${row.stage ?? '—'} · 监督数据配对`, actions));
  const measures = el('div', 'metrics');
  measures.append(metric('原始输入', row.current_curriculum ? '—' : number(row.original_tokens), 'tokens'), metric('SFT 输入', row.current_curriculum ? '—' : number(row.rewritten_tokens), 'tokens'), metric('SFT 目标', row.current_curriculum ? number(row.rewritten_tokens) : '—', '完整目标 tokens'), metric('决策位置', number(row.decision_order ?? row.call_id), `阶段 ${row.stage ?? '—'}`));
  content.append(measures);
  const pending = loading(); content.append(pending);
  try {
    const data = await request(`/api/sft?id=${encodeURIComponent(row.id)}`);
    if (serial !== state.requestSerial || state.view !== 'sft') return;
    pending.remove();
    if (data.description) content.append(notice(data.description));
    if (row.current_curriculum) {
      const evidence = row.first_seen_step != null ? `首次进入监督为 Update ${row.first_seen_step}，依据已核验采样顺序与有效 batch32 推导${row.seen_updates?.length ? `；已出现 ${row.seen_updates.length} 次` : ''}。` : '在当前已核验的训练曝光记录中，尚未找到此决策。';
      content.append(el('p','small-note',`${evidence}${row.artifact_effect ? ` 相邻快照记录：${effectNames[row.artifact_effect] || row.artifact_effect}。` : ''}`));
    }
    if (data.partial_reason) content.append(notice(data.partial_reason, 'warning'));
    const original = data.original || {}, rewritten = data.rewritten || {};
    if (original.input_tokens != null || rewritten.input_tokens != null) {
      clear(measures).append(metric('原始输入', number(original.input_tokens), 'tokens'), metric('SFT 输入', number(rewritten.input_tokens), 'tokens'), metric('输入长度变化', tokenDifference(original.input_tokens, rewritten.input_tokens)), metric('监督 Thinking', `${number(original.reasoning_tokens)} → ${number(rewritten.reasoning_tokens)}`, 'tokens'));
    }
    if (Array.isArray(data.images) && data.images.length) {
      const images = el('div', 'image-strip');
      for (const image of data.images) {
        const url = resolveImage(image.url), title = image.title || '轨迹图片';
        const figure = el('figure');
        if (url) {
          const node = el('img'); node.src = url; node.alt = title; node.loading = 'lazy';
          const link = el('a'); link.href = url; link.setAttribute('aria-label', `放大查看${title}`);
          link.addEventListener('click', event => {event.preventDefault(); imageDialog(url, title);});
          link.append(node); figure.append(link);
        } else if (offline) figure.append(el('p', 'small-note', '此图片未打包'));
        else continue;
        figure.append(el('figcaption', '', title)); images.append(figure);
      }
      content.append(images);
    }
    const left = sidePane('原始轨迹', 'original', '重写前 · 真实监督来源');
    const right = sidePane('SFT 重写轨迹', 'rewritten', '重写后 · 实际使用版本');
    const split = append(el('div', 'split'), left.pane, right.pane); content.append(split);
    for (const [name, current, peer, target] of [['original', original, rewritten, left.body], ['rewritten', rewritten, original, right.body]]) {
      if (current.messages?.length) target.append(messageTimeline(current.messages, name, current));
      if (current.messages?.length) {
        const context = el('details', 'details-group');
        context.append(el('summary', '', `展开输入上下文 · ${current.messages.length} 条消息`));
        const inner = el('div'); renderMessages(current.messages, inner, peer.messages || [], name); context.append(inner); target.append(context);
      }
      const supervised = el('section', 'supervised-output'); supervised.id = `target-${name}`;
      supervised.append(append(el('div', 'supervised-heading'), el('h3', '', '本次监督输出'), badge('Assistant', 'generator')));
      const reasoning = current.reasoning ?? current.current_reasoning;
      if (reasoning !== undefined) supervised.append(detailGroup('当前监督 Thinking', reasoning, 'reasoning', true, peer.reasoning ?? peer.current_reasoning, name));
      const output = current.content ?? current.output;
      if (output) supervised.append(detailGroup('模型正文 / 输出', output, '', true, peer.content ?? peer.output, name));
      else if (current.content_available === false) supervised.append(el('p', 'small-note', '此历史导出未保留目标正文。'));
      else if (current.content_available === true) supervised.append(el('p', 'small-note', '本次输出无正文，见 Thinking 与下方工具调用。'));
      const tools = current.tool_calls;
      if (Array.isArray(tools) && tools.length) tools.forEach(tool => supervised.append(detailGroup(`工具参数 · ${toolDescription(tool)}`, toolArguments(tool), 'code', false)));
      else if (tools === undefined || current.tool_calls_available === false) supervised.append(el('p', 'small-note', '本导出未提供完整工具参数，无法由此判断有无工具调用。'));
      target.append(supervised);
    }
    content.append(el('p', 'small-note', '词级标色反映两侧文本中新增或减少的词，消息顺序与原记录一致。上下文和当前监督目标分开显示。'));
    content.append(sourceDetails(data.provenance || data.source || row.source));
  } catch (error) {
    if (serial !== state.requestSerial) return;
    pending.replaceWith(notice(`配对详情读取失败 · ${error.message}`, 'error'));
  }
}

function chooseCompatibleRight() {
  const runs = state.catalog.runs || [];
  const left = runs.find(run => idOf(run) === state.left);
  if (!left) return;
  const current = runs.find(run => idOf(run) === state.right);
  const hash = left.reference_sha256 || left.case_id;
  const matching = runs.filter(run => idOf(run) !== state.left && (run.reference_sha256 || run.case_id) === hash);
  if (current && idOf(current) !== state.left && (current.reference_sha256 || current.case_id) === hash) return;
  const base = matching.find(run => Number(checkpointOf(run)) === 0 || checkpointOf(run) === 'base');
  state.right = idOf(base || matching.find(run => Number(checkpointOf(run)) === 200) || matching[0] || runs.find(run => idOf(run) !== state.left) || left);
}
function runSelect(label, selected, onChange) {
  const wrap = el('label', 'select-box'); wrap.append(el('span', '', label));
  const select = el('select');
  for (const run of state.catalog.runs || []) select.add(new Option(runLabel(run), idOf(run)));
  select.value = selected || '';
  select.addEventListener('change', () => onChange(select.value)); wrap.append(select); return wrap;
}
function comparisonWarnings(left, right) {
  const warnings = [];
  const a = left.reference_sha256, b = right.reference_sha256;
  if (!a || !b) warnings.push('部分运行缺少原图 SHA-256，无法核验同题配对。');
  else if (a !== b) warnings.push('两侧原图不同，当前只适合分别查看，不能视为同题模型对比。');
  if (left.reviewer !== right.reviewer) warnings.push('两侧 Reviewer 不一致。');
  if (stringify(left.policy) !== stringify(right.policy)) warnings.push('两侧上下文 / 状态策略不同，差异包含运行条件变化。');
  if (state.left === state.right) warnings.push('当前两侧选择了同一份运行。');
  return warnings;
}
function timeline(run, body) {
  const calls = run.calls || [];
  const root = el('div', 'timeline');
  const legend = el('div', 'timeline-key');
  for (const [kind, label] of [['input', '输入'], ['reasoning', 'Thinking'], ['tool', '工具']]) legend.append(append(el('span'), el('i', `swatch ${kind}`), document.createTextNode(label)));
  root.append(legend);
  const weight = call => state.metric === 'seconds' ? Math.max(Number(call.seconds) || 0.1, 0.1) : state.metric === 'tokens' ? Math.max(Number(call.completion_tokens) || Number(call.reasoning_tokens) || 1, 1) : 1;
  for (const [kind, label] of [['input', '输入'], ['reasoning', '模型'], ['tool', '工具']]) {
    const track = el('div', 'lane-track');
    calls.forEach((call, i) => {
      const active = kind === 'input' ? true : kind === 'reasoning' ? Number(call.reasoning_tokens) > 0 || call.reasoning_head || call.reasoning_characters : (call.tools || []).length > 0;
      const point = button('', () => {
        const target = document.getElementById(callDomId(run, call, i));
        target?.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center'});
        target?.querySelector('button')?.click();
      }, `lane-cell${active ? ` lane-${kind}` : ''}`);
      point.style.flex = String(weight(call));
      point.disabled = !active;
      const value = state.metric === 'seconds' ? `${number(call.seconds)} 秒` : state.metric === 'tokens' ? `${number(call.completion_tokens)} 输出 tokens` : `调用 ${callNumber(call, i)}`;
      point.title = `${label} · ${value}`; point.setAttribute('aria-label', point.title);
      track.append(point);
    });
    root.append(append(el('div', 'timeline-lane'), el('span', 'lane-name', label), track));
  }
  const sum = calls.reduce((total, call) => total + weight(call), 0);
  const end = state.metric === 'turns' ? `${calls.length} 次调用` : state.metric === 'seconds' ? `${sum.toFixed(1)} 秒（已记录）` : `${number(Math.round(sum))} tokens（已记录）`;
  root.append(append(el('div', 'timeline-scale'), el('span', '', '0'), el('span', '', end)));
  body.append(root);
}
function callDomId(run, call, index) { return `call-${idOf(run)}-${callNumber(call, index)}`.replace(/[^\w-]/g, '_'); }
function callTitle(call) {
  if ((call.tools || []).length) return call.tools.map(tool => tool.name || tool.function?.name || 'tool').join(' · ');
  return call.complete_stream === false ? '回复未完成 / 流被中断' : call.role === 'reviewer' ? 'Reviewer 回复' : '模型回复';
}
function callDetailContent(data, detail) {
  clear(detail);
  if (data.partial_reason || data.complete === false) detail.append(notice(data.partial_reason || '当前显示的是已保存片段，尚不包含完整原始流。', 'warning'));
  const call = data.call || {};
  const thought = data.reasoning ?? call.reasoning ?? call.reasoning_head;
  if (thought !== undefined && thought !== '') detail.append(detailGroup('Thinking', thought, 'reasoning', true));
  const text = data.content ?? call.content_text;
  if (text) detail.append(detailGroup('模型正文', text, '', true));
  for (const tool of data.tool_calls || []) detail.append(detailGroup(`工具调用 · ${toolDescription(tool)}`, toolArguments(tool), 'code', true));
  for (const result of data.tool_results || []) detail.append(detailGroup(`工具结果 · ${result.name || result.tool_call_id || result.role || 'tool'}`, result.content ?? result.result ?? result, 'code', false));
  if (!data.tool_calls?.length && call.tools?.length) {
    call.tools.forEach(tool => {
      detail.append(detailGroup(`工具参数片段 · ${tool.name}`, tool.argument_head, 'code', false));
      if (tool.result_tail) detail.append(detailGroup(`工具结果尾部 · ${tool.name}`, tool.result_tail, 'code', false));
    });
  }
  if (data.request_messages?.length) {
    const messages = el('details', 'details-group');
    messages.append(el('summary', '', `输入上下文 · ${data.request_messages.length} 条消息`));
    const inner = el('div'); renderMessages(data.request_messages, inner); messages.append(inner); detail.append(messages);
  }
  if (!detail.childNodes.length) detail.append(el('p', 'small-note', '此调用没有可显示的正文记录。'));
  detail.append(sourceDetails(data.provenance));
}
function runPane(run, side) {
  const {pane, body} = sidePane(checkpointLabel(checkpointOf(run)), side, run.label || run.id);
  const model = run.model || state.catalog.meta?.model || 'Qwen 9B';
  body.append(el('div', 'run-identity', `Generator ${model} · Reviewer ${run.reviewer || '身份未记录'}\n状态策略 ${policyLabel(run.policy)}`));
  const top = el('div', 'page-actions');
  append(top, badge(statusLabel(run), run.loop || String(run.status || '').toLowerCase().includes('fail') ? 'error' : ''), badge(run.calls?.length ? `${run.calls.length} 次调用` : run.tool_count != null ? `${run.tool_count} 工具动作（汇总）` : '调用数未记录'), badge(`${(run.xml_files || []).length} 个 XML`));
  body.append(top);
  if (run.duplicate_count) body.append(el('p', 'small-note', `${run.duplicate_count} 份捕获为恢复时复制的历史，不能记成新增推进。`));
  if (run.artifacts) {
    const imageStrip = el('div', 'image-strip');
    for (const [kind, title] of [['reference', '输入参考图'], ['round1', '第一轮渲染'], ['round2', '第二轮渲染']]) {
      if (!run.artifacts[kind]) continue;
      const figure = el('figure'); const url = resolveImage(`/api/run-image?run=${encodeURIComponent(idOf(run))}&kind=${kind}`);
      if (url) {
        const image = el('img'); image.src = url; image.alt = title; image.loading = 'lazy';
        image.addEventListener('error', () => { image.hidden = true; figure.prepend(el('p', 'small-note', '图片读取不可用')); });
        const link = el('a'); link.href = url; link.setAttribute('aria-label', `放大查看${title}`);
        link.addEventListener('click', event => {event.preventDefault(); imageDialog(url, title);});
        link.append(image); figure.append(link);
      } else figure.append(el('p', 'small-note', '此图片未打包'));
      figure.append(el('figcaption', '', title)); imageStrip.append(figure);
    }
    if (imageStrip.childNodes.length) body.append(imageStrip);
  }
  if (run.calls?.length) timeline(run, body);
  const list = el('div', 'call-list');
  (run.calls || []).forEach((call, index) => {
    const key = `${idOf(run)}:${callNumber(call, index)}`;
    const row = el('article', 'call-row'); row.id = callDomId(run, call, index);
    const heading = el('div');
    heading.append(append(el('div', 'call-name'), badge(call.role === 'reviewer' ? 'Reviewer' : 'Generator', callTone(call)), document.createTextNode(`  ${callTitle(call)}`)));
    const preview = call.tools?.[0]?.description || call.content_text || call.reasoning_head || call.reasoning_tail || '';
    heading.append(el('div', 'call-preview', preview));
    const meta = el('div', 'call-meta');
    append(meta, el('span', '', `${number(call.prompt_tokens)} in`), el('span', '', '·'), el('span', '', `${number(call.completion_tokens)} out`), el('span', '', '·'), el('span', '', `${number(call.seconds)} s`));
    heading.append(meta);
    const detail = el('div', 'call-detail'); detail.hidden = !state.expanded.has(key);
    const toggle = button('', async () => {
      const open = detail.hidden;
      detail.hidden = !open; toggle.setAttribute('aria-expanded', String(open));
      if (!open) { state.expanded.delete(key); return; }
      state.expanded.add(key);
      if (state.detailCache.has(key)) { callDetailContent(state.detailCache.get(key), detail); return; }
      clear(detail).append(loading('读取原始调用…'));
      try {
        const data = await request(`/api/call?run=${encodeURIComponent(idOf(run))}&call=${encodeURIComponent(callNumber(call, index))}`);
        state.detailCache.set(key, data); callDetailContent(data, detail);
      } catch (error) { clear(detail).append(notice(error.message, 'error'), button('重新读取', () => { detail.hidden = true; toggle.click(); })); }
    }, 'call-button');
    toggle.setAttribute('aria-expanded', String(!detail.hidden));
    append(toggle, el('span', 'call-index', String(callNumber(call, index)).padStart(2, '0')), heading, el('span', 'call-status', call.complete_stream === false ? '中断' : call.finish_reason || '—'));
    append(row, toggle, detail); list.append(row);
    if (!detail.hidden) {
      if (state.detailCache.has(key)) callDetailContent(state.detailCache.get(key), detail);
      else { detail.hidden = true; toggle.click(); }
    }
  });
  if (!run.calls?.length) {
    const facts = el('div', 'metrics aggregate-metrics');
    if (run.tool_count != null) facts.append(metric('已记录工具动作', number(run.tool_count)));
    if (run.first_xml_tool != null) facts.append(metric('首次写入 XML', `第 ${number(run.first_xml_tool)} 次`, '工具动作'));
    if (run.successful_render_count_min != null) facts.append(metric('成功官方渲染', `≥ ${number(run.successful_render_count_min)}`));
    if (facts.childNodes.length) list.append(facts);
    list.append(notice('此历史运行提供汇总审计与产物，当前没有逐调用正文索引；下方来源保留完整定位信息。'));
  }
  body.append(list, sourceDetails({source: run.source, reference_sha256: run.reference_sha256, reviewer: run.reviewer, model, checkpoint: run.checkpoint, policy: run.policy}));
  return pane;
}
function showRollout() {
  ++state.requestSerial;
  const runs = state.catalog.runs || [];
  clear(content);
  if (!runs.length) { content.append(notice('尚无实际 rollout 记录。')); return; }
  if (!state.left) state.left = idOf(runs.find(run => Number(checkpointOf(run)) === 200) || runs[0]);
  chooseCompatibleRight();
  const left = runs.find(run => idOf(run) === state.left) || runs[0];
  const right = runs.find(run => idOf(run) === state.right) || left;
  renderSidebar();
  content.append(pageHeading('沿着动作，定位分岔', '并排查看同一原图上的真实调用。色带可跳到对应轮次，展开输入、Thinking、工具及执行结果。'));
  const requested = state.catalog.meta?.requested_comparison;
  if (requested && requested.available === false) content.append(notice(requested.reason || '所要求的冻结 9B Reviewer 对照组尚未发现，下面显示现有运行及其真实模型身份。', 'warning'));
  const selects = append(el('div', 'run-selects'),
    runSelect('左侧运行', state.left, value => {state.left = value; chooseCompatibleRight(); renderSidebar(); showRollout();}),
    runSelect('右侧运行', state.right, value => {state.right = value; showRollout();}));
  content.append(selects);
  const differences = comparisonWarnings(left, right);
  if (differences.length) content.append(notice(differences.join(' '), 'warning'));
  const toolbar = el('div', 'toolbar'); toolbar.append(el('span', 'toolbar-title', '调用轨迹'));
  const switches = el('div', 'switch');
  for (const [key, label] of [['turns','轮次'],['seconds','耗时'],['tokens','输出 tokens']]) {
    const control = button(label, () => {state.metric = key; showRollout();}, state.metric === key ? 'active' : '');
    control.setAttribute('aria-pressed', String(state.metric === key)); switches.append(control);
  }
  toolbar.append(switches); content.append(toolbar);
  content.append(append(el('div', 'split'), runPane(left, 'original'), runPane(right, 'rewritten')));
}

function lossChart(records) {
  const points = records.map((row, index) => Array.isArray(row) ? {step: Number(row[0]), loss: Number(row[1])} : {step:Number(row.step ?? row.iteration ?? index+1), loss:Number(row.loss ?? row.train_loss)})
    .filter(row => Number.isFinite(row.step) && Number.isFinite(row.loss)).sort((a,b) => a.step-b.step);
  if (!points.length) return notice('当前索引没有逐步 loss 数据。阶段性的样本 CE 与训练 loss 属于不同口径，页面会按来源分别展示。');
  const wrapper = el('section', 'chart-wrap'); wrapper.append(el('h3', '', '完整训练 loss'));
  const ns='http://www.w3.org/2000/svg';
  const chartWidth=Math.max(280, Math.min(850, content.clientWidth - 34));
  const chartRight=chartWidth-16;
  const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox',`0 0 ${chartWidth} 250`); svg.setAttribute('role','img'); svg.setAttribute('aria-label',`${points.length} 个实际记录的训练更新及其 loss`);
  const create=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!=null)n.textContent=text;return n;};
  const xmin=points[0].step,xmax=points[points.length-1].step;
  const ymin=Math.min(...points.map(p=>p.loss)),ymax=Math.max(...points.map(p=>p.loss));
  const x=v=>50+(v-xmin)/(xmax-xmin||1)*(chartRight-50), y=v=>208-(v-ymin)/(ymax-ymin||1)*170;
  for(let i=0;i<5;i++){
    const value=ymin+(ymax-ymin)*i/4;
    svg.append(create('line',{x1:50,y1:y(value),x2:chartRight,y2:y(value),stroke:'var(--line)','stroke-width':1}));
    svg.append(create('text',{x:40,y:y(value)+4,'text-anchor':'end',fill:'var(--muted)','font-size':11},value.toFixed(2)));
  }
  const tickCount=chartWidth<440?3:5;
  for(let i=0;i<=tickCount;i++){
    const value=xmin+(xmax-xmin)*i/tickCount;
    svg.append(create('text',{x:x(value),y:229,'text-anchor':'middle',fill:'var(--muted)','font-size':11},String(Math.round(value))));
  }
  svg.append(create('path',{d:points.map((p,i)=>`${i?'L':'M'}${x(p.step).toFixed(1)} ${y(p.loss).toFixed(1)}`).join(' '),fill:'none',stroke:'var(--purple)','stroke-width':1.8}));
  points.forEach(point=>{
    const mark=create('circle',{cx:x(point.step),cy:y(point.loss),r:3,fill:'transparent'});
    mark.append(create('title',{},`Step ${point.step} · loss ${point.loss}`));svg.append(mark);
  });
  wrapper.append(svg,el('p','chart-caption',`${points.length} 个实际日志点 · 横轴为完成的训练步数，纵轴为训练 loss。`));
  return wrapper;
}
function showTraining() {
  ++state.requestSerial; clear(content);
  const meta=state.catalog.meta || {},runs=state.catalog.runs || [];
  const steps=[...new Set(runs.map(checkpointOf).filter(v=>v!==null))].sort((a,b)=>Number(a)-Number(b));
  content.append(pageHeading('训练与自主行为', '训练日志、实际保存阶段和自主 rollout 分开核对。'));
  content.append(append(el('div','metrics'),metric('最后完整更新',number(meta.training_steps),'训练日志'),metric('训练状态',meta.training_state || '未记录'),metric('有 rollout 的阶段',String(steps.length),'仅统计已有记录')));
  content.append(lossChart(meta.loss || []));
  const title=el('div','toolbar');title.append(el('h3','','阶段记录'));content.append(title);
  const table=el('table','summary-table');
  const thead=el('thead'),hr=el('tr');['训练阶段','运行数','已索引调用','已存 XML','Reviewer'].forEach(label=>hr.append(el('th','',label)));thead.append(hr);table.append(thead);
  const tbody=el('tbody');
  steps.forEach(step=>{
    const grouped=runs.filter(run=>checkpointOf(run)===step),row=el('tr');
    const fields=[checkpointLabel(step),number(grouped.length),number(grouped.reduce((n,run)=>n+(run.calls || []).length,0)),number(grouped.reduce((n,run)=>n+(run.xml_files || []).length,0)),[...new Set(grouped.map(run=>run.reviewer || '未记录'))].join(' / ')];
    fields.forEach(value=>row.append(el('td','num',value)));tbody.append(row);
  });
  table.append(tbody);content.append(table);
  const notes=el('ul','note-list');
  for(const note of meta.notes || []) {
    let text = typeof note === 'string' ? note : stringify(note);
    if (offline && text.includes('页面首批索引')) text = `${text.split('，页面首批索引')[0]}；本文件的收录范围见顶部离线覆盖说明。`;
    notes.append(el('li','',text));
  }
  if(notes.childNodes.length)content.append(notes);
  content.append(sourceDetails(meta.provenance || meta.sources || meta.training_source));
}
function showView() {
  document.body.classList.toggle('analysis-view',state.view==='analysis');
  for(const tab of document.querySelectorAll('[data-view]')){
    const active=tab.dataset.view===state.view;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));
  }
  if(state.view==='analysis'){
    ++state.requestSerial; clear(content);
    if(offline?.analysis && window.renderDshAnalysis) window.renderDshAnalysis(content,offline.analysis);
    else content.append(notice('这份文件未包含三组轨迹对照。'));
    return;
  }
  renderSidebar();
  if(state.view==='sft')showSft();else if(state.view==='rollout')showRollout();else showTraining();
}
async function loadCatalog() {
  const refresh=$('#refresh');refresh.disabled=true;
  const error=clear($('#fatal-error'));error.hidden=true;
  try {
    state.catalog=await request('/api/catalog');
    $('#analysis-tab').hidden=!offline?.analysis;
    $('#model-name').textContent=state.catalog.meta?.model || 'Qwen 9B';
    $('#sft-count').textContent=number((state.catalog.sft || []).length);
    $('#run-count').textContent=number((state.catalog.runs || []).length);
    const timestamp=state.catalog.meta?.catalog_source_time || state.catalog.meta?.generated_at;
    if(timestamp) $('#snapshot-time').textContent=`${offline ? '离线证据快照' : '证据快照'} ${snapshotTime(timestamp)}`;
    else if (offline) $('#snapshot-time').textContent = '离线证据快照 · 来源时间未记录';
    $('#training-state').textContent=`${state.catalog.meta?.training_state || '已归档'} · ${number(state.catalog.meta?.training_steps)} updates`;
    const episodeSelect = clear($('#episode-filter'));
    if (offline) {
      episodeSelect.add(new Option(`本文件全部 · ${(state.catalog.sft || []).length} 个配对`, 'loaded'));
      const historical = (state.catalog.sft || []).filter(row => !row.current_curriculum);
      if (historical.length) episodeSelect.add(new Option(`历史导出 · 已打包 ${historical.length} 个配对`, 'historical'));
      const episodes = new Map();
      for (const row of state.catalog.sft || []) if (row.current_curriculum && row.episode) episodes.set(row.episode, (episodes.get(row.episode) || 0) + 1);
      for (const [id, count] of episodes) episodeSelect.add(new Option(`${id.slice(0, 12)} · 已打包 ${count} 决策`, id));
    } else {
      episodeSelect.add(new Option('首批已载入 · 4 个完整 episodes', 'loaded'));
      episodeSelect.add(new Option('仅历史 10 条导出', 'historical'));
      for (const episode of state.catalog.meta?.dataset?.episodes || []) episodeSelect.add(new Option(`${episode.id.slice(0, 12)} · ${episode.count} 决策 · ${episode.split}`, episode.id));
    }
    state.episode = 'loaded';
    renderOfflineHeader();
    state.detailCache.clear(); showView();
  } catch(problem){clear(content);error.textContent=`索引读取失败 · ${problem.message}`;error.hidden=false;content.append(button('重新读取',loadCatalog));}
  finally{refresh.disabled=false;}
}
for(const tab of document.querySelectorAll('[data-view]'))tab.addEventListener('click',()=>{state.view=tab.dataset.view;state.query='';state.checkpoint='all';$('#search').value='';showView();});
$('#search').addEventListener('input',event=>{state.query=event.target.value;renderSidebar();});
$('#checkpoint-filter').addEventListener('change',event=>{state.checkpoint=event.target.value;renderSidebar();});
$('#episode-filter').addEventListener('change', async event => {
  const selected = event.target.value; state.episode = selected; state.query = ''; $('#search').value = '';
  if (selected === 'loaded') { await loadCatalog(); return; }
  if (selected === 'historical') {
    state.sample = idOf((state.catalog.sft || []).find(row => !row.current_curriculum) || {});
    renderSidebar(); showSft(); return;
  }
  clear(content).append(loading(offline ? '载入此 episode 已打包的决策…' : '载入此 episode 的全部决策…'));
  event.target.disabled = true;
  try {
    const data = await request(`/api/sft-index?episode=${encodeURIComponent(selected)}`);
    if (state.episode !== selected) return;
    const historical = (state.catalog.sft || []).filter(row => !row.current_curriculum);
    state.catalog.sft = [...(data.sft || []).map(row => ({...row, current_curriculum:true})), ...historical];
    state.sample = idOf(state.catalog.sft[0] || {});
    $('#sft-count').textContent = number(state.catalog.sft.length); renderSidebar(); showSft();
  } catch(error) { clear(content).append(notice(`Episode 读取失败 · ${error.message}`, 'error')); }
  finally { event.target.disabled = false; }
});
$('#refresh').addEventListener('click', () => {
  if (offline) {
    state.view = 'sft'; state.query = ''; state.checkpoint = 'all'; state.sample = null;
    state.left = null; state.right = null; state.metric = 'turns'; state.diff = false;
    state.expanded.clear(); $('#search').value = '';
  }
  loadCatalog();
});
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer); resizeTimer=setTimeout(() => { if (state.view==='training' && state.catalog) showTraining(); },120);
});
loadCatalog();
