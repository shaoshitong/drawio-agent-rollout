/* Read-only, dependency-free views of archived DSH requests and SFT targets. */
(function () {
  'use strict';
  const remembered = new WeakMap();
  const metrics = [
    ['past_output', '累计历史模型输出', 'tokens'],
    ['past_thinking', '累计历史 Thinking', 'tokens'],
    ['past_tools', '累计历史工具调用', '次'],
    ['thinking', '本轮 Thinking', 'tokens'],
    ['output', '本轮模型输出', 'tokens'],
  ];
  const metricTips = {past_output:'本次之前完整轨迹中的 Assistant 正文＋工具名与参数 tokens；不是当前上下文保留窗口。',past_thinking:'本次之前完整轨迹中的模型 Thinking tokens，包括后来从输入移除的历史推理。',past_tools:'本次之前完整轨迹中的工具调用个数；单位为次数，不是tokens。',thinking:'本次目标或模型回复的 Thinking tokens；中断时只计捕获到的部分。',output:'本次正文＋工具名与参数 tokens；不包括Thinking和后续工具返回。'};
  const groups = [
    ['base', '原始 9B', '原始模型在 DSH 中的实际输出'],
    ['sft', 'SFT 教师 · DeepSeek Flash', 'Thinking 由 Qwen 27B 重写'],
    ['trained', 'SFT 后 9B', '160 / 170 / 200 步的实际 rollout'],
  ];
  const roleLabels = {system:'系统', user:'用户 / 状态输入', assistant:'历史 Assistant', tool:'工具结果'};
  const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj || {}, key);
  const finite = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const number = (value, digits = 0) => finite(value) ? Number(value).toLocaleString('zh-CN', {maximumFractionDigits:digits}) : '未记录';
  const str = value => value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const mean = (rows, key) => {
    const values = rows.map(row => row[key]).filter(finite).map(Number);
    return values.length ? values.reduce((a,b) => a+b,0)/values.length : null;
  };
  const mk = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const add = (node, ...children) => {for (const child of children) if (child != null) node.append(child); return node;};
  const chip = (text, tone = '') => mk('span', `da-chip ${tone}`, text);
  const btn = (text, action, cls = '') => {
    const node = mk('button', `da-button ${cls}`, text); node.type = 'button';
    node.addEventListener('click', action); return node;
  };
  const toolName = tool => tool?.function?.name || tool?.name || '工具调用';
  const toolArgs = tool => tool?.function?.arguments ?? tool?.arguments ?? tool?.args;
  const callId = tool => tool?.id || tool?.tool_call_id || '';
  const toolsOf = row => Array.isArray(row.tool_calls) ? row.tool_calls : [];
  const rowNumber = (row, index) => row.number ?? row.ordinal ?? index + 1;
  const excerpted = value => /页面省略|页面节选|已省略|\[truncated\]/i.test(str(value));

  function textBlock(value, cls = '') {
    return mk('pre', `da-text ${cls}`, str(value));
  }
  function field(title, value, options = {}) {
    const root = mk('details', `da-field ${options.tone || ''}`); root.open = !!options.open;
    const summary = mk('summary');
    add(summary, mk('span', '', title), mk('span', 'da-char-count', value == null ? '未导出' : `${number(str(value).length)} 字符`));
    root.append(summary);
    if (value == null) root.append(mk('p', 'da-muted', options.missing || '此字段未保存在本次导出中。'));
    else if (str(value) === '') root.append(mk('p', 'da-muted', options.empty || '此字段为空。'));
    else {
      if (excerpted(value)) root.append(mk('p', 'da-excerpt', '页面文字为节选；token 统计使用源记录，未按节选重新计算。'));
      root.append(textBlock(value, options.code ? 'da-code' : ''));
    }
    return root;
  }
  function sourceInfo(title, value) {
    if (!value) return null;
    return field(title, value, {code:true});
  }
  function statsTable(runs, choose, selected) {
    const wrap = mk('div', 'da-table-scroll');
    const table = mk('table', 'da-stats');
    const head = mk('tr'); head.append(mk('th', '', '同图记录 · 每次请求的均值'));
    metrics.forEach(([key,label,unit]) => {const th=add(mk('th'),mk('span','',label),mk('small','',unit));th.title=metricTips[key];head.append(th);});
    const thead = mk('thead'); thead.append(head); table.append(thead);
    const body = mk('tbody');
    for (const run of runs) {
      const rows = run.rows || [], tr = mk('tr', run.id === selected ? 'da-selected-stat' : '');
      const title = mk('td'); title.append(btn(run.label || run.id, () => choose(run.id), 'da-table-run'));
      title.append(mk('small', 'da-sub', `${number(rows.length)} 次${run.group === 'sft' ? '教师决策' : '请求'}${run.split === 'validation' ? ' · 留出集' : ''}`));
      tr.append(title);
      metrics.forEach(([key]) => {const td=mk('td','',number(mean(rows,key),1));td.title=`${rows.filter(row=>finite(row[key])).length} / ${rows.length} 条有源计数；缺失值不补0`;tr.append(td);});
      body.append(tr);
    }
    table.append(body); wrap.append(table); return wrap;
  }
  function currentCalls(row) {
    const root = mk('div', 'da-current-tools');
    if (!has(row,'tool_calls')) {
      root.append(mk('p', 'da-muted', '本次导出未提供目标工具调用字段。')); return root;
    }
    if (!toolsOf(row).length) {
      root.append(mk('p', 'da-muted', row.complete === false ? '已捕获部分没有完整工具调用。' : '本次输出没有工具调用。')); return root;
    }
    for (const call of toolsOf(row)) {
      const box = mk('section', 'da-tool-call');
      add(box, add(mk('div', 'da-tool-title'), chip(toolName(call), 'tool'), callId(call) ? mk('code', 'da-call-id', callId(call)) : null));
      box.append(field('工具调用参数', toolArgs(call), {code:true}));
      root.append(box);
    }
    return root;
  }
  function executionResults(row, run) {
    const calls=toolsOf(row);
    if(!calls.length)return null;
    const root=mk('section','da-execution-results');
    root.append(mk('h4','','后续工具执行结果'));
    root.append(mk('p','da-muted',run.group==='sft'?'归档中的后续事件，不属于本次 Assistant 监督目标。':'这是模型调用工具之后的执行结果，单独列出。'));
    for(const call of calls){
      const box=mk('div','da-tool-call');box.append(chip(toolName(call),'tool'));
      if(has(call,'result')&&call.result!=null){
        box.append(field('对应执行结果',call.result,{code:true,empty:'对应工具返回为空字符串。'}));
        if(call.result_source||call.result_provenance)box.append(sourceInfo('执行结果来源',call.result_source||call.result_provenance));
      }else box.append(mk('p','da-muted','此导出未保存可对应的执行结果；不据此推断执行成败。'));
      root.append(box);
    }
    return root;
  }
  function messageHistory(row, full = false) {
    const source = full ? row.full_history : row.history;
    const messages = Array.isArray(source) ? source : [];
    const root = mk('details', 'da-history');
    add(root, mk('summary', '', `${full ? '累计历史输出记录' : '本次实际输入上下文'} · ${number(messages.length)} 条记录`));
    if (!messages.length) {root.append(mk('p', 'da-muted', full ? '此离线包未提供完整历史文本。累计统计的来源与覆盖范围见下方来源信息。' : '此记录没有导出输入消息。')); return root;}
    if (full && row.history_complete === false) root.append(mk('p','da-warning','历史捕获不完整；本次累计统计和文字仅覆盖已保存的记录。'));
    if (full && row.history_scope) root.append(mk('p','da-muted',str(row.history_scope)));
    const counts = Object.fromEntries(['system','user','assistant','tool'].map(role => [role,messages.filter(message => message.role === role).length]));
    const images = messages.reduce((n,message) => n + (finite(message.images) ? Number(message.images) : Array.isArray(message.images) ? message.images.length : 0),0);
    const countsRow = mk('div', 'da-message-counts');
    for (const [key,label] of [['system','系统'],['user','用户 / 状态'],['assistant','历史模型'],['tool','工具返回']]) countsRow.append(chip(`${label} ${counts[key]} 条`,key));
    if (images) countsRow.append(chip(`图像引用 ${number(images)} 处`, 'image'));
    root.append(countsRow);
    const known = new Map();
    for (const message of messages) for (const call of message.calls || message.tool_calls || []) if (callId(call)) known.set(callId(call), toolName(call));
    let mounted = false;
    root.addEventListener('toggle', () => {
      if (!root.open || mounted) return;
      mounted = true;
      const list = mk('div', 'da-messages');
      messages.forEach((message,index) => {
        const role = message.role || 'unknown';
        const block = mk('section', `da-message da-role-${role}`);
        const id = message.id || message.tool_call_id || '';
        const heading = add(mk('div', 'da-message-heading'), chip(roleLabels[role] || role,role),mk('span','',`消息 ${index+1}`));
        if (role === 'tool') heading.append(mk('span','',id && known.has(id) ? `对应 ${known.get(id)}` : '调用配对未记录'));
        if (id) heading.append(mk('code','da-call-id',id));
        block.append(heading);
        const thinking = message.thinking ?? message.reasoning_content ?? message.reasoning;
        if (thinking) block.append(field('历史 Thinking',thinking,{tone:'reasoning'}));
        const text = message.text ?? message.content;
        if (text != null && str(text) !== '') block.append(field(role === 'tool' ? '工具返回内容' : '消息正文',text,{code:role==='tool',open:messages.length<=8}));
        const calls = message.calls || message.tool_calls || [];
        for (const call of calls) block.append(field(`工具参数 · ${toolName(call)}`,toolArgs(call),{code:true}));
        const imageCount = finite(message.images) ? Number(message.images) : Array.isArray(message.images) ? message.images.length : 0;
        if (imageCount) block.append(mk('p','da-image-reference',`此消息含 ${number(imageCount)} 处图片输入；这里展示已导出的文字与图片标记。`));
        if (!thinking && (text == null || str(text)==='') && !calls.length && !imageCount) block.append(mk('p','da-muted','此消息未包含可显示的文字字段。'));
        list.append(block);
      });
      root.append(list);
    });
    return root;
  }
  function inputComposition(row, run) {
    const input = mk('section','da-input-context');
    const summary = add(mk('div','da-input-total'),mk('span','',run.input_measurement==='manifest'?'清单记录输入':'本轮实际输入'),mk('strong','',number(row.input)),mk('span','','tokens'));
    input.append(summary);
    input.append(mk('p','da-input-explain',`本次输入实际保留：历史 Assistant 输出 ${number(row.retained_past_output)} tokens、历史 Thinking ${number(row.retained_past_thinking)} tokens${finite(row.retained_past_tools) ? `、工具调用 ${number(row.retained_past_tools)} 次` : ''}。系统 / 用户与状态提示、工具返回、图像和消息模板也占用本轮上下文。`));
    const breakdown = row.input_breakdown;
    if (breakdown) {
      const parts = mk('div','da-breakdown');
      const labels={system_characters:'系统提示文本',user_characters:'用户与状态文本',tool_result_characters:'工具返回文本',assistant_characters:'历史模型文本',message_count:'输入消息',system:'系统提示',system_tokens:'系统提示',user:'用户与状态',user_tokens:'用户与状态',tool_results:'工具返回',tool_result_tokens:'工具返回',images:'图像',image_tokens:'图像',template:'消息模板',template_tokens:'消息模板',assistant:'历史 Assistant',assistant_tokens:'历史 Assistant',history_text_tokens:'历史模型文本',history_tool_tokens:'历史工具参数',context_tokens:'原生输入总长',input_tokens:'原生输入总长'};
      const entries = Array.isArray(breakdown) ? breakdown : Object.entries(breakdown).map(([key,value]) => typeof value === 'object' && value ? {key,...value} : {key,value});
      for (const part of entries) {
        if (!finite(part.value ?? part.tokens ?? part.count)) continue;
        const key = part.key || part.name || '';
        const label=part.label || labels[key] || key;
        if (!label) continue;
        const unit=part.unit||(key.endsWith('_characters')?'字符':key==='images'?'处':key==='message_count'?'条':'tokens');
        parts.append(add(mk('div'),mk('span','',label),mk('strong','',number(part.value ?? part.tokens ?? part.count)),mk('small','',unit)));
      }
      if (parts.childNodes.length) {input.append(parts);input.append(mk('p','da-muted','字符数、图片引用数和消息数用于说明输入来源，不能相加换算为输入 tokens。'));}
    }
    if(row.input_source)input.append(mk('p','da-muted',`输入计数来源：${str(row.input_source)}`));
    return input;
  }
  function rowDetail(row,run,index) {
    const root=mk('div','da-row-detail');
    if (row.complete === false) root.append(mk('p','da-warning','本轮流未完整结束。展示已捕获内容，不能当作完整模型回复。'));
    if (row.copied) root.append(mk('p','da-muted','此记录是恢复时复制的已有捕获，不能算作一次新增生成。'));
    if (run.split === 'validation') root.append(mk('p','da-heldout','同图教师来自留出集，以 SFT 格式展示；没有用于训练模型。'));
    root.append(inputComposition(row,run));
    const values=mk('div','da-row-metrics');
    metrics.forEach(([key,label,unit])=>values.append(add(mk('div'),mk('span','',label),mk('strong','',number(row[key])),mk('small','',unit))));
    root.append(values);
    if (row.history_complete === false) root.append(mk('p','da-warning','累计历史捕获不完整；表中的数值仅代表已经保存的记录。'));
    if (row.history_scope) root.append(mk('p','da-muted',str(row.history_scope)));
    root.append(messageHistory(row,true),messageHistory(row));
    const output=mk('section','da-assistant-output');
    add(output,add(mk('div','da-output-heading'),chip('Assistant 输出','assistant'),mk('span','',run.group==='sft'?'本次监督目标':'本轮实际模型回复')));
    output.append(field(run.group==='sft'?'监督 Thinking':'本轮 Thinking',has(row,'thought')?row.thought:null,{tone:'reasoning',open:true,empty:'本次目标 / 回复的 Thinking 字段为空。'}));
    output.append(field('正文',has(row,'reply')?row.reply:null,{open:!!str(row.reply).trim(),empty:'本次输出没有正文；工具参数单独显示在下方。'}));
    output.append(currentCalls(row)); root.append(output);const results=executionResults(row,run);if(results)root.append(results);
    const provenance={source:run.source,case_id:run.case_id,split:run.split,source_id:row.id,source_number:row.number,history_sha256:row.history_sha256,response_sha256:row.response_sha256,finish_reason:row.finish,complete:row.complete,policy:run.policy,input_source:row.input_source,history_scope:row.history_scope,history_complete:row.history_complete};
    root.append(sourceInfo('来源与计数依据',provenance));
    return root;
  }

  window.renderDshAnalysis = function (host, analysis) {
    if (!host || !analysis) return;
    const cases=Array.isArray(analysis.cases)?analysis.cases:[];
    const runs=Array.isArray(analysis.runs)?analysis.runs:[];
    const state=remembered.get(host)||{caseId:analysis.meta?.default_case||cases[0]?.id||runs[0]?.case_id,runId:null,query:'',opened:new Map()};
    remembered.set(host,state);
    const hydrateMessages=(values,pool)=>(Array.isArray(values)?values:Array.isArray(values?.messages)?values.messages:[]).map(message=>Number.isInteger(message)?pool?.[message]||{role:'unknown',text:`离线包缺少消息 #${message}`} :message);
    const hydrateRow=row=>({...row,history:hydrateMessages(row.history,analysis.history_pool),full_history:hydrateMessages(row.full_history,analysis.full_history_pool||analysis.history_pool)});
    function render() {
      host.replaceChildren();
      const root=mk('div','da-analysis');host.append(root);
      const selectedCase=cases.find(item=>item.id===state.caseId)||cases[0];
      if (selectedCase) state.caseId=selectedCase.id;
      const available=runs.filter(run=>!state.caseId||run.case_id===state.caseId);
      if (!available.some(run=>run.id===state.runId)) state.runId=available.find(run=>run.group==='base')?.id||available[0]?.id;
      const selected=available.find(run=>run.id===state.runId);
      const heading=mk('div','da-heading');
      add(heading,add(mk('div'),mk('h2','','同图看三组轨迹'),mk('p','','原始 9B、SFT 格式教师，以及训练后 9B 的真实输入与输出。')));
      const caseField=mk('label','da-case-select');caseField.append(mk('span','','参考图 / case'));
      const caseSelect=mk('select');caseSelect.setAttribute('aria-label','选择对照 case');
      for(const item of cases){const option=mk('option','',item.label||item.id);option.value=item.id;caseSelect.append(option);}
      caseSelect.value=state.caseId||'';
      caseSelect.addEventListener('change',()=>{state.caseId=caseSelect.value;state.runId=null;state.query='';render();});
      add(caseField,caseSelect);heading.append(caseField);root.append(heading);
      const explanation=mk('div','da-context-note');
      add(explanation,mk('strong','','完整历史累计，与本轮输入窗口分开看'),mk('p','','表中前三项统计本次之前完整轨迹的模型输出、Thinking 和工具调用。本轮输入可能仅保留最近 4 个工具循环，更早内容进入状态包，历史 Thinking 也可能被移除。累计历史可以大于当前输入总长；展开某一轮可分别查看完整历史记录和当时实际输入。'));
      root.append(explanation);
      if(selectedCase){
        const caseInfo=mk('div','da-case-info');
        if(typeof selectedCase.image==='string'&&/^data:image\//.test(selectedCase.image)){const image=mk('img');image.src=selectedCase.image;image.alt=selectedCase.label||'对照参考图';image.loading='lazy';caseInfo.append(image);}
        const caption=mk('div');caption.append(mk('strong','',selectedCase.label||selectedCase.id));
        if(selectedCase.reference_sha256)caption.append(mk('code','da-source-hash',selectedCase.reference_sha256));
        const presentGroups=new Set(available.map(run=>run.group));
        caption.append(mk('p','da-muted',presentGroups.size===3?'下列三组使用同一参考图；策略差异仍需结合来源查看。':'此 case 只有下列已有记录；缺少的组不做跨图配对。'));
        caseInfo.append(caption);root.append(caseInfo);
      }
      const cards=mk('div','da-group-cards');
      for(const [key,label,subtitle] of groups){
        const matches=available.filter(run=>run.group===key),active=selected?.group===key;
        const preferred=matches.find(run=>key==='trained'&&Number(run.checkpoint)===200)||matches[0];
        const card=mk('section',`da-group-card ${active?'da-active-card':''}`);
        const title=btn(label,()=>{if(matches.length){state.runId=preferred.id;state.query='';render();}},'da-card-title');title.disabled=!matches.length;
        add(card,title,mk('p','da-group-subtitle',subtitle));
        if(matches.length){
          const select=mk('select');select.setAttribute('aria-label',`${label}运行记录`);
          matches.forEach(run=>{const option=mk('option','',run.label||run.id);option.value=run.id;select.append(option);});
          select.value=active?state.runId:preferred.id;
          select.addEventListener('change',()=>{state.runId=select.value;state.query='';render();});
          card.append(select);
          const run=matches.find(item=>item.id===select.value)||matches[0],rows=run.rows||[];
          const inputs=rows.map(row=>row.input).filter(finite).map(Number);
          card.append(add(mk('div','da-card-input'),mk('span','',run.input_measurement==='manifest'?'清单记录输入':'平均实际输入'),mk('strong','',number(mean(rows,'input'),1)),mk('span','','tokens')));
          card.append(mk('p','da-muted',inputs.length?`范围 ${number(Math.min(...inputs))}–${number(Math.max(...inputs))} · ${number(rows.length)} 条记录`:`${number(rows.length)} 条记录 · 输入总长未导出`));
          if(run.split==='validation')card.append(chip('同图留出教师 · 未参加训练','heldout'));
        }else card.append(mk('p','da-muted','未找到此 case 的记录'));
        cards.append(card);
      }
      root.append(cards);
      if(!selected){root.append(mk('p','da-muted','当前 case 尚无可浏览的调用。'));return;}
      if(selected.history_note)root.append(add(mk('div','da-history-note'),chip('统计范围'),mk('p','',str(selected.history_note))));
      root.append(statsTable(available,id=>{state.runId=id;state.query='';render();},state.runId));
      root.append(mk('p','da-table-note','均值按本 case 的已导出请求 / 决策计算；“累计历史工具调用”的单位是次数，其余四项为 tokens。累计历史与当前输入窗口是不同范围，五项不能相加得到输入总长；缺失计数不补 0。'));
      const details=mk('details','da-definitions');details.append(mk('summary','','指标定义与资料覆盖'));
      const definitions=analysis.meta?.definitions;
      if(definitions){const list=mk('dl');for(const [key,value] of Array.isArray(definitions)?definitions.map(item=>[item.label||item.key,item.text||item.description||item.value]):Object.entries(definitions)){add(list,mk('dt','',key),mk('dd','',str(value)));}details.append(list);}
      const notes=analysis.meta?.notes;
      if(Array.isArray(notes)){const list=mk('ul');notes.forEach(note=>list.append(mk('li','',str(note))));details.append(list);}
      if(analysis.meta?.all_train_summary) details.append(sourceInfo('全训练集统计（与当前同图均值分开）',analysis.meta.all_train_summary));
      root.append(details);
      const rows=selected.rows||[];
      const header=mk('div','da-run-heading');
      add(header,add(mk('div'),mk('h3','',selected.label||selected.id),mk('p','da-muted',`${number(rows.length)} 次${selected.group==='sft'?'教师决策':'已捕获请求'} · 点击色带或某一轮，查看输入和本次 Assistant 输出`)));
      const search=mk('input','da-call-search');search.type='search';search.placeholder='筛选本轮内容或工具';search.setAttribute('aria-label','筛选调用');search.value=state.query;header.append(search);root.append(header);
      const timeline=mk('div','da-trace');
      const lineRefs=[];
      function openAt(index){const target=lineRefs[index];if(!target)return;target.open=true;target.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});}
      for(const [key,label] of [['input','输入'],['thinking','模型'],['tool','工具']]){
        const lane=mk('div','da-trace-lane');lane.append(mk('span','',label));const track=mk('div','da-trace-track');
        track.style.gap=rows.length>80?'0':rows.length>35?'1px':'2px';
        rows.forEach((row,index)=>{const active=key==='input'||key==='thinking'&&(Number(row.thinking)>0||!!row.thought)||key==='tool'&&toolsOf(row).length>0;
          const point=btn('',()=>openAt(index),`da-trace-point da-lane-${key} ${active?'':'da-inactive'} ${row.complete===false?'da-incomplete':''}`);
          point.disabled=!active;point.title=`第 ${index+1} 次 · 源记录 ${rowNumber(row,index)} · 输入 ${number(row.input)} tokens${row.complete===false?' · 捕获未完成':''}`;point.setAttribute('aria-label',point.title);track.append(point);
        });lane.append(track);timeline.append(lane);
      }
      timeline.append(add(mk('div','da-trace-axis'),mk('span','','第 1 次'),mk('span','',`第 ${number(rows.length)} 次`)));root.append(timeline);
      const list=mk('div','da-rows');root.append(list);
      rows.forEach((row,index)=>{
        const detail=mk('details',`da-request ${row.complete===false?'da-request-incomplete':''}`);lineRefs.push(detail);
        const summary=mk('summary');
        add(summary,mk('span','da-request-index',String(index+1).padStart(2,'0')),add(mk('span','da-request-main'),mk('strong','',toolsOf(row).map(toolName).join(' · ')||str(row.reply).trim().slice(0,70)|| (row.complete===false?'捕获中断 / 未完成回复':'模型回复')),mk('small','',`输入 ${number(row.input)} tokens · Thinking ${number(row.thinking)} · 输出 ${number(row.output)}${row.xml?' · 已有 XML':''}`)),chip(row.complete===false?'流未结束':row.finish||'已导出',row.complete===false?'error':''));
        detail.append(summary);
        const searchText=[index+1,rowNumber(row,index),row.thought,row.reply,...toolsOf(row).map(toolName)].map(str).join(' ').toLowerCase();
        detail.dataset.search=searchText;
        let mounted=false;
        detail.addEventListener('toggle',()=>{if(detail.open&&!mounted){mounted=true;detail.append(rowDetail(hydrateRow(row),selected,index));}if(detail.open)state.opened.set(selected.id,index);});
        list.append(detail);
      });
      function filterRows(){let shown=0;for(const row of lineRefs){row.hidden=!!state.query&&!row.dataset.search.includes(state.query.toLowerCase());if(!row.hidden)shown++;}empty.hidden=shown>0;}
      const empty=mk('p','da-muted','没有符合筛选条件的调用。');empty.hidden=true;list.append(empty);
      search.addEventListener('input',()=>{state.query=search.value;filterRows();});filterRows();
      const first=state.opened.get(selected.id)??0;if(lineRefs[first]&&!lineRefs[first].hidden)lineRefs[first].open=true;
    }
    render();
    return {refresh:render};
  };
})();
