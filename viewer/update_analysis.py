#!/usr/bin/env python3
"""Add same-image DSH comparisons and cumulative full-output statistics."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import server
from export_html import write_html

HERE=Path(__file__).resolve().parent
SAME_SHA='0c9a39b078a6cdd53e6849e2638782ae163c33568cb19462c5954e26bdab5678'
RUN_NAMES={'base9b':'原始 Qwen3.5-9B','matched':'教师重写监督 · 同图留出',
           'step160':'SFT Step160','step170':'SFT Step170 · 原运行',
           'step170_resume':'SFT Step170 · 恢复运行','step200':'SFT Step200',
           'sft':'教师重写监督 · 31组件训练图'}


def update(output):
    match=re.search(r'<script type="application/json" id="offline-data">(.*?)</script>',output.read_text(),re.S)
    if not match:raise ValueError('Existing standalone snapshot is required')
    payload=json.loads(match.group(1))
    meta=server.remote_query({'kind':'analysis_bundle','section':'meta'})
    reference=meta['assets']['reference.png']
    assert sha256(base64.b64decode(reference.split(',',1)[1])).hexdigest()==SAME_SHA
    training_reference=meta['assets']['sft-reference.png']
    training_sha=sha256(base64.b64decode(training_reference.split(',',1)[1])).hexdigest()
    groups={key:[] for key in meta['sections']}
    analysis={'meta':{'default_case':'small96','source_session':'01a1f4b7-1c87-7ac1-bb82-4e5a1a5bee68',
                     'source_report':meta['source'],'source_report_sha256':meta['source_sha256'],
                     'collector_sha256':meta['collector_sha256'],
                     'exported_at':datetime.now(timezone.utc).isoformat(),
                     'notes':[
                         '前三列逐条累计此前保存的完整模型输出，不是当前输入窗口；正文、工具名/参数与Thinking分别重分词，工具调用用次数。',
                         '同图教师为DeepSeek Flash生成、Qwen27B改写当前Thinking的SFT格式留出记录，不是9B生成，也未用于训练。',
                         '同图教师按全部206条监督输出累计，未入选的2条原始回复不计，已按用户要求保留这一可视化范围。31组件训练例按其124条监督记录累计，另13条未入选原始回复不计。',
                         '教师累计历史的Thinking与本轮Thinking都使用对应重写标签，避免把原始教师历史与重写目标混算；实际输入保留窗口另列。',
                         '不同上下文/恢复策略仍构成比较条件差异，不能据此将变化全部归因于训练。170恢复前12条是原运行历史副本，不另计新增实验。',
                         '所有已保存的展示文本重新从源归档导出，未做页面字符截断。原输入本身的状态摘要仍按真实请求保留。',
                         '中断/length回复只代表实际保存部分，缺失记录不当作零；6次基座SSE完整不等于整个任务成功。'],
                     'all_train_summary':{'scope':'旧报告17,839条训练样本的输入保留窗口均值，不是完整历史累计，不能与上表混用','n':meta['n_train'],'retained_means':meta['all_train']}},
              'cases':[], 'runs':[], 'history_pool':[]}
    pool_keys={}
    def intern(message):
        key=json.dumps(message,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        if key not in pool_keys:
            pool_keys[key]=len(analysis['history_pool']);analysis['history_pool'].append(message)
        return pool_keys[key]
    jobs=[]
    for section,total in meta['sections'].items():
        for start in range(0,total,40):jobs.append({'kind':'analysis_bundle','section':section,'start':start,'stop':min(start+40,total)})
    done=0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(server.remote_query,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future];batch=future.result()
            if batch['source_sha256']!=meta['source_sha256']:raise ValueError('Source report changed during export')
            for row in batch['rows']:
                row['history']=[intern(batch['history_pool'][ref]) for ref in row['history']]
                groups[job['section']].append(row)
            done+=len(batch['rows'])
            print(f'full text verified {done}/{sum(meta["sections"].values())} records',flush=True)
    assets=meta['assets']
    analysis['cases']=[{'id':'small96','label':'96组件 · 原始9B / 同图教师 / 160、170、200','reference_sha256':SAME_SHA,'image':reference},
                       {'id':'train31','label':'31组件 · 教师训练样本（无同图学生记录）','reference_sha256':training_sha,'image':training_reference}]
    audit=server.read_json(server.AUDIT/'rollout-gap/summary.json',{})
    protocols=audit.get('protocols',{})
    for section in ('base9b','matched','step160','step170_resume','step170','step200','sft'):
        rows=sorted(groups[section],key=lambda row:row['ordinal'])
        if len(rows)!=meta['sections'][section]:raise ValueError('Incomplete exported run '+section)
        is_teacher=section in ('matched','sft')
        prior=[];totals={'past_output':0,'past_thinking':0,'past_tools':0}
        for i,row in enumerate(rows):
            row['retained_past_output']=row.get('retained_past_output',row['past_output'])
            row['retained_past_thinking']=row.get('retained_past_thinking',row['past_thinking'])
            row['retained_past_tools']=row.get('retained_past_tools',row['past_tools'])
            row.update(totals)
            row['full_history']=list(prior)
            row['history_complete']=True
            row['history_scope']=(f'本图全部{len(rows)}条已选重写监督记录中的此前{i}条；未入选原始回复未计。' if is_teacher else f'本运行从首个捕获开始、此前{i}次真实回复；恢复复制在本运行内只出现一次。')
            for call in row.get('tool_calls',[]):
                ident=call.get('id')
                if not ident:continue
                for later in rows[i+1:]:
                    found=next((analysis['history_pool'][ref] for ref in later['history']
                                if analysis['history_pool'][ref].get('role')=='tool' and analysis['history_pool'][ref].get('id')==ident),None)
                    if found is not None:
                        call['result']=found['text']
                        call['result_source']={'next_record_id':later['id'],'tool_call_id':ident}
                        break
            history_message={'role':'assistant','id':row['id'],'text':row['reply'],'thinking':row['thought'],
                             'calls':[{k:v for k,v in call.items() if k not in ('result','result_source')} for call in row.get('tool_calls',[])],
                             'images':0,'source_ordinal':i+1,'source_number':row['number'],'finish':row.get('finish'),
                             'complete':row.get('complete')}
            prior.append(intern(history_message))
            totals['past_output']+=row['output'];totals['past_thinking']+=row['thinking'];totals['past_tools']+=len(row.get('tool_calls',[]))
        checkpoint=0 if section=='base9b' else None if is_teacher else int(re.search(r'\d+',section).group())
        scope_note=('累计全部206条同图留出监督记录的完整输出，含重写Thinking；未入选的2条原始教师回复不计。当前输入保留窗口另列。' if section=='matched' else
                    '累计本图124条已选训练监督输出，含重写Thinking；未入选的13条原始回复不计。此图没有同图学生运行。' if is_teacher else
                    '累计本运行此前全部已捕获模型回复，包含完整正文、工具参数和Thinking；不受当前输入窗口裁剪影响。')
        analysis['runs'].append({'id':section,'label':RUN_NAMES[section],'group':'sft' if is_teacher else 'base' if section=='base9b' else 'trained',
                                 'checkpoint':checkpoint,'case_id':'train31' if section=='sft' else 'small96',
                                 'source':meta['source'],'policy':protocols.get(section,{}).get('state_prefix_policy','原归档状态前缀，历史Thinking从输入中移除'),
                                 'split':'train' if section=='sft' else 'validation' if section=='matched' else None,
                                 'input_measurement':'manifest' if is_teacher else 'archived', 'history_note':scope_note,
                                 'reviewer':'deepseek-flash','rows':rows})
    payload['analysis']=analysis
    payload['coverage']['analysis_records']=sum(len(run['rows']) for run in analysis['runs'])
    payload['coverage']['same_case_teacher_records']=206
    payload['coverage']['analysis_history_scope']='cumulative full outputs in each archived run or selected supervision sequence'
    write_html(payload,output)
    audit_summary={'at':datetime.now(timezone.utc).isoformat(),'source_report_sha256':meta['source_sha256'],
                   'case_sha256':SAME_SHA,'record_counts':{key:len(rows) for key,rows in groups.items()},
                   'last_cumulative':{run['id']:{key:run['rows'][-1][key] for key in ('past_output','past_thinking','past_tools','retained_past_output','retained_past_thinking')} for run in analysis['runs']},
                   'no_gpu_inference':True,'remote_mutations':False,'text_truncated_by_exporter':False}
    output.with_name('analysis-audit.json').write_text(json.dumps(audit_summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit_summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    update(parser.parse_args().output.resolve())
