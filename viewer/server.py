#!/usr/bin/env python3
"""Serve a local, read-only SFT and rollout evidence browser."""
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from data_adapter import get_sft_catalog, get_sft_detail, get_sft_image

HERE = Path(__file__).resolve().parent
LINKED = Path(os.environ.get('DSH_EVIDENCE_WORKSPACE', '/Users/shaoshitong/Documents/Codex/2026-11-02/chat-thread-codex-threads-01a19ace-60f1'))
AUDIT = LINKED/'outputs/dsh-review-20261122'
SSH = Path(os.environ.get('DSH_EVIDENCE_SSH','/Users/shaoshitong/Documents/deepseek-harness/default-workspace/h20b/bssh'))
SOCKET = os.environ.get('DSH_EVIDENCE_SOCKET','/tmp/drawio-sft-h20-20261123.sock')
SUMMARY = AUDIT/'existing-call-summary.json'
REMOTE = HERE/'remote_reader.py'
CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
REMOTE_LOCK = threading.Semaphore(2)
MAX_CACHE = 10*1024**2


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def remote_query(payload):
    key = json.dumps(payload, sort_keys=True)
    with CACHE_LOCK:
        if key in CACHE:
            CACHE.move_to_end(key)
            return json.loads(CACHE[key])
    script = 'PAYLOAD=' + repr(payload) + '\n' + REMOTE.read_text()
    command = [str(SSH), '-oConnectTimeout=12', '-S', SOCKET,
               'env PYTHONDONTWRITEBYTECODE=1 /DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121/env/bin/python -']
    with REMOTE_LOCK:
        process = subprocess.run(command, input=script, text=True, capture_output=True, timeout=70)
    if process.returncode:
        try:
            problem=json.loads(process.stdout)
            message=problem.get('message','Remote evidence unavailable')
        except ValueError:
            message='远端证据读取失败或连接超时。现有摘要仍可查看。'
        raise RuntimeError(message)
    value = json.loads(process.stdout)
    packed=json.dumps(value,ensure_ascii=False)
    if len(packed.encode()) < 4*1024**2:
        with CACHE_LOCK:
            CACHE[key]=packed
            while sum(len(v.encode()) for v in CACHE.values()) > MAX_CACHE:
                CACHE.popitem(last=False)
    return value


def run_catalog():
    data=read_json(SUMMARY, {'runs':[],'calls':[]})
    calls_by_run={}
    signatures=Counter(c.get('response_sha256') for c in data['calls'] if c.get('response_sha256'))
    for call in data['calls']:
        c=dict(call)
        c.update(id=str(c['call']),index=c['call'],role='generator',summary_only=True)
        c['duplicate_capture']=signatures.get(c.get('response_sha256'),0)>1
        calls_by_run.setdefault(c['run'],[]).append(c)
    runs=[]
    for run in data['runs']:
        name=run['run']
        match=re.search(r'^step(\d+)',name)
        checkpoint=int(match.group(1)) if match else None
        calls=calls_by_run.get(name,[])
        run.update(id=name,label=f"Step {checkpoint} · "+name.replace(f'step{checkpoint}-',''),checkpoint=checkpoint,
                   model='Qwen3.5-9B',reviewer='deepseek-flash',case_id=run.get('reference_sha256','')[:12],
                   calls=calls,source=str(SUMMARY),policy=run.get('state_prefix_policy'),render=run.get('render1'),
                   status=(run.get('loop') or {}).get('outcome') or ('未完成/捕获中断' if any(not c.get('complete_stream') for c in calls) else '已捕获'),
                   duplicate_count=sum(c['duplicate_capture'] for c in calls),
                   reviewer_evidence='evaluation/protocol.json（已只读核对）')
        runs.append(run)
    extra=read_json(HERE/'baseline_runs.json',[])
    runs.extend(extra.get('runs',[]) if isinstance(extra,dict) else extra)
    return runs


def catalog():
    index=read_json(HERE/'current-manifest-index.json',{})
    current=index.get('sft',[])
    historical=get_sft_catalog()
    for row in current:
        row['current_curriculum']=True
        row['step']=None
        row['comparison_kind']='supervision_rewrite'
    training=read_json(HERE/'training-summary.json',{})
    return {'meta':{
        'generated_at':datetime.now(timezone.utc).isoformat(),'model':'Qwen3.5-9B',
        'training_steps':218,'training_state':'用户停止，最新完整保存210步',
        'catalog_source_time':read_json(SUMMARY,{}).get('at'),
        'notes':['原始与重写是同一监督决策，checkpoint rollout 是模型实际生成。',
                 '当前冻结清单21103决策/200 episodes，页面首批索引4个完整episodes；历史10条导出单列。',
                 '151份旧捕获含36份恢复副本，对应115个独立捕获，不按151次独立实验统计。',
                 '历史不同版本的前缀策略、恢复与采样条件可能不同，配对时请核对来源。'],
        'requested_comparison':{'available':True,'reviewer':'deepseek-flash','scope':'原始9B与训练后9B的已有同图记录，固定DeepSeek Flash评审。历史基座为汇总覆盖，状态策略差异另行提示。'},
        'dataset':{k:v for k,v in index.items() if k!='sft'},'loss':training.get('loss',[]),
        'training_source':training.get('source'),
        'source_thread':'01a1f4b7-1c87-7ac1-bb82-4e5a1a5bee68'},
        'sft':current+historical,'runs':run_catalog()}


def get_call(run_id, number):
    runs=run_catalog()
    run=next((r for r in runs if r['id']==run_id),None)
    if run is None:
        raise KeyError('Unknown run')
    call=next((c for c in run['calls'] if str(c['index'])==number),None)
    if call is None:
        raise KeyError('Unknown call')
    if call.get('request') and call.get('response'):
        next_call=next((c for c in run['calls'] if c['index']==call['index']+1),{})
        args={'kind':'call','request':call['request'],'response':call['response']}
        if next_call.get('request'):
            args['next_request']=next_call['request']
        try:
            detail=remote_query(args)
            detail['call']=call
            return detail
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            reason=str(error)
    else:
        reason=call.get('detail_note','这份旧轨迹只保存了工具级摘要，未提供逐调用原文。')
    return {'call':call,'request_messages':[], 'reasoning':call.get('reasoning_head','')+'\n…\n'+call.get('reasoning_tail',''),
            'content':call.get('content_text',''),'tool_calls':call.get('tools',[]),'tool_results':[],
            'complete':False,'summary_only':True,'partial_reason':reason,
            'provenance':{'source':run.get('source'),'response_sha256':call.get('response_sha256')}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(time.strftime('%H:%M:%S'),format % args, flush=True)

    def send_data(self, value, mime='application/json; charset=utf-8', status=200):
        raw=json.dumps(value,ensure_ascii=False).encode() if mime.startswith('application/json') else value
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url=urlsplit(self.path)
        args=parse_qs(url.query)
        try:
            if url.path=='/health':
                self.send_data({'ok':True,'read_only':True})
            elif url.path=='/api/catalog':
                self.send_data(catalog())
            elif url.path=='/api/sft-index':
                self.send_data(remote_query({'kind':'manifest','episode':args.get('episode',[None])[0]}))
            elif url.path=='/api/sft':
                ident=args['id'][0]
                if ident.startswith('current:'):
                    detail=remote_query({'kind':'sft','id':ident})
                    entry=next((row for row in read_json(HERE/'current-manifest-index.json',{}).get('sft',[]) if row['id']==ident),{})
                    for field in ('first_seen_step','seen_updates','artifact_effect','training_link_source'):
                        if field in entry:
                            detail[field]=entry[field]
                    self.send_data(detail)
                else:
                    self.send_data(get_sft_detail(ident))
            elif url.path=='/api/call':
                self.send_data(get_call(args['run'][0],args['call'][0]))
            elif url.path=='/api/run-image':
                run=next(r for r in run_catalog() if r['id']==args['run'][0])
                kind=args.get('kind',['reference'])[0]
                if kind not in ('reference','round1','round2'):
                    raise KeyError('Unknown image')
                artifacts=run.get('artifacts',{})
                candidate=artifacts.get(kind)
                if not candidate and kind=='reference' and run.get('reference_sha256')=='0c9a39b078a6cdd53e6849e2638782ae163c33568cb19462c5954e26bdab5678':
                    candidate=str(LINKED/'outputs/sft-research-20261121/paired-step35-preview/reference.png')
                if not candidate:
                    raise KeyError('No image recorded')
                self.send_data(Path(candidate).read_bytes(),'image/png')
            elif url.path.startswith('/api/sft/') and '/images/' in url.path:
                identity,key=url.path[len('/api/sft/'):].rsplit('/images/',1)
                raw,mime=get_sft_image(unquote(identity),key)
                self.send_data(raw,mime)
            elif url.path in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if url.path=='/' else url.path[1:]
                file=HERE/'static'/name
                mime={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[file.suffix[1:]]
                self.send_data(file.read_bytes(),mime)
            else:
                self.send_data({'error':'Not found'},status=404)
        except KeyError as error:
            self.send_data({'error':str(error)},status=404)
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            self.send_data({'error':'远端原文暂不可用','detail':str(error)},status=502)
        except (OSError,ValueError) as error:
            self.send_data({'error':type(error).__name__,'detail':str(error)},status=500)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8741)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'DSH trajectory viewer http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
