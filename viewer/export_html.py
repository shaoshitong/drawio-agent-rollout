#!/usr/bin/env python3
"""Export the read-only viewer as one self-contained, network-free HTML artifact.

Only selected, human-readable evidence is embedded. Remote raw SSE streams,
SQLite stores and model assets are never staged locally or changed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote

import server
from data_adapter import get_sft_detail, get_sft_image

HERE = Path(__file__).resolve().parent


def image_uri(raw, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')


def build(output):
    catalog = server.catalog()
    selected = server.remote_query({'kind':'offline_selection'})
    historical = [row for row in catalog['sft'] if not row.get('current_curriculum')]
    catalog['sft'] = selected['sft'] + historical
    payload = {'catalog':catalog, 'sft':{}, 'calls':{}, 'images':{}, 'message_pool':[],
               'exported_at':datetime.now(timezone.utc).isoformat()}
    pool_index = {}
    def intern(messages, source_pool=None):
        refs = []
        for message in messages:
            if isinstance(message,int):
                message = source_pool[message]
            packed = json.dumps(message, ensure_ascii=False, sort_keys=True)
            if packed not in pool_index:
                pool_index[packed] = len(payload['message_pool'])
                payload['message_pool'].append(message)
            refs.append(pool_index[packed])
        return refs
    for row in historical:
        detail = get_sft_detail(row['id'])
        for side in ('original','rewritten'):
            detail[side]['messages'] = intern(detail[side].get('messages',[]))
        payload['sft'][row['id']] = detail
        for image in detail.get('images',[]):
            raw,mime = get_sft_image(row['id'],image['id'])
            payload['images'][image['url']] = image_uri(raw,mime)
    print(f'local evidence: {len(historical)} SFT pairs and {len(payload["images"])} image references',flush=True)
    jobs = []
    current = selected['sft']
    for start in range(0,len(current),5):
        jobs.append([{'kind':'sft','id':row['id'],'key':row['id']} for row in current[start:start+5]])
    capture_queries, aliases, signatures = {}, {}, set()
    for run in catalog['runs']:
        calls = run.get('calls',[])
        for call in calls:
            signature = (call.get('request_sha256'),call.get('response_sha256'))
            if not all(signature):
                raise ValueError('Captured source has no request/response checksum')
            signatures.add(signature)
            # Results belong to the next request in THIS run. Resumed copies
            # may share output bytes while having different execution results.
            key = run['id']+'#'+str(call['index'])
            query = {'kind':'call','key':key,'request':call['request'],'response':call['response']}
            following = next((c for c in calls if c['index']==call['index']+1),None)
            if following:
                query['next_request'] = following['request']
            capture_queries[key] = query
            aliases[key] = (run['id'],call)
    unique = list(capture_queries.values())
    for start in range(0,len(unique),16):
        jobs.append(unique[start:start+16])
    results, errors, total = {}, [], 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(server.remote_query,{'kind':'offline_batch','items':items}):len(items) for items in jobs}
        for future in as_completed(pending):
            batch = future.result()
            errors.extend(batch.get('errors',[]))
            for item in batch['details']:
                detail,key = item['detail'],item['key']
                if key.startswith('current:'):
                    if detail['provenance']['manifest_sha256'] != catalog['meta']['dataset']['manifest_sha256']:
                        raise ValueError('Frozen SFT manifest changed during export')
                    for side in ('original','rewritten'):
                        detail[side]['messages'] = intern(detail[side].get('messages',[]),batch['message_pool'])
                    payload['sft'][key] = detail
                else:
                    actual = detail['provenance']
                    call = aliases[key][1]
                    if (call['request_sha256'],call['response_sha256']) != (actual['request_sha256'],actual['response_sha256']):
                        raise ValueError('Captured call checksum differs from catalog')
                    detail['request_messages'] = intern(detail.get('request_messages',[]),batch['message_pool'])
                    results[key] = detail
                total += 1
            print(f'embedded evidence verified: {total}/{len(current)+len(unique)} unique objects; {len(errors)} errors',flush=True)
    if errors:
        raise RuntimeError('Cannot deliver complete selected evidence: '+json.dumps(errors,ensure_ascii=False)[:1500])
    for key, (run_id,call) in aliases.items():
        detail = dict(results[key],call=call)
        payload['calls'].setdefault(run_id,{})[str(call['index'])] = detail
    for run in catalog['runs']:
        for kind,path in run.get('artifacts',{}).items():
            if kind not in ('reference','round1','round2') or not isinstance(path,str):
                continue
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError('Missing comparison image '+str(source))
            mime = mimetypes.guess_type(str(source))[0] or 'image/png'
            payload['images']['/api/run-image?run='+quote(run['id'],safe='')+'&kind='+kind] = image_uri(source.read_bytes(),mime)
    episodes = catalog['meta']['dataset'].get('episodes',[])
    included = {row['episode'] for row in current}
    catalog['meta']['dataset']['episodes'] = [ep for ep in episodes if ep['id'] in included]
    catalog['meta']['notes'] = [note for note in catalog['meta']['notes'] if '首批索引' not in note]
    catalog['meta']['notes'].append(f'此离线 HTML 按每50次更新和动作类别选择当前冻结清单{len(current)}个决策，并包含原有{len(historical)}个历史导出；仅用于逐例检查，不能推导总体质量。')
    catalog['meta']['offline_selection'] = selected['selection']
    payload['coverage'] = {
        'sft_pairs':len(payload['sft']), 'current_pairs':len(current),'historical_pairs':len(historical),
        'captured_calls':sum(len(run.get('calls',[])) for run in catalog['runs']),
        'unique_captures':len(signatures),
        'full_call_outputs':sum(len(calls) for calls in payload['calls'].values()),
        'summary_calls':0,'request_contexts':sum(len(calls) for calls in payload['calls'].values()),
        'loss_points':len(catalog['meta'].get('loss',[])),
        'aggregate_runs':sum(not run.get('calls') for run in catalog['runs']),
        'incomplete_streams':sum(not d.get('complete') for calls in payload['calls'].values() for d in calls.values()),
        'embedded_images':len(payload['images']),
        'source_decisions':catalog['meta']['dataset']['total_decisions'],
        'source_episodes':catalog['meta']['dataset']['total_episodes'],
    }
    write_html(payload,output)


def write_html(payload,output):
    """Repackage embedded evidence without revisiting remote sources."""
    # Escape HTML-significant characters in inert JSON, including literal </script>.
    packed = json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    style = (HERE/'static/style.css').read_text()
    script = (HERE/'static/app.js').read_text().replace('</script','<\\/script')
    shell = (HERE/'static/index.html').read_text()
    shell = shell.replace('<link rel="stylesheet" href="/style.css">','<style>'+style+'</style>')
    shell = shell.replace('<script src="/app.js" defer></script>','')
    shell = shell.replace('<title>Trajectory Lab · Qwen 9B</title>','<title>Trajectory Lab · 离线对照快照</title>')
    csp = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; font-src 'none'; base-uri 'none'; form-action 'none'"
    shell = shell.replace('<meta charset="utf-8">','<meta charset="utf-8">\n  <meta http-equiv="Content-Security-Policy" content="'+csp+'">')
    shell = shell.replace('</body>','<script type="application/json" id="offline-data">'+packed+'</script>\n<script>window.DSH_OFFLINE=JSON.parse(document.getElementById("offline-data").textContent);</script>\n<script>'+script+'</script>\n</body>')
    if 'const offline' not in script and 'DSH_OFFLINE' not in script:
        raise RuntimeError('Viewer does not yet contain offline runtime support')
    raw = shell.encode('utf-8')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = output.with_suffix('.html.partial')
    staging.write_bytes(raw)
    staging.replace(output)
    receipt = {'exported_at':payload['exported_at'],'file':str(output),'bytes':len(raw),'sha256':sha256(raw).hexdigest(),
               'coverage':payload['coverage'],'manifest_sha256':payload['catalog']['meta']['dataset']['manifest_sha256'],
               'network_required':False,'remote_mutations':False,'raw_capture_files_copied':False}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh-shell',action='store_true',help='Reuse the existing embedded evidence, without remote access')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.refresh_shell:
        match = re.search(r'<script type="application/json" id="offline-data">(.*?)</script>',output.read_text(),re.S)
        if not match:
            raise ValueError('No self-contained evidence snapshot found')
        write_html(json.loads(match.group(1)),output)
    else:
        build(output)
