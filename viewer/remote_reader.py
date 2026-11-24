"""Bounded, read-only evidence queries, executed on H20 through existing SSH."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
MANIFEST = B / 'checks/dsh-midflight-20261123/curriculum-final.jsonl'
RUNS = B / 'runs/causal-cot-20261122'


def checked(path):
    p = Path(path).resolve()
    if not p.is_relative_to(B.resolve()):
        raise ValueError('Path outside the evidence root')
    return p


def read_json(path, limit=16 * 1024**2):
    p = checked(path)
    if p.stat().st_size > limit:
        raise ValueError('Evidence object exceeds this bounded query')
    return json.loads(p.read_text())


def clean(value):
    if isinstance(value, dict):
        if value.get('type') == 'image_url':
            image = value.get('image_url') or {}
            url = image.get('url') if isinstance(image, dict) else image
            return {'type': 'image_reference', 'label': '图片保留在远端', 'sha256': sha256(str(url).encode()).hexdigest()}
        return {k: clean(v) for k, v in value.items() if not any(s in k.lower() for s in ('api_key', 'authorization', 'access_token'))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and value.startswith('data:image/'):
        return '[image payload retained remotely]'
    return value


def source_imports():
    candidates = [B/'checks/observation-state-v3-20261124/source-d701a46', B/'checks/observation-state-v3-20261124/source']
    for root in candidates:
        if (root/'sft/data.py').is_file():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root/'rollout'))
            return root
    raise FileNotFoundError('Verified read-only source checkout unavailable')


def manifest_index(episode=None):
    rows = [json.loads(line) for line in MANIFEST.open() if line.strip()]
    by_episode = {}
    for row in rows:
        group = by_episode.setdefault(row['episode'], [])
        group.append(row)
    exposures = {}
    audit = B/'checks/stopped218-supervised-actions-20261124/decisions.jsonl'
    first_episodes = []
    if audit.is_file():
        for line in audit.open():
            item = json.loads(line)
            exposures[item['sample_id']] = item
            ep = item['sample_id'].split(':',1)[0]
            if ep not in first_episodes:
                first_episodes.append(ep)
    if episode is not None and episode not in by_episode:
        raise KeyError('Unknown episode')
    selected = []
    # Default episodes come from the actual supervised curriculum prefix.
    default_episodes = [ep for ep in first_episodes if ep in by_episode][:4]
    groups = [(episode, by_episode[episode])] if episode in by_episode else [(ep,by_episode[ep]) for ep in default_episodes]
    if not groups:
        groups = sorted(by_episode.items())[:4]
    for episode, group in groups:
        for row in group:
            exposure = exposures.get(row['sample_id'],{})
            update = exposure.get('exposure')
            selected.append({
                'id': 'current:' + row['sample_id'], 'sample_id': row['sample_id'],
                'title': f"{episode[:10]} · 决策 {row.get('decision_order', row.get('call_id'))}",
                'category': exposure.get('category','当前冻结训练集'), 'episode': episode, 'stage': row.get('stage'),
                'first_seen_step': update//32+1 if update is not None else None, 'seen_updates': [update//32+1] if update is not None else [],
                'training_link_source': 'verified_sampler_exposure_divided_by_effective_batch_32',
                'artifact_effect': exposure.get('artifact_effect'),
                'original_tokens': None, 'rewritten_tokens': row.get('completion_tokens'),
                'split': row.get('split'), 'source': str(MANIFEST),
                'decision_order': row.get('decision_order'),
            })
    return {'total_decisions': len(rows), 'total_episodes': len(by_episode), 'manifest_sha256': sha256(MANIFEST.read_bytes()).hexdigest(),
            'episodes': [{'id': ep, 'count': len(group), 'split': group[0].get('split'), 'components': group[0].get('components')} for ep,group in sorted(by_episode.items())],
            'selected_episodes': len(groups), 'sft': selected[:1600], 'catalog_limit': 1600}


def side(message):
    message = clean(message)
    return {'messages': [], 'content_available': True, 'tool_calls_available': True, 'reasoning': message.get('reasoning_content') or '',
            'content': message.get('content') or '', 'tool_calls': message.get('tool_calls') or []}


def sft_detail(ident):
    if not ident.startswith('current:'):
        raise ValueError('Unsupported dataset ID')
    key = ident[8:]
    row = next((json.loads(line) for line in MANIFEST.open() if json.loads(line)['sample_id'] == key), None)
    if row is None:
        raise KeyError(key)
    source_imports()
    from sft.data import open_store
    from sft.export_sft import materialize
    store = open_store(MANIFEST.parent / row['snapshot_root'])
    response = store.get(row['response_ref'])
    message = response['choices'][0]['message']
    request = store.get(row['request_ref'])
    rewrite = row.get('reasoning_rewrite')
    overlay = open_store(rewrite['root']).get(rewrite['ref']) if rewrite else None
    example = materialize(store, row, decode_images=False, rewrite=overlay)
    orig = side(message)
    orig['messages'] = clean(request.get('messages', []))
    orig['input_tokens'] = (response.get('usage') or {}).get('prompt_tokens')
    rewritten = side(example['completion'][-1])
    rewritten['messages'] = clean(example['prompt'])
    rewritten['input_tokens'] = row.get('state_prefix_input_tokens')
    return {'id': ident, 'title': key, 'category': '当前冻结训练集', 'episode': row['episode'], 'stage': row.get('stage'),
            'original': orig, 'rewritten': rewritten, 'description': '同一教师决策。左为存档原请求与原目标，右为冻结 SFT 清单实际使用的前缀与监督目标。',
            'provenance': {'manifest': str(MANIFEST), 'manifest_sha256': sha256(MANIFEST.read_bytes()).hexdigest(),
                'sample_id': key, 'snapshot_root': row['snapshot_root'], 'request_ref': row['request_ref'], 'response_ref': row['response_ref'],
                'rewrite': rewrite, 'targets_used_to_build_input': False, 'source': str(MANIFEST)}, 'images': [],
            'original_tokens': (response.get('usage') or {}).get('completion_tokens'), 'rewritten_tokens': row.get('completion_tokens')}


def call_detail(arg):
    request_path, response_path = checked(arg['request']), checked(arg['response'])
    source_imports()
    from trace_proxy import response_summary
    raw = response_path.read_bytes()
    if len(raw) > 12*1024**2:
        raise ValueError('Response exceeds the bounded detail limit')
    request = read_json(request_path)
    summary = response_summary([raw], with_arguments=True, with_reasoning=True)
    tool_calls = [{'id': tool.get('id'), 'function': {'name': tool['name'], 'arguments': tool.get('arguments','')}}
                  for tool in summary.get('tool_calls',{}).values()]
    results = []
    next_request = arg.get('next_request')
    if next_request:
        next_value = read_json(next_request)
        ids = {call['id'] for call in tool_calls if call['id']}
        results = [clean(m) for m in next_value.get('messages',[]) if m.get('role')=='tool' and m.get('tool_call_id') in ids]
    return {'request_messages': clean(request.get('messages', [])), 'reasoning': summary.get('reasoning_text',''),
            'content': summary.get('content_text',''), 'tool_calls': tool_calls, 'tool_results': results,
            'complete': b'data: [DONE]' in raw, 'partial_reason': None if b'data: [DONE]' in raw else '原始 SSE 流未完整结束，保留已捕获的内容。',
            'provenance': {'request': str(request_path), 'response': str(response_path), 'request_sha256': sha256(request_path.read_bytes()).hexdigest(),
                'response_sha256': sha256(raw).hexdigest(), 'model': request.get('model'),
                'sampling': {k: request.get(k) for k in ('temperature','top_p','top_k','max_tokens','seed') if k in request}}}


def main(args):
    kind = args['kind']
    if kind == 'manifest':
        return manifest_index(args.get('episode'))
    if kind == 'sft':
        return sft_detail(args['id'])
    if kind == 'call':
        return call_detail(args)
    raise ValueError('Unknown read operation')


if __name__ == '__main__':
    try:
        print(json.dumps(main(PAYLOAD), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'error': type(error).__name__, 'message': str(error)}, ensure_ascii=False))
        sys.exit(1)
