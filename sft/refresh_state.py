"""Refresh shared prefixes while preserving complete actions and audited reasoning."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import time
from urllib.request import urlopen

from .action_visibility import qualify_target
from .causal_cot import answer_fields, digest, fragments
from .check_prefix_chains import Reader
from .data import load_manifest
from .history import strip_history_thinking
from .rewrite_cot import Rewriter, atomic
from .state_context import StateContext
from .state_prefix import POLICY, snapshot_before_request


class Refresher:
    def __init__(self, args, mutations):
        from transformers import AutoProcessor

        self.args, self.mutations, self.readers = args, mutations, {}
        processor = AutoProcessor.from_pretrained(args.state_prefix_model, local_files_only=True)
        self.state = StateContext(processor)
        editor_args = {**vars(args), 'state_prefix_model': None, 'rewrite_observations': False}
        self.editor = Rewriter(argparse.Namespace(**editor_args))

    def reader(self, root):
        if root not in self.readers:
            self.readers[root] = Reader(root)
        return self.readers[root]

    def refresh(self, row, folder):
        reader = self.reader(row['snapshot_root'])
        request = reader.value(row['request_ref'])
        files, snapshot = snapshot_before_request(row, reader.store)
        # Build the input before reading either the answer or its old reasoning.
        messages, state = self.state.build(request, files, lambda name: reader.store.raw(files[name]),
            row['stage'], row['request_ref'], reader.store.raw)
        if messages is None:
            raise ValueError(state['rejection'])
        assert strip_history_thinking(messages) == messages
        previous = row['reasoning_rewrite']
        overlay = self.reader(previous['root']).value(previous['ref'])
        if any(overlay[key] != row[key] for key in ('request_ref', 'response_ref')):
            raise ValueError('Previous reasoning belongs to another decision')
        response = reader.value(row['response_ref'])
        original = deepcopy(response['choices'][0]['message'])
        response['choices'][0]['message']['reasoning_content'] = overlay['current_reasoning']
        formatted = {**request, 'messages': messages}
        qualify = lambda: qualify_target(self.state.processor, formatted, response, state['files_sha256'],
            lambda name: reader.store.raw(files[name]), state['input_tokens'],
            mutation=self.mutations[row['sample_id']], include_reasoning=True)
        proof, audits = qualify(), []
        source = overlay['current_reasoning']
        if proof.get('rejection') == 'complete_target_exceeds_context':
            compact = []
            for part in fragments(source, self.editor.tokenizer, self.args.block_tokens):
                part['reasoning_source'] = previous
                value, audit = self.editor.fragment(part, ''.join(compact), folder,
                                                     row['sample_id'], row['request_ref'])
                compact.append(value + '\n' if audit['outcome'] == 'compressed' else value)
                audits.append(audit)
            response['choices'][0]['message']['reasoning_content'] = ''.join(compact)
            proof = qualify()
        if proof.get('rejection'):
            raise ValueError(proof['rejection'])
        current = response['choices'][0]['message']
        assert answer_fields(current) == answer_fields(original)
        overlay.update(current_reasoning=current['reasoning_content'],
            state_prefix={**state, 'snapshot': snapshot, 'messages': messages},
            previous_rewrite=previous)
        updated = {**row, 'reasoning_rewrite': {'root': str(self.editor.store.root.resolve()),
                    'ref': self.editor.store.put(overlay)}, 'state_prefix_policy': POLICY,
                   'state_prefix_input_tokens': state['input_tokens'],
                   'completion_tokens': proof['target_tokens'],
                   'token_count': state['input_tokens'] + proof['target_tokens'],
                   'image_count': sum(part.get('type') == 'image_url' for message in messages
                       for part in message.get('content') or [] if isinstance(part, dict))}
        result = {'row': updated, 'proof': {'sample_id': row['sample_id'], **proof,
            'input_tokens': state['input_tokens'], 'recompacted': bool(audits),
            'reasoning_changed': source != current['reasoning_content'],
            'answer_fields_sha256': digest(answer_fields(original)), 'answer_fields_unchanged': True,
            'audit_ref': self.editor.store.put(audits), 'history_thinking_messages': 0}}
        self.editor.store.metadata(row['sample_id'], result['proof'])
        return result


def refresh_shard(args, index, rows, mutations):
    args = argparse.Namespace(**vars(args))
    args.output /= 'shard' + str(index)
    args.output.mkdir(exist_ok=True)
    worker, results, failures = Refresher(args, mutations), [], []
    for row in rows:
        folder = args.output / 'episodes' / row['episode']
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (str(row['call_id']) + '.json')
        try:
            result = json.loads(path.read_text()) if path.exists() else worker.refresh(row, folder)
            if any(result['row'][key] != row[key] for key in ('request_ref', 'response_ref')):
                raise ValueError('Resumed decision source changed')
            atomic(path, result)
            results.append(result)
        except Exception as error:
            failures.append({'sample_id': row['sample_id'], 'error': type(error).__name__ + ': ' + str(error)})
        atomic(args.output / 'progress.json', {'at': time.time(), 'completed': len(results),
            'failures': len(failures), 'total': len(rows), 'training_started': False})
    return results, failures


def main():
    from setproctitle import setproctitle

    setproctitle('python')
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'selection-plan', 'model', 'state-prefix-model', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:32527/v1/chat/completions')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--block-tokens', type=int, default=768)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = (args.output / 'worker.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with urlopen(args.endpoint.split('/v1/')[0] + '/get_server_info', timeout=30) as response:
        server = json.load(response)
    if server['allow_auto_truncate'] is not False or server['tp_size'] != 4:
        raise ValueError('Reasoning repair requires TP4 and no automatic truncation')
    rows = load_manifest(args.manifest)
    selection = json.loads(args.selection_plan.read_text())
    mutations = {row['sample_id']: row['mutation'] for row in selection['decisions']}
    plan = {'source_commit': args.source_commit, 'policy': POLICY, 'rows': len(rows),
            'manifest_sha256': sha256(args.manifest.read_bytes()).hexdigest(),
            'selection_sha256': sha256(args.selection_plan.read_bytes()).hexdigest(),
            'model': str(args.model), 'state_prefix_model': str(args.state_prefix_model),
            'workers': args.workers, 'block_tokens': args.block_tokens,
            'endpoint': args.endpoint, 'training_started': False}
    saved = args.output / 'plan.json'
    if saved.exists() and json.loads(saved.read_text()) != plan:
        raise ValueError('Output belongs to a different refresh plan')
    atomic(saved, plan)
    episodes = {episode: index % args.workers for index, episode in enumerate(sorted({r['episode'] for r in rows}))}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(refresh_shard, args, index,
            [r for r in rows if episodes[r['episode']] == index], mutations) for index in range(args.workers)]
        finished = [future.result() for future in as_completed(futures)]
    results = {item['row']['sample_id']: item for items, _ in finished for item in items}
    failures = [failure for _, errors in finished for failure in errors]
    atomic(args.output / 'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} decisions incomplete; verified rows are resumable')
    with (args.output / 'refreshed.jsonl').open('w') as stream:
        stream.writelines(json.dumps(results[row['sample_id']]['row']) + '\n' for row in rows)
    atomic(args.output / 'completed.json', {**plan, 'at': time.time(), 'failures': 0,
        'recompacted': sum(item['proof']['recompacted'] for item in results.values()),
        'answer_fields_unchanged': True, 'model_quality_verified': False})


if __name__ == '__main__':
    main()
