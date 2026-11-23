"""Run the frozen acceptance set for one or more checkpoints and score it."""
import argparse, json, subprocess, sys, time
from pathlib import Path

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
Q = Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38')
REL = B / 'releases/28d8a383ebf711beb256cd1a6ede8d0f8be2dc08'
ACC = B / 'checks/dsh-acceptance-20261123'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', type=Path, default=ACC / 'acceptance-set.json')
    p.add_argument('--models', required=True, help='name=path[,name=path]')
    p.add_argument('--gpu', default='6')
    p.add_argument('--port', type=int, default=32635)
    p.add_argument('--editor-endpoint', default='http://127.0.0.1:32527/v1/chat/completions')
    p.add_argument('--editor-model', type=Path, default=Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38/models/Qwen3.8-27B'))
    p.add_argument('--key-file', type=Path, required=True, help='reviewer credential, one line')
    p.add_argument('--out', type=Path, default=ACC / 'runs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cases = json.loads(a.set.read_text())['cases']
    models = [m.split('=', 1) for m in a.models.split(',') if m]
    key = a.key_file.read_text().strip()
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in models:
        for case in cases:
            dest = a.out / name / case['episode']
            if (dest / 'case/round2.drawio').exists():
                results.append({'model': name, 'episode': case['episode'], 'status': 'reused'})
                continue
            cmd = [sys.executable, '-m', 'sft.evaluate_checkpoint', '--root', str(Q), '--out', str(dest),
                   '--reference', case['reference_png'], '--model-path', path,
                   '--source-commit', 'acceptance-20261123', '--gpu', a.gpu, '--port', str(a.port),
                   '--causal-context', '--editor-model', str(a.editor_model),
                   '--editor-endpoint', a.editor_endpoint]
            if a.dry_run:
                print(' '.join(cmd))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, input=(key + '\n').encode(), cwd=str(REL),
                                  env={**__import__('os').environ, 'PYTHONPATH': str(REL)},
                                  capture_output=True)
            (dest.parent / (dest.name + '.log')).write_bytes(proc.stdout + proc.stderr)
            results.append({'model': name, 'episode': case['episode'],
                            'status': 'ok' if proc.returncode == 0 else 'failed', 'rc': proc.returncode})
            print(json.dumps(results[-1]), flush=True)
    if not a.dry_run:
        (a.out / 'run-summary.json').write_text(json.dumps({'at': time.time(), 'results': results}, indent=1) + '\n')
        print(json.dumps({'results': len(results), 'failed': sum(r['status'] == 'failed' for r in results)}))

if __name__ == '__main__':
    main()
