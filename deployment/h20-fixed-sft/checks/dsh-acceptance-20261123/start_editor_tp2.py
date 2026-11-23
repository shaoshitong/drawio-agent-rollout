"""Start the 27B context editor on a GPU subset (default 4,5 = TP2) for acceptance runs.

Reuses the frozen server helpers; does not modify the release. The editor only needs to
speak the same OpenAI chat-completions contract the trace proxy sends.
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
REL = B / 'releases/28d8a383ebf711beb256cd1a6ede8d0f8be2dc08'
sys.path.insert(0, str(REL))
sys.path.insert(0, str(REL / 'rollout'))
from h20_eval import environment, gpu_inventory, start_server
from launch import ready

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--port', type=int, default=32527)
    p.add_argument('--context-tokens', type=int, default=262144)
    p.add_argument('--concurrency', type=int, default=8)
    a = p.parse_args()
    ids = [int(x) for x in a.gpus.split(',')]
    if not ids or any(i not in (4, 5, 6, 7) for i in ids):
        raise SystemExit('refuse: GPUs must be a subset of 4-7')
    inv = gpu_inventory()
    busy = [r for r in inv if r['index'] in ids and r['memory_used_mib'] > 32]
    if busy:
        raise SystemExit(f'refuse: GPUs busy: {[(r["index"], r["memory_used_mib"]) for r in busy]}')
    a.output.mkdir(parents=True, exist_ok=False)
    state = {'at': time.time(), 'physical_gpus': ids, 'tensor_parallel': len(ids),
             'port': a.port, 'model': 'Qwen/Qwen3.8-27B', 'training': False}
    status = a.output / 'status.json'
    status.write_text(json.dumps(state, indent=2) + '\n')
    server, log = start_server(a.root, a.output / 'model', 'generator', 'Qwen3.8-27B',
        ','.join(map(str, ids)), a.port, a.context_tokens, 0.65, environment(a.root), sampling='qwen38',
        extra_args=('--max-running-requests', str(a.concurrency), '--cuda-graph-max-bs', str(a.concurrency),
                    '--max-mamba-cache-size', str(max(16, a.concurrency * 3)),
                    '--max-total-tokens', str(max(a.context_tokens, a.concurrency * 16384))))
    state.update(server_pid=server.pid, phase='loading')
    status.write_text(json.dumps(state, indent=2) + '\n')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    ready(server, a.port, timeout=900)
    state.update(at=time.time(), phase='ready')
    status.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)
    server.wait()

if __name__ == '__main__':
    main()
