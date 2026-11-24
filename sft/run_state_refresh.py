"""Own the TP4 editor lifecycle for state refreshes and targeted rewrites."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .evaluate_checkpoint import ready, save, stop_model
from h20_eval import environment, start_server


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    from setproctitle import setproctitle

    setproctitle('python')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'state-prefix-model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--manifest', type=Path)
    sources.add_argument('--source-manifest', type=Path)
    parser.add_argument('--selection-plan', type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--port', type=int, default=32527)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.selection_plan):
        parser.error('--selection-plan is required with --manifest and unavailable with --source-manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    devices = '4,5,6,7'
    inventory = subprocess.check_output(['nvidia-smi', '-i', devices,
        '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    if min(int(line.split(',')[1]) for line in inventory.splitlines()) <= 49152:
        raise RuntimeError('TP4 reasoning repair needs 48 GiB free on each task GPU')
    env = environment(args.root)
    env['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK'] = '0'
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    server, log = start_server(args.root, args.output / 'editor', 'generator', 'Qwen3.8-27B',
        devices, args.port, 32768, 0.25, env, sampling='qwen38',
        extra_args=('--max-running-requests', str(args.workers), '--max-total-tokens', '65536',
                    '--cuda-graph-max-bs', str(args.workers)))
    save(args.output / 'launch.json', {'at': time.time(), 'pid': os.getpid(),
        'editor_pid': server.pid, 'devices': devices, 'command': server.args,
        'source_commit': args.source_commit, 'inventory_before': inventory, 'training_started': False})
    try:
        ready(server, args.port, timeout=900)
        mode = ['sft.rewrite_cot', '--source-manifest', str(args.source_manifest)] if args.source_manifest else [
            'sft.refresh_state', '--manifest', str(args.manifest), '--selection-plan', str(args.selection_plan)]
        command = [sys.executable, '-m', *mode, '--model', str(args.root / 'models/Qwen3.8-27B'),
            '--state-prefix-model', str(args.state_prefix_model), '--output', str(args.output / 'data'),
            '--source-commit', args.source_commit, '--workers', str(args.workers),
            '--endpoint', f'http://127.0.0.1:{args.port}/v1/chat/completions']
        save(args.output / 'refresh-command.json', command)
        result = subprocess.run(command)
        save(args.output / 'exit.json', {'at': time.time(), 'code': result.returncode})
        return result.returncode
    finally:
        stop_model(server, log)


if __name__ == '__main__':
    raise SystemExit(main())
