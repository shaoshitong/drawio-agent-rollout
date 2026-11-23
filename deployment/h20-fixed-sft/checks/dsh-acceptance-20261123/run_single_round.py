"""Single-round acceptance: generate one diagram from the reference, render, no reviewer.

Declared deviation: the teacher protocol is two rounds with an independent review. This
variant exists only for the case where no reviewer credential is available; results must be
labelled "single round, no reviewer".
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
REL = B / 'releases/28d8a383ebf711beb256cd1a6ede8d0f8be2dc08'
sys.path[:0] = [str(REL / 'sft'), str(REL / 'rollout')]
from sft.evaluate_checkpoint import (start_model, stop_model, prepare_case,  # noqa: E402
                                     evaluation_config, start_context_proxy)
from episode import run_generation, render, validate_xml, save  # noqa: E402
from launch import ready  # noqa: E402
from hashlib import sha256  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--gpu', default='6')
    p.add_argument('--port', type=int, default=32635)
    p.add_argument('--editor-model', type=Path, default=Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38/models/Qwen3.8-27B'))
    p.add_argument('--editor-endpoint', default='http://127.0.0.1:32527/v1/chat/completions')
    p.add_argument('--no-context', action='store_true')
    a = p.parse_args()
    a.root, a.out = Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38'), a.out.absolute()
    a.reference, a.model_path = a.reference.resolve(), a.model_path.resolve()
    a.source_commit = 'acceptance-single-round-20261123'
    a.causal_context = not a.no_context
    a.sampling_preset = 'precise-coding'
    a.generator_model = 'Qwen/Qwen3.5-9B'
    a.preserve_thinking = False
    a.out.mkdir(parents=True, exist_ok=False)
    server = log = proxy = plog = None
    try:
        server, log = start_model(a)
        ready(server, a.port, timeout=1800)
        case = prepare_case(a)
        config = evaluation_config(a)
        if a.causal_context:
            proxy, plog = start_context_proxy(a, case)
        session, _ = run_generation(config, 1, None)
        xml = case / 'round1.drawio'
        validity = validate_xml(xml)
        metadata = render(xml, case / 'round1.png', config.render_runtime)
        save(a.out / 'render1.json', {**validity, **metadata})
        result = {'completed': True, 'rounds': 1, 'reviewer': None, 'at': time.time(),
                  'candidate': str(xml), 'sha256': sha256(xml.read_bytes()).hexdigest(),
                  'generator_session': session, 'validity': validity,
                  'protocol_deviation': 'single round, no independent reviewer'}
        save(a.out / 'completed.json', result)
        print(json.dumps(result), flush=True)
    finally:
        if proxy:
            stop_model(proxy, plog)
        if server:
            stop_model(server, log)


if __name__ == '__main__':
    main()
