"""Capture exact early DSH requests under the existing checkpoint-evaluation settings."""

import argparse
from hashlib import sha256
from http.server import ThreadingHTTPServer
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit

from .evaluate_checkpoint import evaluation_config, prepare_case, ready, save, start_model, stop_model
from rollout import run_turn
from trace_proxy import Proxy


def run(args):
    args.root, args.out = args.root.absolute(), args.out.absolute()
    args.reference, args.model_path = args.reference.resolve(), args.model_path.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    used = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 128:
        raise RuntimeError("Selected diagnostic GPU is occupied")
    case = prepare_case(args)
    config = evaluation_config(args)
    server, log = start_model(args)
    proxy = None
    save(args.out / "launch.json", {"pid": os.getpid(), "server_pid": server.pid,
                                   "at": time.time(), "source_commit": args.source_commit})
    try:
        ready(server, args.port, timeout=900)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        proxy.upstream, proxy.audit = urlsplit(config.generator_url), args.out / "generator-api.jsonl"
        proxy.capture_raw, proxy.request_limit = True, args.request_limit
        proxy.request_counter, proxy.limit_reached = count(1), False
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        config.generator_url = f"http://127.0.0.1:{proxy.server_port}/v1"
        save(args.out / "protocol.json", {"source_commit": args.source_commit,
            "reference_sha256": sha256(args.reference.read_bytes()).hexdigest(),
            "skill_sha256": sha256((case / ".agents/skills/drawio-reconstruction/SKILL.md").read_bytes()).hexdigest(),
            "context_window": config.context_window, "max_tokens": config.max_tokens,
            "sampling_preset": config.sampling_preset, "seed": config.seed,
            "preserve_thinking": config.preserve_thinking, "request_limit": args.request_limit,
            "scope": "Only original first generator stage; same settings, full API capture, no recovery prompt or reviewer call.",
            "gold_available": False, "mcp": False, "training": False})
        failure = None
        try:
            run_turn(config, "generator", 1, None)
        except RuntimeError as error:
            if not proxy.limit_reached:
                raise
            failure = str(error)
        result = {"at": time.time(), "host_censored": proxy.limit_reached,
            "native_error_after_host_limit": failure, "candidate_exists": (case / "round1.drawio").exists(),
            "full_rollout_completed": False, "quality_score_assigned": False}
        save(args.out / "capture-result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        if proxy:
            proxy.shutdown()
        stop_model(server, log)
        if proxy:
            proxy.server_close()


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "reference", "model-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=32637)
    parser.add_argument("--request-limit", type=int, default=60)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
