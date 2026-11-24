"""Continue a training seed with real DSH and a local TP4 Qwen3.8 teacher."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .evaluate_checkpoint import ready, reconstruct, save, stop_model
from h20_eval import environment, start_server


def verify_seed(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["split"] != "train" or receipt["gold_available"] or receipt["archived_current_response_read"]:
        raise ValueError("A gold-blind training seed is required")
    files = {str(file.relative_to(path / "case")): sha256(file.read_bytes()).hexdigest()
             for file in (path / "case").rglob("*") if file.is_file()}
    if files != receipt["copied_files"] or sha256((path / "past.json").read_bytes()).hexdigest() != receipt["past_sha256"]:
        raise ValueError("Seed files differ from their archived receipt")
    return receipt


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "build-seed", "state-prefix-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--port", type=int, default=32697)
    parser.add_argument("--fresh", action="store_true", help="Use only the verified training reference")
    parser.add_argument("--verified-draft-from", type=Path, help="Review and revise an archived native write")
    args = parser.parse_args()
    receipt = verify_seed(args.build_seed)
    key = sys.stdin.readline().strip()
    if not key:
        parser.error("Pass the existing reviewer credential on private stdin")
    devices = "4,5,6,7"
    memory = subprocess.check_output(["nvidia-smi", "-i", devices,
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if min(map(int, memory.splitlines())) <= 49152:
        raise RuntimeError("TP4 requires at least 48 GiB free on each authorized GPU")
    args.out.mkdir(parents=True)
    args.reference = args.build_seed / "case/reference.png"
    inherit = not (args.fresh or args.verified_draft_from)
    if not inherit:
        args.build_seed = None
    args.model_path = args.root / "models/Qwen3.8-27B"
    args.generator_model, args.sampling_preset = "Qwen/Qwen3.8-27B", "qwen38"
    args.causal_context = True
    args.preprocessor_python = Path(sys.executable)
    args.deployment = "TP4 Qwen3.8 teacher with verified draft handoff" if args.verified_draft_from else (
        "TP4 Qwen3.8 teacher with inherited pre-construction observations" if inherit else
        "TP4 Qwen3.8 teacher from reference only")
    env = environment(args.root)
    env.update(SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="0", DSH_MATCH_TRAINING_IMAGES="1")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    server, log = start_server(args.root, args.out / "teacher", "generator", "Qwen3.8-27B",
        devices, args.port, 32768, .25, env, sampling=args.sampling_preset, preserve_tool_schema=True)
    save(args.out / "launch.json", {"at": time.time(), "pid": os.getpid(), "server_pid": server.pid,
        "seed_id": receipt["seed_id"], "inherited_observations": inherit,
        "devices": devices, "source_commit": args.source_commit,
        "supervision_admitted": False, "training_started": False})
    try:
        ready(server, args.port, timeout=900)
        args.out = args.out / "evaluation"
        args.out.mkdir()
        reconstruct(args, key)
    except BaseException as error:
        save(Path(log.name).parent.parent / "failure.json",
             {"type": type(error).__name__, "message": str(error), "at": time.time()})
        raise
    finally:
        stop_model(server, log)


if __name__ == "__main__":
    main()
