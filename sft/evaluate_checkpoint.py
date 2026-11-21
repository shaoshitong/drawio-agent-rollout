"""Evaluate one Qwen checkpoint with the frozen DeepSeek reviewer used in its teacher data."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "rollout")]

from episode import render, run_generation, run_review, save, validate_xml
from gateway import Provider, gateway
from launch import ready
from rollout import review_payload
from setup_sampling import create_overlay
from store import Store


def start_model(args):
    serving = args.out / "serving"
    create_overlay(args.model_path, serving, "precise-coding")
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(serving),
               "--served-model-name", "Qwen/Qwen3.5-9B", "--host", "127.0.0.1",
               "--port", str(args.port), "--tp", "1", "--dtype", "bfloat16",
               "--context-length", "262144", "--max-total-tokens", "262144",
               "--max-running-requests", "1", "--mem-fraction-static", "0.85",
               "--cuda-graph-max-bs", "1", "--max-mamba-cache-size", "16",
               "--sampling-defaults", "model", "--reasoning-parser", "qwen3",
               "--tool-call-parser", "qwen3_coder", "--preferred-sampling-params",
               json.dumps({"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0})]
    save(args.out / "server-command.json", command)
    log = (args.out / "server.log").open("wb")
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"))
    return process, log


def reconstruct(args, key):
    case = args.out / "case"
    case.mkdir()
    shutil.copyfile(args.reference, case / "reference.png")
    skill = case / ".agents/skills/drawio-reconstruction"
    skill.mkdir(parents=True)
    shutil.copyfile(HERE.parent / "rollout/reconstruction_skill.txt", skill / "SKILL.md")
    store = Store(args.out / "reviewer-archive")
    token = secrets.token_urlsafe(32)
    auth = args.out / "reviewer-auth.json"
    with os.fdopen(os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump({"headers": {"Authorization": "Bearer " + token}}, stream)
    reviewer = gateway(store, Provider(key), args.out.name, "reviewer", case, token, 12, 12)
    config = SimpleNamespace(out=args.out, runtime=args.root / "runtime",
        render_runtime=args.root / "renderer", renderer_python=args.root / "renderer/venv/bin/python",
        python_runtime_root=Path(sys.base_prefix),
        extra_runtime_mounts=list(dict.fromkeys((Path(sys.prefix), Path(sys.prefix).resolve()))),
        generator_model="Qwen/Qwen3.5-9B", reviewer_model="deepseek-flash",
        generator_url=f"http://127.0.0.1:{args.port}/v1",
        reviewer_url=f"http://127.0.0.1:{reviewer.server_port}/v1",
        generator_auth_file=None, reviewer_auth_file=auth,
        context_window=262144, reviewer_context_window=1048576, max_tokens=131072,
        reasoning_effort="high", preserve_thinking=False, qwen38_effort="xhigh",
        sampling_preset="precise-coding", reviewer_sampling_preset="model", seed=42,
        rounds=2, skill_mode="full", single_self_review=True, seeded=False,
        resume=True, resume_note="", compaction_headroom_tokens=None, turn_timeout=21600)
    protocol = {"source_commit": args.source_commit, "model_path": str(args.model_path),
        "generator": config.generator_model, "reviewer": config.reviewer_model,
        "generator_context": config.context_window, "generator_max_output": config.max_tokens,
        "reviewer_context": config.reviewer_context_window, "reviewer_max_output": 32768,
        "reviewer_thinking": False, "reviewer_training_loss": False, "rounds": 2,
        "reviews": 1, "seed": 42, "generator_sampling": "precise-coding",
        "native_compaction_unchanged": True, "preserve_thinking": False,
        "full_skill": True, "single_self_review": True, "training": False,
        "gold_available": False, "gold_feedback": False, "mcp": False,
        "reference_sha256": sha256(args.reference.read_bytes()).hexdigest(),
        "deployment": "FP32 checkpoint loaded by SGLang in BF16; numerical PEFT/export parity not yet certified",
        "reviewer_matches_teacher_model": True, "reviewer_budget_cny": 12}
    save(args.out / "protocol.json", protocol)
    generator_session = reviewer_session = None
    try:
        for number in (1, 2):
            generator_session, _ = run_generation(config, number, generator_session)
            xml = case / f"round{number}.drawio"
            validity = validate_xml(xml)
            metadata = render(xml, case / f"round{number}.png", config.render_runtime)
            save(args.out / f"render{number}.json", {**validity, **metadata})
            save(args.out / "progress.json", {"generation": number, "at": time.time(), **validity})
            if number == 1:
                digest = sha256(xml.read_bytes()).hexdigest()
                reviewer.stage = number
                reviewer_session, event = run_review(config, number, reviewer_session)
                save(case / "review1.json", review_payload(event))
                if sha256(xml.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("Reviewer changed the candidate")
        result = {"completed": True, "at": time.time(), "generator_session": generator_session,
                  "reviewer_session": reviewer_session, "fidelity_verified": False,
                  "candidate": str(case / "round2.drawio"), "reviewer_cost": store.summary()}
        save(args.out / "completed.json", result)
        print(json.dumps(result), flush=True)
    finally:
        reviewer.shutdown()
        reviewer.server_close()
        auth.unlink(missing_ok=True)


def main():
    from setproctitle import setproctitle
    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "reference", "model-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=32635)
    args = parser.parse_args()
    args.root, args.out = args.root.absolute(), args.out.absolute()
    args.model_path, args.reference = args.model_path.resolve(), args.reference.resolve()
    key = sys.stdin.readline().strip()
    if not key:
        parser.error("Provider credential must arrive on private stdin")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if int(inventory.split(",")[-1]) > 1024:
        raise RuntimeError("Selected evaluation GPU is already occupied: " + inventory)
    server, log = start_model(args)
    save(args.out / "launch.json", {"pid": os.getpid(), "server_pid": server.pid,
                                   "gpu": args.gpu, "inventory_before": inventory, "at": time.time()})
    try:
        ready(server, args.port, timeout=900)
        print(json.dumps({"server_ready": True, "at": time.time()}), flush=True)
        reconstruct(args, key)
    except Exception as error:
        save(args.out / "failure.json", {"type": type(error).__name__, "message": str(error), "at": time.time()})
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        log.close()


if __name__ == "__main__":
    main()
