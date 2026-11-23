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


def start_model(args, python=None):
    serving = args.out / "serving"
    create_overlay(args.model_path, serving, getattr(args, "sampling_preset", "precise-coding"),
                   preserve_tool_schema=True)
    context = "32768" if getattr(args, "causal_context", False) else "262144"
    command = [str(python or sys.executable), "-m", "sglang.launch_server", "--model-path", str(serving),
               "--served-model-name", getattr(args, "generator_model", "Qwen/Qwen3.5-9B"),
               "--host", getattr(args, "host", "127.0.0.1"),
               "--port", str(args.port), "--tp", "1", "--dtype", "bfloat16",
               "--context-length", context, "--max-total-tokens", context,
               "--max-running-requests", "1", "--mem-fraction-static", str(getattr(args, "mem_fraction_static", 0.85)),
               "--cuda-graph-max-bs", "1", "--max-mamba-cache-size", "16",
               "--sampling-defaults", "model", "--reasoning-parser", "qwen3",
               "--tool-call-parser", "qwen3_coder", "--preferred-sampling-params",
               json.dumps({"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0})]
    save(args.out / "server-command.json", command)
    log = (args.out / "server.log").open("wb")
    environment = dict(os.environ, PATH=str(Path(python or sys.executable).parent) + os.pathsep + os.environ["PATH"],
        CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    if getattr(args, "generator_model", "") == "Qwen/Qwen3.8-27B":
        environment["DSH_QWEN38_PARSER_COMPAT"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join((str(HERE.parent / "rollout/h20_bootstrap"),
            str(HERE.parent / "rollout"), environment.get("PYTHONPATH", "")))
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True,
                               env=environment)
    return process, log


def start_context_proxy(args, case):
    python = args.preprocessor_python or Path(sys.executable)
    command = [str(python), str(HERE.parent / "rollout/trace_proxy.py"),
        "--port", str(args.port + 101), "--upstream", f"http://127.0.0.1:{args.port}",
        "--audit", str(args.out / "generator-context-requests.jsonl"), "--capture-raw",
        "--context-cache", str(args.out / "context"), "--context-model", str(args.editor_model),
        "--context-source-commit", args.source_commit, "--context-endpoint", args.editor_endpoint,
        "--state-prefix-model", str(args.model_path), "--state-case", str(case)]
    environment = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ["PATH"],
                       CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    log = (args.out / "context-proxy.log").open("wb")
    proxy = subprocess.Popen(command, stdout=log, stderr=log, env=environment, start_new_session=True)
    try:
        ready(proxy, args.port + 101, timeout=120)
    except Exception:
        stop_model(proxy, log)
        raise
    save(args.out / "context-proxy-command.json", command)
    return proxy, log


def stop_model(server, log):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    log.close()


def prepare_case(args):
    case = args.out / "case"
    case.mkdir()
    shutil.copyfile(args.reference, case / "reference.png")
    skill = case / ".agents/skills/drawio-reconstruction"
    skill.mkdir(parents=True)
    shutil.copyfile(HERE.parent / "rollout/reconstruction_skill.txt", skill / "SKILL.md")
    return case


def evaluation_config(args, reviewer_url="", reviewer_auth=None):
    causal = getattr(args, "causal_context", False)
    return SimpleNamespace(out=args.out, runtime=args.root / "runtime",
        render_runtime=args.root / "renderer", renderer_python=args.root / "renderer/venv/bin/python",
        python_runtime_root=Path(sys.base_prefix),
        extra_runtime_mounts=list(dict.fromkeys((Path(sys.prefix), Path(sys.prefix).resolve()))),
        generator_model=getattr(args, "generator_model", "Qwen/Qwen3.5-9B"), reviewer_model="deepseek-flash",
        generator_url=f"http://127.0.0.1:{args.port + (101 if causal else 0)}/v1",
        reviewer_url=reviewer_url, generator_auth_file=getattr(args, "generator_auth_file", None),
        reviewer_auth_file=reviewer_auth,
        context_window=262144, reviewer_context_window=1048576, max_tokens=32768 if causal else 131072,
        reasoning_effort=getattr(args, "reasoning_effort", "high"),
        preserve_thinking=getattr(args, "preserve_thinking", False), qwen38_effort="xhigh",
        sampling_preset=getattr(args, "sampling_preset", "precise-coding"), reviewer_sampling_preset="model", seed=42,
        rounds=2, skill_mode="full", single_self_review=True, seeded=False, preserve_tool_schema=True,
        resume=True, resume_note="", compaction_headroom_tokens=None, no_auto_compaction=causal, turn_timeout=21600)


def reconstruct(args, key):
    case = prepare_case(args)
    store = Store(args.out / "reviewer-archive")
    token = secrets.token_urlsafe(32)
    auth = args.out / "reviewer-auth.json"
    with os.fdopen(os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump({"headers": {"Authorization": "Bearer " + token}}, stream)
    reviewer = gateway(store, Provider(key), args.out.name, "reviewer", case, token, 12, 12)
    config = evaluation_config(args, f"http://127.0.0.1:{reviewer.server_port}/v1", auth)
    protocol = {"source_commit": args.source_commit, "model_path": str(args.model_path),
        "generator": config.generator_model, "reviewer": config.reviewer_model,
        "generator_context": config.context_window, "generator_max_output": config.max_tokens,
        "reviewer_context": config.reviewer_context_window, "reviewer_max_output": 32768,
        "reviewer_thinking": False, "reviewer_training_loss": False, "rounds": 2,
        "reviews": 1, "seed": 42, "generator_sampling": config.sampling_preset,
        "native_compaction_unchanged": not getattr(args, "causal_context", False), "preserve_thinking": config.preserve_thinking,
        "full_skill": True, "single_self_review": True, "training": False,
        "gold_available": False, "gold_feedback": False, "mcp": False,
        "reference_sha256": sha256(args.reference.read_bytes()).hexdigest(),
        "deployment": getattr(args, "deployment", "Checkpoint served in BF16; numerical PEFT/export parity not yet certified"),
        "reviewer_matches_teacher_model": True, "reviewer_budget_cny": 12}
    protocol["preserve_raw_tool_schema"] = True
    if getattr(args, "causal_context", False):
        protocol.update(generator_context=32768, generator_raw_history_window=config.context_window,
                        state_prefix_policy="own-state-prefix-v2", generator_auto_compaction=False)
    save(args.out / "protocol.json", protocol)
    generator_session = reviewer_session = None
    context_proxy = context_log = None
    try:
        if getattr(args, "causal_context", False):
            context_proxy, context_log = start_context_proxy(args, case)
        for number in (1, 2):
            if getattr(args, "generator_service", None):
                args.generator_service.stage = number
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
        if context_proxy:
            stop_model(context_proxy, context_log)
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
    parser.add_argument("--external-generator", action="store_true")
    parser.add_argument("--generator-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sampling-preset", default="precise-coding")
    parser.add_argument("--preserve-thinking", action="store_true")
    parser.add_argument("--causal-context", action="store_true")
    parser.add_argument("--editor-model", type=Path)
    parser.add_argument("--editor-endpoint", default="http://127.0.0.1:32527/v1/chat/completions")
    parser.add_argument("--preprocessor-python", type=Path)
    args = parser.parse_args()
    if args.causal_context and not args.editor_model:
        parser.error("causal-context requires editor-model")
    args.root, args.out = args.root.absolute(), args.out.absolute()
    args.model_path, args.reference = args.model_path.resolve(), args.reference.resolve()
    key = sys.stdin.readline().strip()
    if not key:
        parser.error("Provider credential must arrive on private stdin")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    server = log = None
    if not args.external_generator:
        inventory = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
            "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
        if int(inventory.split(",")[-1]) > 1024:
            raise RuntimeError("Selected evaluation GPU is already occupied: " + inventory)
        server, log = start_model(args)
    save(args.out / "launch.json", {"pid": os.getpid(), "server_pid": server.pid if server else None,
                                   "gpu": args.gpu, "external_generator": args.external_generator,
                                   "at": time.time()})
    try:
        if server:
            ready(server, args.port, timeout=900)
        print(json.dumps({"server_ready": True, "at": time.time()}), flush=True)
        reconstruct(args, key)
    except Exception as error:
        save(args.out / "failure.json", {"type": type(error).__name__, "message": str(error), "at": time.time()})
        raise
    finally:
        if server:
            stop_model(server, log)


if __name__ == "__main__":
    main()
