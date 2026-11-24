"""Evaluate a frozen checkpoint alongside its training job on task-owned GPUs."""

import argparse
from copy import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

from .evaluate_checkpoint import reconstruct, ready, save, start_model, stop_model
from .detect_loop import reasoning_loop
from h20_eval import environment, start_server
from trace_proxy import response_summary


def watch_evaluation(devices, services, done, output, active=None):
    active = active if active is not None else {"folder": output / "evaluation", "started": time.time()}
    while not done.wait(5):
        raw = subprocess.check_output(["nvidia-smi", "-i", devices,
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        free = [int(line) for line in raw.splitlines()]
        if min(free) < 12288:
            save(output / "memory-guard.json", {"at": time.time(), "free_mib": free,
                 "action": "stop evaluation services only"})
            for process, log in services:
                stop_model(process, log)
            return
        folder = active["folder"]
        raw = folder / "generator-context-requests-raw"
        paths = sorted(raw.glob("*.response.bin"))
        if not paths or paths[-1].stat().st_mtime < active["started"]:
            continue
        summary = response_summary([paths[-1].read_bytes()], with_reasoning=True)
        loop = reasoning_loop(summary["reasoning_text"])
        if loop:
            save(folder / "host-loop-outcome.json", {
                "outcome": "HOST_DETECTED_REASONING_LOOP", "at": time.time(),
                "natural_model_end": False, "completed_reconstruction": False,
                "exploratory_posthoc_rule": True,
                "training_modified": False, "raw_trace_retained": True,
                "response": str(paths[-1]), "audit": loop})
            process, log = services[-1]
            stop_model(process, log)
            return


def load_references(args):
    records = json.loads(args.cases.read_text())["cases"] if args.cases else [
        {"episode": "reference", "reference_png": str(args.reference)}]
    if not records:
        raise ValueError("The reference set is empty")
    references = []
    for record in records:
        path = Path(record["reference_png"])
        if args.cases and not path.is_absolute():
            path = args.cases.parent / path
        digest = sha256(path.read_bytes()).hexdigest()
        if args.cases and digest != record["reference_sha256"]:
            raise ValueError("Reference image hash changed: " + record["episode"])
        references.append({"episode": record["episode"], "reference_png": str(path.resolve()),
                           "reference_sha256": digest})
    return references


def run_references(args, key, references, active):
    results = []
    for record in references:
        case = copy(args)
        case.reference = Path(record["reference_png"])
        case.out = args.out / "evaluation"
        if args.cases:
            case.out /= record["episode"]
        case.out.mkdir(parents=True)
        active.update(folder=case.out, started=time.time())
        row = {"episode": record["episode"], "out": str(case.out), "completed": False,
               "fidelity_verified": False}
        try:
            reconstruct(case, key)
            reference = case.out / "case/reference.png"
            if sha256(reference.read_bytes()).hexdigest() != record["reference_sha256"]:
                raise ValueError("Evaluated reference differs from the frozen input")
            result = json.loads((case.out / "completed.json").read_text())
            if not result["completed"]:
                raise RuntimeError("Reconstruction did not complete")
            row.update(completed=result["completed"], candidate=result["candidate"])
        except Exception as error:
            row.update(error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            results.append(row)
            save(args.out / "evaluation-results.json", {"total": len(references), "cases": results,
                 "unattempted": [item["episode"] for item in references[len(results):]]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model-path", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument("--reference", type=Path)
    references.add_argument("--cases", type=Path, help="Frozen cases with reference_png and reference_sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume-generator", type=Path)
    parser.add_argument("--resume-note", default="")
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--editor-gpu", default="7")
    parser.add_argument("--port", type=int, default=32680)
    parser.add_argument("--editor-port", type=int, default=32780)
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    parser.add_argument("--editor-fraction", type=float, default=0.56)
    args = parser.parse_args()
    if args.cases and args.resume_generator:
        parser.error("--resume-generator applies to a single --reference")
    devices = [args.gpu, *args.editor_gpu.split(",")]
    if len(set(devices)) != len(devices) or not set(devices) <= {"4", "5", "6", "7"}:
        parser.error("Use distinct task-owned GPUs among 4-7")
    key = sys.stdin.readline().strip()
    if not key:
        parser.error("Reviewer credential must arrive on private stdin")
    args.out.mkdir(parents=True, exist_ok=False)
    args.causal_context = True
    args.editor_model = args.root / "models/Qwen3.8-27B"
    args.editor_endpoint = f"http://127.0.0.1:{args.editor_port}/v1/chat/completions"
    args.preprocessor_python = Path(sys.executable)
    args.generator_model = "Qwen/Qwen3.5-9B"
    args.sampling_preset = "precise-coding"
    args.preserve_thinking = False
    args.deployment = "Frozen full-parameter checkpoint loaded in BF16; shared training GPUs"
    services, done = [], Event()
    active = {"folder": args.out / "evaluation", "started": time.time()}
    watcher = Thread(target=watch_evaluation, args=(args.gpu + "," + args.editor_gpu,
                     services, done, args.out, active), daemon=True)
    watcher.start()
    try:
        references = load_references(args)
        save(args.out / "references.json", {"references": references})
        editor_environment = environment(args.root)
        # Shared TP ranks still allocate using the minimum free memory.
        editor_environment["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
        editor = start_server(args.root, args.out / "editor", "generator", "Qwen3.8-27B",
            args.editor_gpu, args.editor_port, 262144, args.editor_fraction,
            editor_environment, sampling="qwen38")
        services.append(editor)
        ready(editor[0], args.editor_port, timeout=900)
        serving = copy(args)
        serving.out = args.out / "generator-service"
        serving.out.mkdir()
        generator = start_model(serving, args.root / "envs/serving/bin/python")
        services.append(generator)
        ready(generator[0], args.port, timeout=900)
        save(args.out / "services.json", {"at": time.time(), "editor_pid": editor[0].pid,
             "generator_pid": generator[0].pid, "generator_gpu": args.gpu,
             "editor_gpu": args.editor_gpu, "training_unchanged": True,
             "source_commit": args.source_commit})
        run_references(args, key, references, active)
    except Exception as error:
        save(args.out / "failure.json", {"at": time.time(), "type": type(error).__name__,
             "message": str(error)})
        raise
    finally:
        done.set()
        watcher.join(timeout=10)
        for process, log in reversed(services):
            if not log.closed:
                stop_model(process, log)


if __name__ == "__main__":
    main()
