"""Evaluate checkpoint variants on one reference with the same frozen harness."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


def run(args):
    key = sys.stdin.readline().strip()
    if not key:
        raise ValueError("Reviewer credential must arrive on private stdin")
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"source_commit": args.source_commit,
            "reference_sha256": sha256(args.reference.read_bytes()).hexdigest(),
            "models": args.model, "context_window": 262144, "max_output": 131072,
            "generator_sampling": "precise-coding", "preserve_thinking": False,
            "rounds": 2, "reviews": 1, "reviewer": "deepseek-flash",
            "full_skill": True, "single_self_review": True,
            "gold_feedback": False, "mcp": False, "training_modified": False}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for specification in args.model:
        name, model = specification.split("=", 1)
        output = args.out / name
        command = [sys.executable, "-m", "sft.evaluate_checkpoint", "--root", str(args.root),
                   "--out", str(output), "--reference", str(args.reference),
                   "--model-path", model, "--source-commit", args.source_commit,
                   "--gpu", args.gpu, "--port", str(args.port)]
        with (args.out / (name + ".log")).open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            process.stdin.write((key + "\n").encode())
            process.stdin.close()
            (args.out / "active.json").write_text(json.dumps({"name": name, "pid": process.pid,
                "at": time.time(), "command": command}) + "\n")
            code = process.wait()
        results.append({"name": name, "exit_code": code, "at": time.time()})
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "exit-code").write_text(str(int(any(row["exit_code"] for row in results))) + "\n")


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="Report name=checkpoint path")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=32635)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
