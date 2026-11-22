"""Serve the existing 27B rewriter on the four explicitly authorized H20 GPUs."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rollout"))
    from h20_eval import environment, gpu_inventory, start_server
    from launch import ready

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--port", type=int, default=32527)
    args = parser.parse_args()
    inventory = gpu_inventory()
    selected = [row for row in inventory if row["index"] in (4, 5, 6, 7)]
    if len(selected) != 4 or any(row["memory_used_mib"] > 32 for row in selected):
        raise RuntimeError("Authorized GPU4-7 must be idle before starting the rewriter")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"at": time.time(), "source_commit": args.source_commit, "controller_pid": os.getpid(),
             "model": "Qwen/Qwen3.8-27B", "physical_gpus": [4, 5, 6, 7], "tensor_parallel": 4,
             "context_tokens": 32768, "port": args.port, "training": False, "inventory_before": selected}
    status = args.output / "status.json"
    status.write_text(json.dumps(state, indent=2) + "\n")
    server, log = start_server(args.root, args.output / "model", "generator", "Qwen3.8-27B",
        "4,5,6,7", args.port, 32768, 0.65, environment(args.root), sampling="qwen38",
        extra_args=("--max-running-requests", "4", "--cuda-graph-max-bs", "4"))
    state.update(server_pid=server.pid, phase="loading")
    status.write_text(json.dumps(state, indent=2) + "\n")

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        ready(server, args.port, timeout=900)
        # This request contains no dataset, answers, or future trajectory state.
        body = {"model": state["model"], "messages": [{"role": "user", "content": "Reply with READY only."}],
                "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}}
        request = Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            probe = json.load(response)
        if not probe["choices"][0]["message"].get("content", "").strip():
            raise RuntimeError("Rewriter probe returned no visible answer")
        (args.output / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        state.update(at=time.time(), phase="ready", probe_completed=True)
        status.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state), flush=True)
        server.wait()
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        log.close()
        state.update(at=time.time(), phase="stopped", server_exit_code=server.returncode)
        status.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
