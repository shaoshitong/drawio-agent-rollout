"""Replay one captured request under controlled SGLang sampling settings."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.request import ProxyHandler, Request, build_opener

from .evaluate_checkpoint import ready, save, start_model, stop_model
from direct_probe import stream_request
from trace_proxy import response_summary


def reply_summary(output):
    summary = response_summary([(output / "response.sse").read_bytes()], with_arguments=True)
    calls = list(summary["tool_calls"].values())
    for call in calls:
        try:
            json.loads(call["arguments"])
            call["arguments_parse"] = True
        except (KeyError, ValueError):
            call["arguments_parse"] = False
    text = (output / "reasoning_content.txt").read_text()
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    summary["reasoning_lines"] = sum(lines.values())
    summary["unique_reasoning_lines"] = len(lines)
    summary["max_identical_line_count"] = max(lines.values(), default=0)
    return summary


def run(args):
    args.out = args.out.absolute()
    args.model_path = args.model_path.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    used = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 128:
        raise RuntimeError("Selected diagnostic GPU is occupied")
    original_bytes = args.request.read_bytes()
    original = json.loads(original_bytes)
    save(args.out / "protocol.json", {"source_commit": args.source_commit,
        "source_request": str(args.request.resolve()), "source_request_sha256": sha256(original_bytes).hexdigest(),
        "common_max_tokens": args.max_tokens, "model_path": str(args.model_path),
        "scope": "Same archived messages, images and tools; no target, hints or tool execution. "
                 "Diagnostic output cap, not a full-rollout quality assessment."})
    server, log = start_model(args)
    results = {}
    try:
        ready(server, args.port, timeout=900)
        opener = build_opener(ProxyHandler({}))
        conditions = {"greedy": {"temperature": 0.0}, "original": {},
                      "presence15": {"presence_penalty": 1.5}}
        for name, change in conditions.items():
            output = args.out / name
            output.mkdir()
            with opener.open(Request(f"http://127.0.0.1:{args.port}/flush_cache", method="POST"),
                             timeout=30) as response:
                (output / "cache-flush.txt").write_bytes(response.read())
            body = dict(original, max_tokens=args.max_tokens, **change)
            payload = json.dumps(body, ensure_ascii=False).encode()
            (output / "request.json").write_bytes(payload)
            state = stream_request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                   payload, output, None, args.timeout)
            record = dict(reply_summary(output), stream=state, sampling_changes=change,
                          request_sha256=sha256(payload).hexdigest())
            save(output / "result.json", record)
            results[name] = record
            save(args.out / "results.json", results)
            print(json.dumps({"condition": name, "finish_reason": record.get("finish_reason"),
                              "tool_names": [call["name"] for call in record["tool_calls"].values()],
                              "usage": record.get("usage"), "stream_error": state["error"]}), flush=True)
            if not state["response_complete"] or state["error"]:
                raise RuntimeError(f"Incomplete diagnostic response: {name}")
    finally:
        stop_model(server, log)


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "request", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=32639)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    code = 1
    try:
        run(args)
        code = 0
    finally:
        (args.out / "exit-code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
