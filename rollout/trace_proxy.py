"""Transparent local inference audit proxy; no reward or gold endpoint exists."""

import argparse
import base64
from hashlib import sha256
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit


def images(value):
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            url = value["image_url"]["url"]
            if url.startswith("data:"):
                payload = base64.b64decode(url.split(",", 1)[1])
                yield {"bytes": len(payload), "sha256": sha256(payload).hexdigest(), "inline": True}
            else:
                yield {"url_sha256": sha256(url.encode()).hexdigest(), "inline": False}
        for item in value.values():
            yield from images(item)
    elif isinstance(value, list):
        for item in value:
            yield from images(item)


def response_summary(chunks, with_arguments=False, with_reasoning=False):
    summary = {"content_characters": 0, "reasoning_characters": 0, "tool_calls": {}}
    reasoning = []
    for line in b"".join(chunks).splitlines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        try:
            event = json.loads(line[6:])
        except ValueError:
            summary["incomplete_sse_records"] = summary.get("incomplete_sse_records", 0) + 1
            continue
        if event.get("usage"):
            summary["usage"] = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("finish_reason"):
                summary["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            summary["content_characters"] += len(delta.get("content") or "")
            summary["reasoning_characters"] += len(delta.get("reasoning_content") or "")
            if with_reasoning:
                reasoning.append(delta.get("reasoning_content") or "")
            for call in delta.get("tool_calls") or []:
                tool = summary["tool_calls"].setdefault(
                    str(call.get("index", 0)), {"name": "", "argument_characters": 0}
                )
                function = call.get("function", {})
                tool["name"] += function.get("name") or ""
                tool["argument_characters"] += len(function.get("arguments") or "")
                if with_arguments:
                    tool["arguments"] = tool.get("arguments", "") + (function.get("arguments") or "")
                    if call.get("id"):
                        tool["id"] = call["id"]
    if with_reasoning:
        summary["reasoning_text"] = "".join(reasoning)
    return summary


class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        data = json.loads(body)
        state_proof = None
        if getattr(self.server, "state_policy", None):
            from sft.state_prefix import CONTEXT_BUDGET
            from sft.store import snapshot

            store = self.server.state_store
            reference = store.put(data)
            snapshot_ref = snapshot(store, self.server.state_case)
            files = store.get(snapshot_ref)
            stage = 2 if "review1.json" in files else 1
            try:
                messages, state_proof = self.server.state_policy.build(data, files,
                    lambda name: store.raw(files[name]), stage, reference, store.raw)
            except Exception as error:
                with self.server.audit.open("a") as log:
                    log.write(json.dumps({"state_context_error": type(error).__name__, "at": time.time()}) + "\n")
                self.send_error(502, "State preprocessing failed")
                return
            state_proof["snapshot_ref"] = snapshot_ref
            store.metadata("last_state_prefix", state_proof)
            if messages is None:
                self.send_error(413, "Protected own state exceeds the fixed input budget")
                return
            data["messages"] = messages
            data["max_tokens"] = min(data.get("max_tokens", CONTEXT_BUDGET), CONTEXT_BUDGET - state_proof["input_tokens"] - 1)
            body = json.dumps(data, ensure_ascii=False).encode()
        elif self.server.strip_history or self.server.context_policy:
            from sft.history import strip_history_thinking
            from sft.causal_cot import digest

            data["messages"] = strip_history_thinking(data["messages"])
            if self.server.context_policy:
                try:
                    data["messages"], _ = self.server.context_policy.rewrite(data["messages"], digest(data["messages"]))
                except Exception as error:
                    with self.server.audit.open("a") as log:
                        log.write(json.dumps({"context_error": type(error).__name__, "at": time.time()}) + "\n")
                    self.send_error(502, "Context preprocessing failed")
                    return
            body = json.dumps(data, ensure_ascii=False).encode()
        record = {
            "request_number": next(self.server.request_counter),
            "timestamp": time.time(),
            "path": self.path,
            "model": data.get("model"),
            "image_inputs": list(images(data)),
            "messages": len(data.get("messages", [])),
            "tool_observations": sum(m.get("role") == "tool" for m in data.get("messages", [])),
            "tool_definitions": len(data.get("tools", [])),
        }
        if state_proof:
            record["state_prefix"] = state_proof
        record["sampling"] = {
            key: data[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "max_tokens",
                "repetition_penalty",
                "min_p",
                "seed",
                "chat_template_kwargs",
            )
            if key in data
        }
        capture = None
        if self.server.capture_raw:
            directory = self.server.audit.parent / (self.server.audit.stem + "-raw")
            directory.mkdir(exist_ok=True)
            name = str(time.time_ns())
            (directory / (name + ".request.json")).write_bytes(body)
            record["raw_request"] = str(directory / (name + ".request.json"))
            capture = (directory / (name + ".response.bin")).open("wb")
        with self.server.audit.open("a") as log:
            log.write(json.dumps(record) + "\n")
        if self.server.request_limit and record["request_number"] > self.server.request_limit:
            self.server.limit_reached = True
            payload = json.dumps({"error": {"type": "diagnostic_capture_limit",
                "message": "Host diagnostic request limit reached; this is not a completed rollout."}}).encode()
            if capture:
                capture.write(payload)
                capture.close()
            with self.server.audit.open("a") as log:
                log.write(json.dumps({"response_to": record["timestamp"], "status": 429,
                                      "host_diagnostic_limit": True}) + "\n")
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        upstream = HTTPConnection(self.server.upstream.hostname, self.server.upstream.port, timeout=900)
        upstream.request("POST", self.path, body, {"Content-Type": "application/json"})
        response = upstream.getresponse()
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in ("connection", "transfer-encoding", "keep-alive"):
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = []
        stream_error = None
        first_byte_seconds = None
        try:
            while chunk := response.read1(65536):
                if first_byte_seconds is None:
                    first_byte_seconds = time.time() - record["timestamp"]
                chunks.append(chunk)
                if capture:
                    capture.write(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError as error:
            stream_error = type(error).__name__
        finally:
            if capture:
                capture.close()
            upstream.close()
            self.close_connection = True
            with self.server.audit.open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "response_to": record["timestamp"],
                            "status": response.status,
                            "seconds": round(time.time() - record["timestamp"], 3),
                            "first_byte_seconds": first_byte_seconds,
                            "stream_error": stream_error,
                            **response_summary(chunks),
                        }
                    )
                    + "\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--capture-raw", action="store_true")
    parser.add_argument("--request-limit", type=int, default=0)
    parser.add_argument("--strip-history-thinking", action="store_true")
    parser.add_argument("--context-cache", type=Path)
    parser.add_argument("--context-model", type=Path)
    parser.add_argument("--context-source-commit")
    parser.add_argument("--context-endpoint", default="http://127.0.0.1:32527/v1/chat/completions")
    parser.add_argument("--state-prefix-model", type=Path)
    parser.add_argument("--state-case", type=Path)
    args = parser.parse_args()
    if args.strip_history_thinking or args.context_cache or args.state_prefix_model:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    server.upstream = urlsplit(args.upstream)
    server.audit = args.audit
    server.capture_raw = args.capture_raw
    server.strip_history, server.context_policy = args.strip_history_thinking, None
    server.state_policy = None
    if args.context_cache:
        from types import SimpleNamespace
        from sft.rewrite_cot import Rewriter
        from sft.observation_context import ObservationContext

        if not args.context_model or not args.context_source_commit:
            parser.error("context-cache requires context-model and context-source-commit")
        editor = Rewriter(SimpleNamespace(model=args.context_model, output=args.context_cache,
            endpoint=args.context_endpoint, source_commit=args.context_source_commit, rewrite_observations=False))
        server.context_policy = ObservationContext(editor)
        if args.state_prefix_model:
            from transformers import AutoProcessor
            from sft.state_context import StateContext

            if not args.state_case:
                parser.error("state-prefix-model requires state-case")
            processor = AutoProcessor.from_pretrained(args.state_prefix_model, local_files_only=True)
            server.state_policy = StateContext(processor, server.context_policy)
            server.state_store, server.state_case = editor.store, args.state_case
    elif args.state_prefix_model:
        parser.error("state-prefix-model requires the causal editor context options")
    server.request_counter, server.request_limit, server.limit_reached = count(1), args.request_limit, False
    server.serve_forever()


if __name__ == "__main__":
    main()
