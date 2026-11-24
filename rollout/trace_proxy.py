"""Transparent local inference audit proxy; no reward or gold endpoint exists."""

import argparse
import base64
from hashlib import sha256
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
import time
from urllib.parse import urlsplit


HEARTBEAT = b": keepalive\n\n"


def read_response(response, events):
    try:
        while chunk := response.read1(65536):
            events.put(chunk)
    except (OSError, HTTPException) as error:
        events.put(error)
    finally:
        events.put(None)


def response_chunks(response, interval=15):
    events = Queue()
    Thread(target=read_response, args=(response, events), daemon=True).start()
    pending = b""
    while True:
        try:
            chunk = events.get(timeout=interval)
        except Empty:
            yield HEARTBEAT
            continue
        if chunk is None:
            if pending:
                yield pending
            return
        if isinstance(chunk, Exception):
            raise chunk
        pending += chunk
        # Insert comments between complete SSE events, never inside JSON data.
        boundaries = [index + len(separator) for separator in (b"\n\n", b"\r\n\r\n", b"\r\r")
                      if (index := pending.rfind(separator)) >= 0]
        if boundaries:
            boundary = max(boundaries)
            yield pending[:boundary]
            pending = pending[boundary:]


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
    reasoning, content = [], []
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
                content.append(delta.get("content") or "")
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
        summary["content_text"] = "".join(content)
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
            tools = data.get("tools") or []
            for tool in tools:
                tool["function"].setdefault("strict", False)
            data["tools"] = tools
            data["chat_template_kwargs"] = {
                **(data.get("chat_template_kwargs") or {}), "drawio_raw_tools": tools,
            }
            from sft.state_prefix import CONTEXT_BUDGET, NativeCounter
            from sft.store import snapshot

            past = getattr(self.server, "past_observations", None)
            if past:
                messages = data["messages"]
                first_reply = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
                data["messages"] = messages[:first_reply] + past + messages[first_reply:]
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
            if state_proof.get("build_phase"):
                data.update(state_proof["build_phase"]["model_options"])
            input_tokens = state_proof["input_tokens"]
            backend = getattr(self.server, "backend_processor", None)
            if backend:
                counter = NativeCounter(backend, data["tools"], chat_template_kwargs=data["chat_template_kwargs"])
                state_proof["backend_input_tokens"] = counter(messages)
                input_tokens = max(input_tokens, state_proof["backend_input_tokens"])
            data["max_tokens"] = min(data.get("max_tokens", CONTEXT_BUDGET), CONTEXT_BUDGET - input_tokens - 1)
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
        if getattr(self.server, "thinking_control", None):
            data.update(self.server.thinking_control)
            if state_proof and state_proof.get("build_phase"):
                from thinking_control import tool_prefix

                phase = state_proof["build_phase"]
                prefix = tool_prefix(self.server.state_policy.processor.tokenizer, data["tools"],
                                     phase["forced_call"], phase["content_marker"])
                data["custom_params"] = {**data["custom_params"], "forced_prefix_ids": prefix}
                phase["forced_prefix_tokens"] = len(prefix)
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
        if getattr(self.server, "thinking_control", None):
            record["thinking_loop_recovery"] = data["custom_params"]
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
        streaming = response.getheader("Content-Type", "").startswith("text/event-stream")
        self.send_response(response.status)
        blocked = {"connection", "transfer-encoding", "keep-alive"}
        if streaming:
            blocked.add("content-length")
        for key, value in response.getheaders():
            if key.lower() not in blocked:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = []
        stream_error = None
        first_byte_seconds = None
        heartbeats = 0
        try:
            incoming = (response_chunks(response, getattr(self.server, "heartbeat_seconds", 15))
                        if streaming else iter(lambda: response.read1(65536), b""))
            for chunk in incoming:
                heartbeat = streaming and chunk == HEARTBEAT
                heartbeats += int(heartbeat)
                if first_byte_seconds is None and not heartbeat:
                    first_byte_seconds = time.time() - record["timestamp"]
                chunks.append(chunk)
                if capture:
                    capture.write(chunk)
                    if heartbeat:
                        capture.flush()
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, HTTPException) as error:
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
                            "sse_heartbeats": heartbeats,
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
    parser.add_argument("--backend-prefix-model", type=Path)
    parser.add_argument("--state-case", type=Path)
    parser.add_argument("--past-observations", type=Path)
    parser.add_argument("--recover-thinking-loop", action="store_true")
    parser.add_argument("--build-after-tools", type=int, default=0)
    args = parser.parse_args()
    if args.strip_history_thinking or args.context_cache or args.state_prefix_model:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    server.upstream = urlsplit(args.upstream)
    server.audit = args.audit
    server.capture_raw = args.capture_raw
    server.strip_history, server.context_policy = args.strip_history_thinking, None
    server.state_policy = None
    server.thinking_control = None
    if args.state_prefix_model:
        from transformers import AutoProcessor
        from sft.state_context import StateContext
        from sft.store import Store

        if not args.state_case or not args.context_cache:
            parser.error("state-prefix-model requires state-case and context-cache")
        processor = AutoProcessor.from_pretrained(args.state_prefix_model, local_files_only=True)
        if args.backend_prefix_model:
            server.backend_processor = AutoProcessor.from_pretrained(args.backend_prefix_model, local_files_only=True)
        server.state_policy = StateContext(processor, args.build_after_tools)
        server.state_store = Store(args.context_cache / "archive")
        server.state_case = args.state_case
        if args.past_observations:
            server.past_observations = json.loads(args.past_observations.read_text())
        if args.recover_thinking_loop or args.build_after_tools:
            from thinking_control import request_control

            server.thinking_control = request_control(processor.tokenizer, args.state_prefix_model)
            server.thinking_control["custom_params"]["recover_thinking_loop"] = args.recover_thinking_loop
    elif args.context_cache:
        from types import SimpleNamespace
        from sft.rewrite_cot import Rewriter
        from sft.observation_context import ObservationContext

        if not args.context_model or not args.context_source_commit:
            parser.error("context-cache requires context-model and context-source-commit")
        editor = Rewriter(SimpleNamespace(model=args.context_model, output=args.context_cache,
            endpoint=args.context_endpoint, source_commit=args.context_source_commit, rewrite_observations=False))
        server.context_policy = ObservationContext(editor)
    server.request_counter, server.request_limit, server.limit_reached = count(1), args.request_limit, False
    server.serve_forever()


if __name__ == "__main__":
    main()
