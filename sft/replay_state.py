"""Replay one archived model request and let DSH execute its first tool batch."""

import argparse
from hashlib import sha256
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from threading import Thread
from types import SimpleNamespace

from .audit_transitions import event_calls
from .evaluate_checkpoint import evaluation_config, validate_xml
from .gateway import parse_reply
from rollout import run_turn


class ReplayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        replay = self.server
        if replay.response.exists():
            replay.followups += 1
            self.send_error(410, "Replay stops after the first model reply and tool batch")
            return
        connection = HTTPConnection("127.0.0.1", replay.upstream, timeout=1800)
        try:
            connection.request("POST", "/v1/chat/completions", replay.request,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            replay.status = response.status
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.end_headers()
            with replay.response.open("xb") as archive:
                while chunk := response.read1(65536):
                    archive.write(chunk)
                    archive.flush()
                    self.wfile.write(chunk)
        finally:
            connection.close()


def workspace_hashes(case):
    return {str(item.relative_to(case)): sha256(item.read_bytes()).hexdigest()
            for item in case.rglob("*") if not item.is_symlink() and item.is_file()}


def replay_state(args):
    receipt = next(row for row in json.loads((args.states / "manifest.json").read_text())
                   if row["id"] == args.state)
    archived = args.states / args.state
    request_bytes = (archived / "request.json").read_bytes()
    request = json.loads(request_bytes)
    if (sha256(request_bytes).hexdigest() != receipt["request_sha256"]
            or workspace_hashes(archived / "case") != receipt["workspace_files"]):
        raise ValueError("Request or workspace differs from the captured state")
    if request.get("tool_choice") not in (None, "auto") or not request.get("stream"):
        raise ValueError("Expected an unrestricted streaming request")
    args.output.mkdir(parents=True)
    case = args.output / "case"
    shutil.copytree(archived / "case", case)
    with ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler) as proxy:
        proxy.request, proxy.upstream = request_bytes, args.upstream
        proxy.response = args.output / "response.bin"
        proxy.followups, proxy.status = 0, None
        worker = Thread(target=proxy.serve_forever, daemon=True)
        worker.start()
        config = evaluation_config(SimpleNamespace(out=args.output, root=args.root, port=args.upstream,
                                                   causal_context=True, generator_model=request["model"]))
        config.generator_url = f"http://127.0.0.1:{proxy.server_port}/v1"
        config.resume = False
        error = None
        try:
            run_turn(config, "generator", 1, None)
        except RuntimeError as exc:
            error = str(exc)
        finally:
            proxy.shutdown()
            worker.join()
    events_path = args.output / "generator/round1/events.jsonl"
    events = event_calls(events_path) if events_path.is_file() else []
    raw = proxy.response.read_bytes() if proxy.response.is_file() else b""
    response, usage = parse_reply(raw) if raw and proxy.status == 200 else ({}, {})
    choices = response.get("choices", [])
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message", {})
    parsed = message.get("tool_calls") or []
    after = workspace_hashes(case)
    changed = {path: digest for path, digest in after.items() if receipt["workspace_files"].get(path) != digest}
    candidates = []
    for path in changed:
        if Path(path).suffix not in (".drawio", ".xml"):
            continue
        try:
            details = validate_xml(case / path)
        except (SyntaxError, ValueError):
            continue
        candidates.append({"path": path, **details})
    complete = proxy.status == 200 and b"data: [DONE]" in raw and choice.get("finish_reason") is not None
    actions = [{"id": item["callId"], "name": item["tool"], "result": item["result"]} for item in events]
    parsed_actions = [(item["id"], item["function"]["name"]) for item in parsed]
    observed_actions = [(item["id"], item["name"]) for item in actions]
    report = {
        "state": args.state, "variant": args.variant, "request_sha256": receipt["request_sha256"],
        "workspace_sha256": sha256(json.dumps(receipt["workspace_files"], sort_keys=True).encode()).hexdigest(),
        "output_budget": request["max_tokens"], "usage": usage, "finish_reason": choice.get("finish_reason"),
        "complete": complete, "model_tool_calls": [item["function"]["name"] for item in parsed],
        "reasoning_characters": len(message.get("reasoning_content") or ""),
        "visible_reply_characters": len(message.get("content") or ""),
        "tool_events": actions, "valid_editable_xml": candidates, "blocked_followups": proxy.followups,
        "changed_files": sorted(changed), "dsh_error": None if proxy.followups else error,
    }
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not complete or sorted(observed_actions) != sorted(parsed_actions) or any(item["result"] is None for item in actions):
        raise RuntimeError("Incomplete replay or tool execution; see result.json and captured logs")
    print(json.dumps({k: v for k, v in report.items() if k != "tool_events"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--upstream", required=True, type=int)
    replay_state(parser.parse_args())


if __name__ == "__main__":
    main()
