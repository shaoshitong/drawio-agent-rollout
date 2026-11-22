"""Causal observation summaries shared by offline data and the inference proxy."""

from copy import deepcopy
import json
from pathlib import Path
import re
import threading

from .audit_context_budget import tool_result
from .causal_cot import digest, fragments


POLICY = "arrival-observation-v1"
PROTECTED = re.compile(
    r"<mxfile|<mxGraphModel|<mxCell|```|Traceback|\b(?:ERROR|Exception)\b|"
    r"official_drawio|png_sha256|\"exit_code\"\s*:\s*[1-9]|Process exited with code [1-9]|"
    r"^\s*(?:from \S+ import|import \S+|def |class |for .+ in |while |return |raise )|"
    r"^\s*[A-Za-z_]\w*(?:\[[^\n]+\])?\s*=", re.M,
)


def text_parts(message):
    content = message.get("content")
    if isinstance(content, str):
        return [(None, content)]
    return [(index, part["text"]) for index, part in enumerate(content or []) if part.get("type") == "text"]


def apply_observations(messages, changes):
    result = deepcopy(messages)
    for change in changes:
        message = result[change["index"]]
        part = change["part"]
        source = message["content"] if part is None else message["content"][part]["text"]
        if digest(source) != change["source_sha256"]:
            raise ValueError("Observation summary source differs from the archived prefix")
        if part is None:
            message["content"] = change["text"]
        else:
            message["content"][part]["text"] = change["text"]
    return result


class ObservationContext:
    def __init__(self, rewriter):
        self.rewriter = rewriter
        self.root = Path(rewriter.args.output) / "context-cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def summarize(self, text, function, source_ref):
        name = function.get("name", "")
        arguments = str(function.get("arguments", ""))
        if name == "skill" or re.search(r"review[12]\.json|\.drawio|\.xml|\.py\b|SKILL", arguments):
            return text
        if PROTECTED.search(text) or self.rewriter.count(text) <= 768:
            return text
        key = digest({"policy": POLICY, "editor": self.rewriter.args.source_commit,
                      "text": text, "function": function})
        path = self.root / (key + ".json")
        # At most four producers, with one immutable result for each observation.
        with self.lock:
            if path.exists():
                return json.loads(path.read_text())["text"]
        wrapped = text.startswith("<tool_response>") and text.endswith("</tool_response>")
        source = text[len("<tool_response>"):-len("</tool_response>")] if wrapped else text
        compact, audits = [], []
        for part in fragments(source, self.rewriter.tokenizer, 768):
            value, audit = self.rewriter.fragment(part, "".join(compact), self.root,
                                                  "observation:" + key, source_ref)
            compact.append(value + "\n" if audit["outcome"] == "compressed" else value)
            audits.append(audit)
        value = "".join(compact)
        if wrapped:
            value = "<tool_response>" + value + "</tool_response>"
        if self.rewriter.count(value) >= self.rewriter.count(text):
            value = text
        record = {"policy": POLICY, "source_sha256": digest(text), "text": value,
                  "source_tokens": self.rewriter.count(text), "compact_tokens": self.rewriter.count(value),
                  "audit_ref": self.rewriter.store.put(audits)}
        with self.lock:
            if not path.exists():
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            return json.loads(path.read_text())["text"]

    def rewrite(self, messages, source_ref):
        calls, changes = {}, []
        for index, message in enumerate(messages):
            calls.update({call["id"]: call["function"] for call in message.get("tool_calls") or []})
            if not tool_result(message):
                continue
            function = calls.get(message.get("tool_call_id"))
            if function is None:
                continue
            for part, source in text_parts(message):
                value = self.summarize(source, function, source_ref)
                if value != source:
                    changes.append({"index": index, "part": part, "source_sha256": digest(source), "text": value})
        return apply_observations(messages, changes), changes
