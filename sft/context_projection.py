"""History-only projection candidate; not enabled in training or serving."""

from copy import deepcopy
from hashlib import sha256
import json
import re

from .audit_context_budget import result_anchors, tool_result


QUOTES = re.compile(r'`([^`\n]{24,512})`|"([^"\n]{24,512})"|“([^”\n]{24,512})”')
PROTOCOL = "protected-history-v1"
THINKING_PROTOCOL = "history-v2"


def message_text(message):
    content = message.get("content") or ""
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


def strip_history_thinking(messages):
    """Remove explicit historical reasoning; preserve ambiguous/literal content."""
    candidate = deepcopy(messages)
    for message in candidate:
        if message["role"] != "assistant":
            continue
        if isinstance(message.get("reasoning_content"), str):
            # Keep an explicit empty field so the template does not reinterpret
            # literal </think> text in visible content as a reasoning boundary.
            message["reasoning_content"] = ""
            continue
        content = message.get("content")
        if isinstance(content, str) and content.lstrip().startswith("<think>") and "</think>" in content:
            message["content"] = content.split("</think>", 1)[1].lstrip("\n")
            message["reasoning_content"] = ""
    return candidate


def dependencies(messages, groups):
    """Protect repair chains and sources quoted by past assistant messages."""
    results = {index for index, message in enumerate(messages) if tool_result(message)}
    repair = set()
    errors = set(groups["error"])
    calls = {call["id"]: call["function"] for message in messages
             for call in message.get("tool_calls") or []}
    repairing = False
    quoted = set()
    prior_results = []
    for index, message in enumerate(messages):
        text = message_text(message)
        if index in results:
            repairing = repairing or index in errors
            if repairing:
                repair.add(index)
                arguments = json.dumps(calls.get(message.get("tool_call_id"), {}).get("arguments", {}))
                rendered = "png_sha256" in text or "official_drawio" in text or (
                    "render" in arguments and ".drawio" in arguments)
                repairing = not (rendered and index not in errors)
            prior_results.append((index, " ".join(text.split())))
        if message["role"] != "assistant":
            continue
        history = (message.get("reasoning_content") or "") + "\n" + text
        fragments = {" ".join(next(value for value in match.groups() if value).split())
                     for match in QUOTES.finditer(history)}
        for fragment in fragments:
            source = next((source for source, output in reversed(prior_results) if fragment in output), None)
            if source is not None:
                quoted.add(source)
    return {"repair_chain": sorted(repair), "history_quote": sorted(quoted)}


def result_stub(message, function):
    text = message_text(message)
    raw = text.encode()
    arguments = json.dumps(function.get("arguments", {}), sort_keys=True, ensure_ascii=False)
    metadata = {"tool": function.get("name"), "call_id": message.get("tool_call_id"),
                "arguments_sha256": sha256(arguments.encode()).hexdigest(),
                "output_sha256": sha256(raw).hexdigest(), "output_bytes": len(raw)}
    stub = "[Earlier tool text omitted; re-read own files if needed. " + json.dumps(metadata) + "]"
    if text.startswith("<tool_response>") and text.endswith("</tool_response>"):
        stub = "<tool_response>\n" + stub + "\n</tool_response>"
    return stub


def project_history(messages, keep_results):
    """Keep all calls, assistant text and image parts; only replace old result text."""
    groups = result_anchors(messages)
    groups.update(dependencies(messages, groups))
    results = [index for index, message in enumerate(messages) if tool_result(message)]
    groups["recent"] = results[-keep_results:] if keep_results else []
    protected = set().union(*groups.values())
    calls = {call["id"]: call["function"] for message in messages
             for call in message.get("tool_calls") or []}
    candidate = deepcopy(messages)
    cleared = sorted(set(results) - protected)
    for index in cleared:
        message = candidate[index]
        text = message_text(message)
        if not text or "[Earlier tool text omitted; re-read own files if needed. " in text:
            continue
        stub = result_stub(message, calls.get(message.get("tool_call_id"), {}))
        if isinstance(message.get("content"), str):
            message["content"] = stub
            continue
        first = True
        for part in message.get("content") or []:
            if part.get("type") == "text":
                part["text"] = stub if first else ""
                first = False
    return candidate, {"keep_results": keep_results, "protected": groups, "cleared": cleared}


def select_history(messages, prompt_tokens, *, context=262144, reserve=49152, margin=256,
                   max_new_tokens=131072, strip_thinking=False):
    """Budget using only input tokens; the callback includes native image expansion."""
    original = prompt_tokens(messages)
    options = {"protocol": THINKING_PROTOCOL if strip_thinking else PROTOCOL,
               "original_prompt_tokens": original, "reserve": reserve,
               "margin": margin, "context": context, "requested_max_new_tokens": max_new_tokens}
    if original + reserve + margin <= context:
        return messages, {**options, "stage": 0, "history_thinking_removed": False,
                          "triggered": False, "fits": True, "prompt_tokens": original,
                          "max_new_tokens": min(max_new_tokens, context - original - margin)}
    attempts = []
    if strip_thinking:
        candidate = strip_history_thinking(messages)
        count = prompt_tokens(candidate)
        attempts.append({"stage": 1, "keep_results": "all", "prompt_tokens": count})
        if count + reserve + margin <= context:
            return candidate, {**options, "stage": 1, "history_thinking_removed": True,
                               "triggered": True, "fits": True, "keep_results": "all",
                               "prompt_tokens": count, "attempts": attempts,
                               "max_new_tokens": min(max_new_tokens, context - count - margin)}
    for keep in (16, 8, 4, 0):
        # All anchors are computed from the original input, including old thoughts.
        candidate, details = project_history(messages, keep)
        if strip_thinking:
            candidate = strip_history_thinking(candidate)
        count = prompt_tokens(candidate)
        attempts.append({"stage": 2, "keep_results": keep, "prompt_tokens": count})
        if count + reserve + margin <= context:
            return candidate, {**options, **details, "stage": 2, "history_thinking_removed": strip_thinking,
                               "triggered": True, "fits": True,
                               "prompt_tokens": count, "attempts": attempts,
                               "max_new_tokens": min(max_new_tokens, context - count - margin)}
    return None, {**options, "triggered": True, "fits": False, "attempts": attempts,
                  "failure": "protected_history_exceeds_budget"}
