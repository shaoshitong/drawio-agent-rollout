"""Offline label eligibility; no target-dependent information changes a prefix."""

from hashlib import sha256
import json
from pathlib import Path

from .audit_rewrites import native_answer
from .state_prefix import CODE_SUFFIXES, CONTEXT_BUDGET, text


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for item in value.values() for part in strings(item)]
    if isinstance(value, list):
        return [part for item in value for part in strings(item)]
    return []


def visible_text(message):
    parts = [text(message)]
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        parts.extend(strings(json.loads(arguments) if isinstance(arguments, str) else arguments))
    return "\n".join(parts)


def windows(value):
    return {value[index:index + 64] for index in range(len(value) - 63)}


def copied_dependencies(messages, message, files, read_file):
    visible = "\n".join(visible_text(part) for part in messages)
    target = visible_text(message)
    target_windows, available = windows(target), None
    shared, missing, affected = set(), set(), []
    old_literals = []
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        old_literals.extend(arguments[key] for key in ("old_string", "oldString")
                            if isinstance(arguments.get(key), str) and arguments[key])
    missing_literals = [part for part in old_literals if part not in visible]
    names = [name for name in files if Path(name).suffix in CODE_SUFFIXES
             and (Path(name).parent == Path(".") or name in target)]
    for name in names:
        source = read_file(name).decode()
        if source in visible:
            continue
        common = windows(source) & target_windows
        if not common:
            continue
        if available is None:
            available = windows(visible)
        absent = common - available
        shared.update(common)
        missing.update(absent)
        if absent:
            affected.append(name)
    return {"copied_window_characters": 64, "shared_windows_checked": len(shared),
            "missing_copied_windows": len(missing), "affected_files": affected,
            "missing_edit_literals": len(missing_literals),
            "missing_window_hashes": sorted(sha256(part.encode()).hexdigest() for part in missing)[:8],
            "scope": "Necessary copied-content visibility check; not proof of sufficient causal context"}


def qualify_target(processor, request, response, files, read_file, input_tokens,
                   mutation=True, include_reasoning=False):
    options = {"tokenize": False, "enable_thinking": True, "tools": request.get("tools", [])}
    completion = native_answer(processor, request, response, options, include_reasoning)
    count = len(processor.tokenizer(completion, add_special_tokens=False)["input_ids"])
    proof = {"target_tokens": count, "target_includes_reasoning": include_reasoning,
             "target_used_only_for_label_eligibility": True}
    if input_tokens + count > CONTEXT_BUDGET:
        return {**proof, "rejection": "complete_target_exceeds_context"}
    if mutation:
        visibility = copied_dependencies(request["messages"], response["choices"][0]["message"], files, read_file)
        proof["visibility"] = visibility
        if visibility["missing_copied_windows"] or visibility["missing_edit_literals"]:
            proof["rejection"] = "copied_pre_action_content_not_visible"
    return proof
