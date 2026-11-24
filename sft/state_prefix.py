"""Bound a decision prefix using only its history and its own pre-request files."""

import base64
from bisect import bisect_left
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3

from .audit_context_budget import template_messages, tool_result
from .history import strip_history_thinking


POLICY = "own-state-prefix-v3"
INPUT_BUDGET = 24576
CONTEXT_BUDGET = 32768
CODE_SUFFIXES = {".py", ".js", ".sh", ".xml", ".drawio", ".svg"}
ERROR = re.compile(r"Traceback|Process exited with code [1-9]|\"is_error\"\s*:\s*true|^\s*(?:Error|ERROR):", re.M)


def snapshot_before_request(row, store):
    ledger = store.root / "ledger.sqlite"
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
        episode, stage, request, reference, started = db.execute(
            "select episode,stage,request,artifacts,started from calls where id=?", (row["call_id"],)
        ).fetchone()
    if (episode, stage, request) != (row["episode"], row["stage"], row["request_ref"]):
        raise ValueError("Pre-request file snapshot does not belong to this decision")
    return store.get(reference), {"reference": reference, "call_started": started, "stage": stage,
                                  "timing": "gateway snapshot before reserve/provider request"}


def image_part(raw):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}


def text(message):
    content = message.get("content") or ""
    return content if isinstance(content, str) else "\n".join(part.get("text", "") for part in content)


def tool_cycles(messages):
    cycles = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            cycles.append([])
        if cycles and message["role"] != "system":
            cycles[-1].append(message)
    return cycles


def latest_error(cycles):
    pending = None
    for cycle in cycles:
        output = "\n".join(text(message) for message in cycle if tool_result(message))
        if ERROR.search(output):
            pending = cycle
        elif "official_drawio" in output or "png_sha256" in output:
            pending = None
    return pending


def keep_lines(lines, tokenizer, budget):
    lines = list(lines)
    source = "\n".join(lines)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalized = backend.normalizer.normalize_str(source) if backend and backend.normalizer else source
    if backend and normalized == source:
        encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        starts = [start for start, _ in offsets]
        boundaries = {span[0] for _, span in backend.pre_tokenizer.pre_tokenize_str(source)}
        position = 0
        for line in lines:
            index = bisect_left(starts, position)
            aligned = position in boundaries and (index == 0 or offsets[index - 1][1] <= position)
            count = len(starts) - index if aligned else len(
                tokenizer(source[position:], add_special_tokens=False)["input_ids"])
            if count <= budget:
                return source[position:]
            position += len(line) + 1
        return ""
    while lines and len(tokenizer("\n".join(lines), add_special_tokens=False)["input_ids"]) > budget:
        lines.pop(0)
    return "\n".join(lines)


def action_ledger(cycles):
    lines = []
    for index, cycle in enumerate(cycles, 1):
        failed = any(ERROR.search(text(message)) for message in cycle if tool_result(message))
        for message in cycle:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = str(function.get("arguments", ""))
                paths = sorted(set(re.findall(r"(?:/case/)?[\w.-]+\.(?:drawio|xml|py|png|svg|json)", arguments)))
                lines.append(f"{index}: {function['name']} files={','.join(paths)} error_observed={failed}")
    return lines


def keep_observations(observations, tokenizer, budget=2048):
    selected, used = [], 0
    for observation in reversed(observations):
        size = observation["tokens"] + 3
        if used + size <= budget:
            selected.insert(0, observation["text"])
            used += size
    while selected and len(tokenizer.encode("\n".join(selected), add_special_tokens=False)) > budget:
        selected.pop(0)
    return "\n".join(selected), len(selected)


def add_observations(messages, observations, counter, count):
    content = messages[-1]["content"][0]
    original = content["text"]
    budget = min(2048, INPUT_BUDGET - count - 32)
    while budget > 0:
        summary, retained = keep_observations(observations, counter.tokenizer, budget)
        if not summary:
            break
        content["text"] = original + "\n\nEarlier observations (complete retained results):\n" + summary
        expanded = counter(messages)
        if expanded <= INPUT_BUDGET:
            return expanded, retained
        budget -= expanded - INPUT_BUDGET + 4
    content["text"] = original
    return count, 0


def current_files(files, recent, stage):
    history = json.dumps(recent, ensure_ascii=False)
    selected = {name for name in files if Path(name).suffix in CODE_SUFFIXES and name in history}
    xml = next((f"round{number}.drawio" for number in range(stage, 0, -1)
                if f"round{number}.drawio" in files), None)
    if xml:
        selected.add(xml)
    for name in (".agents/skills/drawio-reconstruction/SKILL.md", "review1.json"):
        if name in files:
            selected.add(name)
    png = next((f"round{number}.png" for number in range(stage, 0, -1) if f"round{number}.png" in files), None)
    return sorted(selected), png


def resolve_images(messages, read_blob):
    messages = deepcopy(messages)
    for message in messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if part.get("type") != "image_url":
                continue
            value = part["image_url"]["url"]
            if isinstance(value, tuple):
                part["image_url"]["url"] = value[1] + "," + base64.b64encode(read_blob(value[2])).decode()
    return messages


class NativeCounter:
    def __init__(self, processor, tools, read_blob=None):
        self.processor, self.tools = processor, tools
        self.tokenizer = processor.tokenizer
        self.read_blob = read_blob

    @lru_cache(maxsize=512)
    def image_tokens(self, url):
        from PIL import Image

        raw = self.read_blob(url[2]) if isinstance(url, tuple) else base64.b64decode(url.split(",", 1)[1])
        with Image.open(BytesIO(raw)) as image:
            patches = self.processor.image_processor.get_number_of_image_patches(image.height, image.width)
        return patches // self.processor.image_processor.merge_size**2

    def __call__(self, messages):
        rendered = self.processor.apply_chat_template(template_messages(messages), tokenize=False,
            tools=self.tools, enable_thinking=True, add_generation_prompt=True)
        count = len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        for message in messages:
            content = message.get("content")
            for part in content if isinstance(content, list) else []:
                if part.get("type") == "image_url":
                    count += self.image_tokens(part["image_url"]["url"]) - 1
        return count


def build_state_prefix(messages, files, read_file, stage, counter, summaries=()):
    """No target argument: input construction cannot choose files using the answer."""
    messages = strip_history_thinking(messages)
    cycles = tool_cycles(messages)
    recent = cycles[-4:]
    selected, png = current_files(files, recent, stage)
    systems = [message for message in messages if message["role"] == "system"]
    users = [message for message in messages if message["role"] == "user" and not tool_result(message)]
    instructions = [message for message in users if isinstance(message.get("content"), str)]
    task = users[:1]
    latest = instructions[-1:] if instructions and instructions[-1] not in task else []
    state = [f"Own workspace before the current decision, generator round {stage}."]
    state.append("The latest available render can predate edits; it is not a reference answer.")
    hashes = {}
    optional = None
    latest_xml = next((f"round{number}.drawio" for number in range(stage, 0, -1)
                       if f"round{number}.drawio" in files), None)
    for name in selected:
        raw = read_file(name)
        hashes[name] = sha256(raw).hexdigest()
        state.append(f"Own file: path={json.dumps(name)} bytes={len(raw)} sha256={hashes[name]}")
        if name in (".agents/skills/drawio-reconstruction/SKILL.md", "review1.json"):
            state.append(f"<own_file path={json.dumps(name)}>\n{raw.decode()}\n</own_file>")
        elif name == latest_xml:
            optional = f"<own_file path={json.dumps(name)}>\n{raw.decode()}\n</own_file>"
    state.append("Files remain available in your workspace. Read them with tools when needed; omitted file content is not known.")
    content = [{"type": "text", "text": ""}]
    for name in ("reference.png", png):
        if name:
            raw = read_file(name)
            hashes[name] = sha256(raw).hexdigest()
            content.extend(({"type": "text", "text": "Image: " + name}, image_part(raw)))
    packet = {"role": "user", "content": content}
    pending = latest_error(cycles)
    for keep in range(min(4, len(cycles)), 0, -1) if cycles else (0,):
        earlier = [item for item in summaries if item["cycle"] < len(cycles) - keep]
        retained = cycles[-keep:] if keep else []
        protected = [pending] if pending and pending not in retained else []
        history = [message for cycle in protected + retained for message in cycle if message not in task + latest]
        ledger = keep_lines(action_ledger(cycles[:-keep] if keep else cycles), counter.tokenizer, 1024)
        content[0]["text"] = "\n\n".join(state) + "\n\nEarlier actions:\n" + ledger
        candidate = systems + task + history + latest + [packet]
        count = counter(candidate)
        if count <= INPUT_BUDGET:
            injected = False
            if optional:
                content[0]["text"] += "\n\n" + optional
                extended = counter(candidate)
                if extended <= INPUT_BUDGET:
                    count, injected = extended, True
                else:
                    content[0]["text"] = content[0]["text"][:-len(optional) - 2]
            count, observation_count = add_observations(candidate, earlier, counter, count)
            return resolve_images(candidate, counter.read_blob), {"policy": POLICY, "input_tokens": count, "recent_cycles": keep,
                "original_cycles": len(cycles), "files_sha256": hashes,
                "latest_xml_fulltext_injected": injected,
                "past_observation_candidates": len(earlier),
                "past_observations_retained": observation_count,
                "history_thinking_messages": 0, "targets_used_to_build_input": False}
    from .observation_context import pack_pixel_observations

    packed, changes = pack_pixel_observations(messages)
    if changes:
        candidate, proof = build_state_prefix(packed, files, read_file, stage, counter, summaries)
        proof["lossless_pixel_observations"] = len(changes)
        return candidate, proof
    return None, {"policy": POLICY, "rejection": "protected_state_exceeds_input_budget",
                   "input_tokens": count, "files_sha256": hashes, "targets_used_to_build_input": False}
