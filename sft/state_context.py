"""Use one prefix policy for immutable teacher snapshots and live owned workspaces."""

from functools import lru_cache
from copy import deepcopy
import json

from .history import strip_history_thinking
from .observation_context import (
    PROTECTED, measured_observation, numeric_observation, prose_observation, text_parts,
)
from .state_prefix import INPUT_BUDGET, NativeCounter, build_state_prefix, tool_cycles, tool_result


def build_phase(request, files, stage, limit, read_file=None):
    cycles = tool_cycles(request["messages"])
    completed = sum(len(message.get("tool_calls") or []) for cycle in cycles
        if any(tool_result(message) for message in cycle) for message in cycle)
    if not limit or stage != 1 or f"round{stage}.drawio" in files or completed < limit:
        return request, None
    last_cycle = next(cycle for cycle in reversed(cycles) if any(tool_result(m) for m in cycle))
    calls = [call["function"] for message in last_cycle for call in message.get("tool_calls") or []]
    last = calls[-1]
    arguments = last["arguments"]
    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    script = "/case/build_drawio.py"
    execute = "build_drawio.py" in files and last["name"] == "write" and arguments.get("file_path") == script
    name = "bash" if execute else "write"
    draft = deepcopy(next(tool for tool in request["tools"] if tool["function"]["name"] == name))
    marker = None if execute else "DRAWIO_PROGRAM_CONTENT"
    call = {"name": name, "arguments": {"description": "Compile the initial editable diagram",
            "command": "timeout 120s python3 " + script} if execute else {"file_path": script, "content": marker}}
    options = {"tools": [draft], "tool_choice": "auto",
        "parallel_tool_calls": False, "separate_reasoning": False,
        "chat_template_kwargs": {**request.get("chat_template_kwargs", {}),
                                 "drawio_raw_tools": [draft], "enable_thinking": False}}
    instruction = ("Execute your construction program to create /case/round1.drawio." if execute else
        "Initial inspection is complete. Write /case/build_drawio.py now: a compact Python construction "
        "program that creates /case/round1.drawio with native editable objects from the reference. "
        "Use arrays and finite loops for repeated components; do not unroll repeated XML by hand. "
        "Use current estimates for an initial draft and refine after rendering. Do not crop or scan pixels.")
    if read_file:
        from io import BytesIO
        from PIL import Image
        from .state_prefix import text

        with Image.open(BytesIO(read_file("reference.png"))) as image:
            instruction += f" Reference image dimensions: {image.width} by {image.height} pixels. Geometry uses this original pixel frame."
        if "build_drawio.py" in files and not execute:
            instruction += "\nYour previous program:\n" + read_file("build_drawio.py").decode()
            instruction += "\nObserved execution result:\n" + "\n".join(text(m) for m in last_cycle if tool_result(m))
    message = {"role": "user", "content": instruction}
    return {**request, **options, "messages": [*request["messages"], message]}, {
        "completed_tools": completed, "inspection_limit": limit, "model_options": options,
        "forced_call": call, "content_marker": marker, "fully_forced": execute}


class StateContext:
    def __init__(self, processor, build_after_tools=0):
        self.processor = processor
        self.build_after_tools = build_after_tools

    @lru_cache(maxsize=8192)
    def observation_tokens(self, text):
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def past_observations(self, messages):
        candidates, calls = [], {}
        cycles = tool_cycles(strip_history_thinking(messages))
        past = [(index, message) for index, cycle in enumerate(cycles) for message in cycle]
        for index, message in past:
            calls.update({call["id"]: call["function"] for call in message.get("tool_calls") or []})
            function = calls.get(message.get("tool_call_id"), {})
            for _, source in text_parts(message):
                numeric = message["role"] == "tool" and numeric_observation(source)
                if numeric or prose_observation(source) and not PROTECTED.search(source):
                    value = measured_observation(source, function) if numeric else source
                    candidates.append({"numeric": numeric, "cycle": index, "text": value,
                                       "tokens": self.observation_tokens(value)})
        # Numeric facts have priority; no model or target participates in selection.
        return sorted(candidates, key=lambda item: (item["numeric"], item["cycle"]))

    def build(self, request, files, read_file, stage, source_ref, read_blob=None):
        request, phase = build_phase(request, files, stage, self.build_after_tools, read_file)
        counter = NativeCounter(self.processor, request.get("tools", []), read_blob,
                                request.get("chat_template_kwargs") if phase else None,
                                input_budget=12288 if phase else INPUT_BUDGET, allow_state_only=bool(phase))
        summaries = self.past_observations(request["messages"])
        candidate, proof = build_state_prefix(request["messages"], files, read_file, stage, counter, summaries)
        proof["source_request_ref"] = source_ref
        if phase:
            phase["input_budget"] = counter.input_budget
            proof["build_phase"] = phase
        return candidate, proof
