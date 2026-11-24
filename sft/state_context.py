"""Use one prefix policy for immutable teacher snapshots and live owned workspaces."""

from functools import lru_cache
from copy import deepcopy

from .history import strip_history_thinking
from .observation_context import (
    PROTECTED, measured_observation, numeric_observation, prose_observation, text_parts,
)
from .state_prefix import INPUT_BUDGET, NativeCounter, build_state_prefix, tool_cycles, tool_result


def build_phase(request, files, stage, limit):
    cycles = tool_cycles(request["messages"])
    completed = sum(len(message.get("tool_calls") or []) for cycle in cycles
        if any(tool_result(message) for message in cycle) for message in cycle)
    if not limit or stage != 1 or f"round{stage}.drawio" in files or completed < limit:
        return request, None
    draft = deepcopy(next(tool for tool in request["tools"] if tool["function"]["name"] == "write"))
    draft["function"]["parameters"] = {"type": "object", "properties": {
        "file_path": {"type": "string", "enum": ["/case/round1.drawio"]},
        "content": {"type": "string", "description": "Complete native editable draw.io XML initial draft."}},
        "required": ["file_path", "content"], "additionalProperties": False}
    options = {"tools": [draft], "tool_choice": {"type": "function", "function": {"name": "write"}},
        "parallel_tool_calls": False, "separate_reasoning": False,
        "chat_template_kwargs": {**request.get("chat_template_kwargs", {}),
                                 "drawio_raw_tools": [draft], "enable_thinking": False}}
    message = {"role": "user", "content": "Initial inspection is complete. Write /case/round1.drawio "
        "now using native editable objects from the reference and your observed facts. Use current "
        "estimates for an initial draft; render it and refine it afterwards. No more inspection before writing."}
    return {**request, **options, "messages": [*request["messages"], message]}, {
        "completed_tools": completed, "inspection_limit": limit, "model_options": options}


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
        request, phase = build_phase(request, files, stage, self.build_after_tools)
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
