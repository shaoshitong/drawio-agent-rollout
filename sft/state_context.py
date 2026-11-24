"""Use one prefix policy for immutable teacher snapshots and live owned workspaces."""

from functools import lru_cache

from .history import strip_history_thinking
from .observation_context import (
    PROTECTED, measured_observation, numeric_observation, prose_observation, text_parts,
)
from .state_prefix import NativeCounter, build_state_prefix, tool_cycles


class StateContext:
    def __init__(self, processor):
        self.processor = processor

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
        counter = NativeCounter(self.processor, request.get("tools", []), read_blob)
        summaries = self.past_observations(request["messages"])
        candidate, proof = build_state_prefix(request["messages"], files, read_file, stage, counter, summaries)
        proof["source_request_ref"] = source_ref
        return candidate, proof
