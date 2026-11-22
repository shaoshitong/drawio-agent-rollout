"""Use one prefix policy for immutable teacher snapshots and live owned workspaces."""

from .history import strip_history_thinking
from .observation_context import PROTECTED, prose_observation, text_parts
from .state_prefix import NativeCounter, build_state_prefix, tool_cycles


class StateContext:
    def __init__(self, processor, observations):
        self.processor, self.observations = processor, observations

    def prose(self, messages, source_ref):
        candidates, calls = [], {}
        cycles = tool_cycles(strip_history_thinking(messages))
        for cycle in cycles[:-4]:
            for message in cycle:
                calls.update({call["id"]: call["function"] for call in message.get("tool_calls") or []})
                function = calls.get(message.get("tool_call_id"), {"name": "earlier-visible-prose"})
                for _, source in text_parts(message):
                    if message["role"] in ("assistant", "tool", "user") and prose_observation(source) and not PROTECTED.search(source):
                        candidates.append((source, function))
        # Select complete past observations before editing; never inspect a target.
        selected, count = [], 0
        for source, function in reversed(candidates):
            size = self.observations.rewriter.count(source)
            if count + size > 8192:
                break
            selected.append((source, function))
            count += size
        return [self.observations.summarize(source, function, source_ref)
                for source, function in reversed(selected)]

    def build(self, request, files, read_file, stage, source_ref, read_blob=None):
        counter = NativeCounter(self.processor, request.get("tools", []), read_blob)
        candidate, proof = build_state_prefix(request["messages"], files, read_file, stage, counter)
        if candidate is None:
            return None, proof
        summaries = self.prose(request["messages"], source_ref)
        if summaries:
            candidate, proof = build_state_prefix(request["messages"], files, read_file, stage, counter, summaries)
        proof["prose_observations"] = len(summaries)
        return candidate, proof
