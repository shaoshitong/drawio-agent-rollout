"""Input-prefix policy: only the separately supplied current target keeps thinking."""

from copy import deepcopy


HISTORY_POLICY = "no-history-thinking-v1"


def strip_history_thinking(messages):
    """Remove assistant reasoning from a prefix without changing calls or images."""
    result = deepcopy(messages)
    for message in result:
        if message["role"] != "assistant":
            continue
        content = message.get("content")
        explicit = isinstance(message.get("reasoning_content"), str)
        if not explicit and isinstance(content, str) and content.lstrip().startswith("<think>") and "</think>" in content:
            message["content"] = content.split("</think>", 1)[1].lstrip("\n")
        # An explicit empty field also prevents literal code containing </think>
        # from being reinterpreted as reasoning by the native template.
        message["reasoning_content"] = ""
    return result
