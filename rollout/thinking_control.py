"""End a repeated reasoning block without stopping its answer or tool call."""

import json


class ThinkingLoopProcessor:
    def __init__(self):
        self.tokenizer = None

    def next_token(self, req, params):
        from sft.detect_loop import reasoning_loop

        output = req.output_ids
        state = getattr(req, "_drawio_thinking_control", None)
        if state is None:
            start, end = params["think_start_id"], params["think_end_id"]
            # Previous closed reasoning blocks must not disable the current one.
            opened = max((i for i, token in enumerate(req.origin_input_ids) if token == start), default=-1)
            closed = max((i for i, token in enumerate(req.origin_input_ids) if token == end), default=-1)
            state = {"start": 0 if opened > closed else None, "scanned": 0,
                     "finished": False, "forcing": None}
            req._drawio_thinking_control = state
        if state["finished"]:
            return None
        recent = output[state["scanned"]:]
        if params["think_end_id"] in recent:
            state["finished"] = True
            return None
        if state["forcing"] is not None:
            return params["newline_id"] if len(output) == state["forcing"] else params["think_end_id"]
        if state["start"] is None:
            if params["think_start_id"] not in output:
                return None
            state["start"] = output.index(params["think_start_id"]) + 1
        if len(output) - state["scanned"] < 32:
            return None
        state["scanned"] = len(output)
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(params["model_path"], local_files_only=True)
        text = self.tokenizer.decode(output[state["start"]:], skip_special_tokens=False)
        audit = reasoning_loop(text)
        if audit is None:
            return None
        state["forcing"] = len(output)
        print(json.dumps({"thinking_loop_forced": True, "rid": req.rid,
                          "output_tokens": len(output), "audit": audit}), flush=True)
        return params["newline_id"]

    def __call__(self, logits, custom_param_list=None):
        for index, params in enumerate(custom_param_list or []):
            if not params or "think_end_id" not in params:
                continue
            token = self.next_token(params["__req__"], params)
            if token is not None:
                logits[index, :] = -float("inf")
                logits[index, token] = 0.0
        return logits


def request_control(tokenizer, model_path):
    import dill

    return {"custom_logit_processor": json.dumps({"callable": dill.dumps(ThinkingLoopProcessor).hex()}),
            "custom_params": {"model_path": str(model_path),
                "think_start_id": tokenizer.convert_tokens_to_ids("<think>"),
                "think_end_id": tokenizer.convert_tokens_to_ids("</think>"),
                "newline_id": tokenizer.encode("\n", add_special_tokens=False)[0]}}
