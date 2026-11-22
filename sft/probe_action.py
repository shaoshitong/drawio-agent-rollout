"""Generate one diagnostic action from an unchanged archived multimodal prefix."""

import argparse
import json
from pathlib import Path
import re
import time

from .probe_parity import load_checkpoint, native_prompt


class ActionProgress:
    def __init__(self, output, prompt_tokens, terminator):
        self.output, self.prompt_tokens, self.terminator = output, prompt_tokens, terminator

    def __call__(self, input_ids, scores, **kwargs):
        count = input_ids.shape[-1] - self.prompt_tokens
        finished = input_ids[0, -len(self.terminator):].tolist() == self.terminator
        if finished or count % 128 == 0:
            record = {"generated_tokens": count, "first_tool_call_closed": finished, "at": time.time()}
            (self.output / "progress.json").write_text(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        return finished


def run(args):
    import torch
    from setproctitle import setproctitle
    from transformers import StoppingCriteriaList

    setproctitle("python")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["sample_id"] == args.sample_id)
    processor, model, _ = load_checkpoint(args)
    batch = native_prompt(processor, row, args.probes)
    batch = {key: value.to("cuda") for key, value in batch.items()}
    prompt_tokens = batch["input_ids"].shape[-1]
    stop = ActionProgress(args.output, prompt_tokens,
                          processor.tokenizer.encode("</tool_call>", add_special_tokens=False))
    started = time.time()
    print(json.dumps({"generation_started": True, "sample_id": args.sample_id,
                      "variant": args.variant, "prompt_tokens": prompt_tokens}), flush=True)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    eos = sorted({token for token in [*eos, processor.tokenizer.eos_token_id] if token is not None})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sequences = model.generate(**batch, do_sample=False, use_cache=True, logits_to_keep=1,
            max_new_tokens=args.max_new_tokens, eos_token_id=eos,
            stopping_criteria=StoppingCriteriaList([stop]))
    continuation = sequences[0, prompt_tokens:]
    text = processor.tokenizer.decode(continuation, skip_special_tokens=False)
    action_text = text.split("</think>", 1)[-1]
    closed = "<tool_call>" in action_text and "</tool_call>" in action_text
    record = {"sample_id": args.sample_id, "variant": args.variant, "prompt_tokens": prompt_tokens,
        "generated_tokens": len(continuation), "max_new_tokens": args.max_new_tokens, "eos_token_ids": eos,
        "finish_reason": "first_tool_call" if closed else "eos" if int(continuation[-1]) in eos else "length",
        "tool_names": re.findall(r"<function=([^>]+)>", action_text),
        "crop_marker": bool(re.search(r"\.crop\s*\(", action_text)),
        "native_xml_marker": bool(re.search(r"<(?:mxfile|mxGraphModel|mxCell)(?:\s|>)", action_text)),
        "seconds": time.time() - started, "generation_text": text,
        "scope": "One greedy action, diagnostic output cap; no tool execution, no full rollout. "
                 "Only the archived prompt and original images enter the model; the recorded target is not supplied."}
    (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "generation_text"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "model", "adapter", "probes", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("base", "probes", "output"))
    parser.add_argument("--variant", choices=("base", "peft", "merged", "full"), required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
