"""Audit a proposed input protocol without activating or rewriting SFT examples."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from .audit_context_budget import render, template_messages, tool_result
from .audit_context_references import quoted_gaps
from .check_prefix_chains import Reader
from .context_projection import PROTOCOL, THINKING_PROTOCOL, select_history, strip_history_thinking
from .data import load_manifest


def image_expansions(path):
    records = json.loads(path.read_text())["records"]
    return {row["sample_id"]: row["native_image_verification"]["token_count"] -
            row["variants"]["protected_tool_text_keep_16"]["text_tokens"] for row in records}


def image_parts(messages):
    return [part for message in messages if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") in ("image", "image_url")]


def verify_invariants(processor):
    """Exercise the installed template, including ambiguous inline-thinking cases."""
    examples = [
        {"content": "visible answer", "reasoning_content": "OLD_EXPLICIT_REASONING"},
        {"content": "<think>OLD_INLINE_REASONING</think>\nvisible answer"},
        {"content": "<think>unclosed literal"},
        {"content": "```xml\n<think>literal</think>\n```", "reasoning_content": "OLD_CODE_REASONING"},
        {"content": "<think>literal</think>", "reasoning_content": "OLD_FIELD_REASONING"},
        {"content": None, "reasoning_content": "OLD_CALL_REASONING", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
    ]
    checked, native_flags = 0, {}
    for preserve in (False, True):
        for thinking in (False, True):
            for example in examples:
                messages = [{"role": "system", "content": "instructions"},
                            {"role": "user", "content": [{"type": "text", "text": "reference"}, {"type": "image"}]},
                            {"role": "assistant", **example}, {"role": "user", "content": "next task"},
                            {"role": "assistant", **example}]
                original = deepcopy(messages)
                stripped = strip_history_thinking(messages)
                assert messages == original and strip_history_thinking(stripped) == stripped
                assert image_parts(messages) == image_parts(stripped)
                assert [m.get("tool_calls") for m in messages] == [m.get("tool_calls") for m in stripped]
                assert [m for m in messages if m["role"] != "assistant"] == [
                    m for m in stripped if m["role"] != "assistant"]
                if isinstance(example.get("reasoning_content"), str) or "</think>" not in (example["content"] or ""):
                    assert stripped[2]["content"] == example["content"]
                kwargs = {"preserve_thinking": preserve, "enable_thinking": thinking}
                before = processor.apply_chat_template(template_messages(messages), tokenize=False,
                                                       add_generation_prompt=True, **kwargs)
                after = processor.apply_chat_template(template_messages(stripped), tokenize=False,
                                                      add_generation_prompt=True, **kwargs)
                if example is examples[0]:
                    native_flags[preserve, thinking] = before
                assert before.rsplit("<|im_start|>assistant", 1)[1] == after.rsplit("<|im_start|>assistant", 1)[1]
                assert "OLD_" not in after
                if example["content"] and "literal" in example["content"]:
                    assert "literal" in after
                target = [{"role": "assistant", "content": "finished", "reasoning_content": "current" if thinking else ""}]
                first = render(processor, template_messages(messages), target, [], kwargs)
                second = render(processor, template_messages(stripped), target, [], kwargs)
                assert first[1:] == second[1:]
                checked += 1
    opened, active = [], [True]

    def observe(event, arguments):
        if active[0] and event == "open":
            opened.append(str(arguments[0]))

    sys.addaudithook(observe)
    try:
        select_history(messages, lambda values: len(json.dumps(values)), context=100,
                       reserve=10, margin=1, strip_thinking=True)
    finally:
        active[0] = False
    assert not opened
    return {"native_template_cases": checked, "target_bytes_and_tokens_unchanged": True,
            "visible_content_and_calls_preserved": True, "images_preserved": True,
            "generation_prompt_tail_preserved": True, "unclosed_and_literal_tags_preserved": True,
            "pure_transform_file_opens": len(opened),
            "preserve_thinking_changes_native_prefix": native_flags[False, True] != native_flags[True, True],
            "enable_thinking_changes_native_prefix": native_flags[False, True] != native_flags[False, False],
            "history_thinking_strip_idempotent": True, "semantic_sufficiency_certified": False}


def audit_row(processor, reader, row, expansion, args):
    request = reader.value(row["request_ref"])
    response = reader.value(row["response_ref"])["choices"][0]["message"]
    prompt, completion = template_messages(request["messages"]), template_messages([response])
    options = {**row.get("chat_template_kwargs", {}), "tools": request.get("tools", []), "tokenize": False}
    _, target_hash, target_count = render(
        processor, prompt, completion, request.get("tools", []), row.get("chat_template_kwargs", {}))
    if target_count != row["completion_tokens"]:
        raise ValueError("Original target count differs: " + row["sample_id"])
    if "prompt_tokens" in row:
        text = processor.apply_chat_template(prompt, add_generation_prompt=True, **options)
        expansion = row["prompt_tokens"] - len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def count(messages):
        text = processor.apply_chat_template(messages, add_generation_prompt=True, **options)
        return len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]) + expansion

    candidate, details = select_history(prompt, count, context=args.context, reserve=args.reserve,
                                         margin=args.margin, max_new_tokens=args.max_new_tokens,
                                         strip_thinking=args.strip_history_thinking)
    record = {"sample_id": row["sample_id"], "split": row["split"],
              "previously_rejected": "rejection" in row, "target_tokens": target_count,
              "closing_reply": not response.get("tool_calls"),
              "target_sha256": target_hash, "image_expansion_tokens": expansion, **details}
    if "prompt_tokens" in row and record["original_prompt_tokens"] != row["prompt_tokens"]:
        raise ValueError("Original prompt count differs: " + row["sample_id"])
    if candidate is None:
        return record
    if len(candidate) != len(prompt) or image_parts(candidate) != image_parts(prompt):
        raise ValueError("Projection changed message/image order: " + row["sample_id"])
    expected = strip_history_thinking(prompt) if details.get("history_thinking_removed") else prompt
    if any(after != before for before, after in zip(expected, candidate) if not tool_result(before)):
        raise ValueError("Projection changed a non-result message: " + row["sample_id"])
    _, candidate_hash, candidate_targets = render(
        processor, candidate, completion, request.get("tools", []), row.get("chat_template_kwargs", {}))
    if (candidate_hash, candidate_targets) != (target_hash, target_count):
        raise ValueError("Projection changed current supervision: " + row["sample_id"])
    before = processor.apply_chat_template(prompt, add_generation_prompt=True, **options)
    after = processor.apply_chat_template(candidate, add_generation_prompt=True, **options)
    if before.count("<|image_pad|>") != after.count("<|image_pad|>"):
        raise ValueError("Projection changed image placeholders: " + row["sample_id"])
    record.update(target_unchanged=True, images_unchanged=True,
                  target_fits_remaining=target_count <= args.context - details["prompt_tokens"] - args.margin,
                  prompt_after_sha256=sha256(after.encode()).hexdigest(),
                  reference_audit=quoted_gaps(before, after, response.get("reasoning_content") or ""))
    if details.get("history_thinking_removed"):
        # Audit current-label support after selection; never use it to choose a stage/K.
        stripped = processor.apply_chat_template(strip_history_thinking(prompt),
                                                add_generation_prompt=True, **options)
        target_text = (response.get("reasoning_content") or "") + "\n" + (response.get("content") or "")
        for call in response.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            target_text += "\n" + (arguments if isinstance(arguments, str) else json.dumps(arguments))
        record["history_thinking_reference_audit"] = quoted_gaps(
            before, stripped, target_text)
    return record


def audit(args):
    from transformers import AutoProcessor

    rows = json.loads(args.manifest.read_text())
    accepted = load_manifest(args.accepted_manifest)
    triggered = [row for row in accepted if row["prompt_tokens"] + args.reserve + args.margin > args.context]
    selected = [row for row in accepted if row["prompt_tokens"] +
                (args.selection_reserve or args.reserve) + args.margin > args.context]
    rows += selected
    expansions = image_expansions(args.image_expansion_report)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "invariants.json").write_text(json.dumps(verify_invariants(processor), indent=2) + "\n")
    readers, records = {}, []
    with (args.output / "decisions.jsonl").open("w") as stream:
        for row in rows:
            root = row["snapshot_root"]
            if root not in readers:
                readers[root] = Reader(root)
            record = audit_row(processor, readers[root], row, expansions.get(row["sample_id"]), args)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            if len(records) % 25 == 0:
                print(json.dumps({"measured": len(records), "total": len(rows),
                                  "fits": sum(row["fits"] for row in records)}), flush=True)
    successful = [row for row in records if row["fits"]]
    summary = {"at": time.time(), "protocol": THINKING_PROTOCOL if args.strip_history_thinking else PROTOCOL,
               "records": len(records), "selected_accepted_records": len(selected),
               "accepted_records_examined_for_trigger": len(accepted), "additional_triggers": len(triggered),
               "fits": len(successful), "failures": len(records) - len(successful),
               "fits_by_prior_status": dict(Counter("rejected" if row["previously_rejected"] else "accepted"
                                                    for row in successful)),
               "keep_results": dict(Counter(row.get("keep_results", "unchanged") for row in successful)),
               "stages": dict(Counter(row["stage"] for row in successful)),
               "remaining_reference_flags": sum(bool(row["reference_audit"]["missing_after"]) for row in successful),
               "history_thinking_reference_flags": sum(bool(row.get("history_thinking_reference_audit", {})
                                                              .get("missing_after")) for row in successful),
               "target_does_not_fit_after_selection": sum(not row["target_fits_remaining"] for row in successful),
               "fitting_rejected_train_closings": sum(row["previously_rejected"] and row["split"] == "train"
                                                      and row["closing_reply"] for row in successful),
               "all_successful_targets_and_images_unchanged": all(
                   row["target_unchanged"] and row["images_unchanged"] for row in successful),
               "prompt_token_range": [min(row["prompt_tokens"] for row in successful),
                                      max(row["prompt_tokens"] for row in successful)] if successful else [],
               "source_sha256": sha256(Path(__file__).with_name("context_projection.py").read_bytes()).hexdigest(),
               "context": args.context, "reserve": args.reserve, "margin": args.margin,
               "projection_uses_future_target": False, "training_modified": False,
               "training_examples_recovered": False, "semantic_sufficiency_certified": False,
               "method": "Native text tokenization plus previously verified fixed image expansion; no new image preprocessing."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "accepted-manifest", "image-expansion-report", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--reserve", type=int, default=49152)
    parser.add_argument("--selection-reserve", type=int)
    parser.add_argument("--strip-history-thinking", action="store_true")
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=131072)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
