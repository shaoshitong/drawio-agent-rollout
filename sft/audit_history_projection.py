"""Audit a proposed input protocol without activating or rewriting SFT examples."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from .audit_context_budget import render, template_messages, tool_result
from .audit_context_references import quoted_gaps
from .check_prefix_chains import Reader
from .context_projection import PROTOCOL, select_history
from .data import load_manifest


def image_expansions(path):
    records = json.loads(path.read_text())["records"]
    return {row["sample_id"]: row["native_image_verification"]["token_count"] -
            row["variants"]["protected_tool_text_keep_16"]["text_tokens"] for row in records}


def image_parts(messages):
    return [part for message in messages if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") in ("image", "image_url")]


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
                                         margin=args.margin, max_new_tokens=args.max_new_tokens)
    record = {"sample_id": row["sample_id"], "split": row["split"],
              "previously_rejected": "rejection" in row, "target_tokens": target_count,
              "target_sha256": target_hash, "image_expansion_tokens": expansion, **details}
    if "prompt_tokens" in row and record["original_prompt_tokens"] != row["prompt_tokens"]:
        raise ValueError("Original prompt count differs: " + row["sample_id"])
    if candidate is None:
        return record
    if len(candidate) != len(prompt) or image_parts(candidate) != image_parts(prompt):
        raise ValueError("Projection changed message/image order: " + row["sample_id"])
    if any(after != before for before, after in zip(prompt, candidate) if not tool_result(before)):
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
                  prompt_after_sha256=sha256(after.encode()).hexdigest(),
                  reference_audit=quoted_gaps(before, after, response.get("reasoning_content") or ""))
    return record


def audit(args):
    from transformers import AutoProcessor

    rows = json.loads(args.manifest.read_text())
    accepted = load_manifest(args.accepted_manifest)
    triggered = [row for row in accepted if row["prompt_tokens"] + args.reserve + args.margin > args.context]
    rows += triggered
    expansions = image_expansions(args.image_expansion_report)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
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
    summary = {"at": time.time(), "protocol": PROTOCOL, "records": len(records),
               "accepted_records_examined_for_trigger": len(accepted), "additional_triggers": len(triggered),
               "fits": len(successful), "failures": len(records) - len(successful),
               "fits_by_prior_status": dict(Counter("rejected" if row["previously_rejected"] else "accepted"
                                                    for row in successful)),
               "keep_results": dict(Counter(row.get("keep_results", "unchanged") for row in successful)),
               "remaining_reference_flags": sum(bool(row["reference_audit"]["missing_after"]) for row in successful),
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
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=131072)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
