"""Audit frozen reasoning sidecars; never send answer-dependent feedback to the editor."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .audit_context_budget import template_messages
from .causal_cot import anchors, answer_fields, apply_rewrite, digest, numbers
from .check_prefix_chains import Reader
from .data import load_manifest, open_store
from .history import HISTORY_POLICY, strip_history_thinking


def visible_text(message):
    content = message.get("content") or ""
    parts = [content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)]
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        parts.append(arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False))
    return "\n".join(parts)


def native_answer(processor, request, response, options):
    prompt = strip_history_thinking(template_messages(request["messages"]))
    completion = template_messages([response["choices"][0]["message"]])
    completion[0]["reasoning_content"] = ""
    prefix = processor.apply_chat_template(prompt, add_generation_prompt=True, **options)
    full = processor.apply_chat_template(prompt + completion, add_generation_prompt=False, **options)
    if not full.startswith(prefix):
        raise ValueError("Current answer changes its native conditioning prefix")
    return full[len(prefix):]


def audit_row(row, reader, store, processor):
    original_request = reader.value(row["request_ref"])
    original_response = reader.value(row["response_ref"])
    overlay = store.get(row["reasoning_rewrite"]["ref"])
    request, response = apply_rewrite(original_request, original_response, row, overlay)
    original = original_response["choices"][0]["message"]
    compact = response["choices"][0]["message"]
    if answer_fields(original) != answer_fields(compact):
        raise ValueError("Current nonreasoning fields changed: " + row["sample_id"])
    if len(request["messages"]) != len(original_request["messages"]) or any(
            answer_fields(a) != answer_fields(b)
            for a, b in zip(request["messages"], original_request["messages"])):
        raise ValueError("Historical nonreasoning fields changed: " + row["sample_id"])
    options = {**row.get("chat_template_kwargs", {}), "tokenize": False,
               "tools": original_request.get("tools", [])}
    before = native_answer(processor, original_request, original_response, options)
    after = native_answer(processor, request, response, options)
    if before != after:
        raise ValueError("Native answer serialization changed: " + row["sample_id"])
    source = original.get("reasoning_content") or ""
    rewritten = compact.get("reasoning_content") or ""
    count = lambda text: len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
    # These are answer-overlap candidates, not demonstrated causal dependencies.
    # The audit is separate from compression: no retry, filtering or selection uses them.
    overlap = numbers(source) & numbers(visible_text(original))
    return {"sample_id": row["sample_id"], "split": row["split"],
            "source_reasoning_tokens": count(source), "rewritten_reasoning_tokens": count(rewritten),
            "answer_sha256": digest(answer_fields(original)), "native_answer_sha256": digest(before),
            "answer_overlap_values": len(overlap), "omitted_answer_overlap_values": len(overlap - numbers(rewritten)),
            "omitted_source_anchors": len(anchors(source) - anchors(rewritten)),
            "answer_fields_exact": True, "history_nonreasoning_fields_exact": True, "native_answer_exact": True}


def audit_wire(root, store):
    counts = Counter()
    for path in sorted((root / "episodes").glob("*/requests.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            body = store.get(record["request_ref"])
            data = json.loads(body["messages"][1]["content"])
            if len(body["messages"]) != 2 or [m["role"] for m in body["messages"]] != ["system", "user"]:
                raise ValueError("Editor input contains a conversation history")
            if set(data) - {"source_fragment", "earlier_compact", "candidate", "previous_proposal", "verification_feedback"}:
                raise ValueError("Unexpected editor evidence field")
            if digest(data["source_fragment"]) != record["source_sha256"] or digest(data["earlier_compact"]) != record["previous_sha256"]:
                raise ValueError("Actual editor source/previous fragment differs from its provenance")
            if digest(body) != record["request_sha256"]:
                raise ValueError("Actual request body digest differs")
            counts["requests"] += 1
            counts["errors"] += bool(record.get("error"))
    return dict(counts)


def main():
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    readers, stores, records = {}, {}, []
    for row in load_manifest(args.manifest):
        source, rewrite = row["snapshot_root"], row["reasoning_rewrite"]["root"]
        if source not in readers:
            readers[source] = Reader(source)
        if rewrite not in stores:
            stores[rewrite] = open_store(rewrite)
        records.append(audit_row(row, readers[source], stores[rewrite], processor))
    source_tokens = sum(row["source_reasoning_tokens"] for row in records)
    compact_tokens = sum(row["rewritten_reasoning_tokens"] for row in records)
    overlap = sum(row["answer_overlap_values"] for row in records)
    omitted = sum(row["omitted_answer_overlap_values"] for row in records)
    summary = {"at": time.time(), "decisions": len(records), "source_reasoning_tokens": source_tokens,
               "history_policy": HISTORY_POLICY,
               "rewritten_reasoning_tokens": compact_tokens,
               "reasoning_reduction": 1 - compact_tokens / source_tokens if source_tokens else 0,
               "answer_overlap_values": overlap, "omitted_answer_overlap_values": omitted,
               "omitted_overlap_fraction": omitted / overlap if overlap else 0,
               "overlap_scope": "Unique numeric overlap per decision, not proof of action dependencies; corpus-level reporting only",
               "nonreasoning_fields_and_native_answers_exact": True,
               "wire": {root: audit_wire(Path(root).parent, store) for root, store in stores.items()},
               "model_quality_acceptance": False, "answer_feedback_used_for_compression": False}
    args.output.write_text(json.dumps({"summary": summary, "decisions": records}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
