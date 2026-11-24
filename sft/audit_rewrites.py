"""Audit frozen reasoning sidecars; never send answer-dependent feedback to the editor."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .audit_context_budget import template_messages, tool_result
from .causal_cot import anchors, answer_fields, apply_rewrite, digest, numbers, protected_values
from .check_prefix_chains import Reader
from .data import load_manifest, open_store
from .detect_loop import reasoning_loop
from .history import HISTORY_POLICY, strip_history_thinking
from .observation_context import apply_observations, text_parts


def visible_text(message):
    content = message.get("content") or ""
    parts = [content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)]
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        parts.append(arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False))
    return "\n".join(parts)


def native_answer(processor, request, response, options, include_reasoning=False):
    prompt = strip_history_thinking(template_messages(request["messages"]))
    completion = template_messages([response["choices"][0]["message"]])
    if not include_reasoning:
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
    state = overlay.get("state_prefix")
    expected_history = state["messages"] if state else apply_observations(original_request["messages"], overlay.get("observations", []))
    if state:
        from .state_prefix import snapshot_before_request

        files, snapshot = snapshot_before_request(row, reader.store)
        if state["snapshot"] != snapshot or any(files[name] != value for name, value in state["files_sha256"].items()):
            raise ValueError("State prefix file hashes differ from the exact pre-request snapshot")
    if len(request["messages"]) != len(expected_history) or any(
            answer_fields(a) != answer_fields(b)
            for a, b in zip(request["messages"], expected_history)):
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
            "severe_reasoning_loop": reasoning_loop(rewritten),
            "answer_fields_exact": True, "history_changes_match_audited_observations": True,
            "observation_summaries": len(overlay.get("observations", [])), "native_answer_exact": True}


def audit_wire(root, store, rows, readers):
    by_id = {row["sample_id"]: row for row in rows}
    by_request = {row["request_ref"]: row for row in rows}
    counts = Counter()
    paths = list((root / "episodes").glob("*/requests.jsonl"))
    paths += list((root / "context-cache").glob("requests.jsonl"))
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            body = store.get(record["request_ref"])
            data = json.loads(body["messages"][1]["content"])
            if len(body["messages"]) != 2 or [m["role"] for m in body["messages"]] != ["system", "user"]:
                raise ValueError("Editor input contains a conversation history")
            if set(data) - {"source_fragment", "earlier_compact", "candidate", "previous_proposal", "verification_feedback", "required_source_anchors"}:
                raise ValueError("Unexpected editor evidence field")
            if "required_source_anchors" in data and data["required_source_anchors"] != sorted(protected_values(data["source_fragment"])):
                raise ValueError("Required anchors are not derived solely from the current source")
            if digest(data["source_fragment"]) != record["source_sha256"] or digest(data["earlier_compact"]) != record["previous_sha256"]:
                raise ValueError("Actual editor source/previous fragment differs from its provenance")
            row = by_id.get(record["sample_id"]) or by_request[record["archived_request_ref"]]
            reader = readers[row["snapshot_root"]]
            if record["sample_id"].startswith("observation:"):
                request = reader.value(record["archived_request_ref"])
                sources = [text for message in request["messages"] if message["role"] in ("assistant", "tool", "user")
                           for _, text in text_parts(message)]
                sources = [text[len("<tool_response>"):-len("</tool_response>")]
                           if text.startswith("<tool_response>") and text.endswith("</tool_response>") else text
                           for text in sources]
            elif record.get("reasoning_source"):
                reference = record["reasoning_source"]
                previous = open_store(reference["root"]).get(reference["ref"])
                if any(previous[key] != row[key] for key in ("request_ref", "response_ref")):
                    raise ValueError("Recompaction source belongs to another decision")
                sources = [previous["current_reasoning"]]
            else:
                response = reader.value(row["response_ref"])
                sources = [response["choices"][0]["message"].get("reasoning_content") or ""]
            if not any(source[record["source_start"]:record["source_end"]] == data["source_fragment"] for source in sources):
                raise ValueError("Editor fragment is not the declared original reasoning/observation span")
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
    rows = load_manifest(args.manifest)
    for row in rows:
        source = row["snapshot_root"]
        if source not in readers:
            readers[source] = Reader(source)
        if "reasoning_rewrite" not in row:
            continue
        rewrite = row["reasoning_rewrite"]["root"]
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
               "current_answers_exact_and_history_changes_audited": True,
               "state_rejected_decisions": sum(bool(row.get("state_rejection")) for row in rows),
               "wire": {root: audit_wire(Path(root).parent, store, rows, readers)
                   for root, store in stores.items()},
               "model_quality_acceptance": False, "answer_feedback_used_for_compression": False}
    args.output.write_text(json.dumps({"summary": summary, "decisions": records}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
