"""Compare archived base/SFT behavior with the exact supervised token partitions."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from .detect_loop import reasoning_loop
from .data import open_store
from .summarize_capture import summarize


RUNS = {
    "base9b": "base9b-small96-dsh-flash-20261124-causal-control",
    "step160": "step160-small96-dsh-flash-20261124",
    "step170": "step170-small96-dsh-flash-20261124-sharedtp",
    "step170_resume": "step170-small96-dsh-flash-20261124-resume-budget",
    "step200": "step200-small96-dsh-flash-20261124-fixed-runtime",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return {"n": len(values), "mean": sum(values) / len(values),
            **{name: values[round((len(values) - 1) * fraction)]
               for name, fraction in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))},
            "max": values[-1]}


def training_summary(rows):
    partition = Counter()
    for row in rows:
        partition.update(row["partition_tokens"])
    total = sum(partition.values())
    return {"decisions": len(rows), "partition_tokens": dict(partition),
            "thinking_fraction": partition["reasoning"] / total,
            "prompt_tokens": quantiles(row["prompt_tokens"] for row in rows),
            "thinking_tokens": quantiles(row["partition_tokens"].get("reasoning", 0) for row in rows),
            "nonthinking_tokens": quantiles(row["target_tokens"] - row["partition_tokens"].get("reasoning", 0)
                                             for row in rows),
            "thinking_over_target": quantiles(row["partition_tokens"].get("reasoning", 0) / row["target_tokens"]
                                               for row in rows)}


def export_training(rows, manifest, tokenizer, output):
    indexed = {row["sample_id"]: row for row in manifest}
    stores, records, order = {}, [], Counter()
    for row in rows:
        source = indexed[row["sample_id"]]
        root = source["snapshot_root"]
        if root not in stores:
            stores[root] = open_store(root)
        message = stores[root].get(source["response_ref"])["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        arguments = [call["function"]["arguments"] for call in calls]
        order[source["episode"]] += 1
        partition = row["partition_tokens"]
        records.append({"split": source["split"], "episode": source["episode"],
            "decision_index": order[source["episode"]], "sample_id": row["sample_id"], "stage": row["stage"],
            "components": source["final_components"], "input_tokens": row["prompt_tokens"],
            "thinking_tokens_native": partition.get("reasoning", 0),
            "tool_calls_tokens_native": partition.get("tool_call", 0),
            "visible_answer_tokens_native": partition.get("visible_content", 0),
            "format_tokens_native": partition.get("format", 0), "target_tokens_native": row["target_tokens"],
            "tool_arguments_tokens_retokenized": sum(len(tokenizer.encode(
                value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                add_special_tokens=False)) for value in arguments),
            "visible_answer_tokens_retokenized": len(tokenizer.encode(message.get("content") or "", add_special_tokens=False)),
            "tools": ",".join(call["function"]["name"] for call in calls), "target_sha256": row["target_sha256"]})
        if len(records) % 1000 == 0:
            (output / "progress.json").write_text(json.dumps({"training_rows": len(records), "total": len(rows)}))
    write_csv(output / "sft-decisions.csv", records)
    write_csv(output / "same-image-teacher.csv", [row for row in records if row["episode"] == "ebf9d337d02a5300"])


def write_csv(path, records):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def tool_detail(tool):
    raw = tool.get("arguments", "")
    try:
        args = json.loads(raw)
    except ValueError:
        return {"name": tool["name"], "arguments_incomplete": True}
    return {"name": tool["name"], "description": args.get("description", "")[:140],
            "file_path": args.get("file_path", args.get("path")),
            "command_excerpt": args.get("command", "")[:180],
            "crop": ".crop(" in raw, "resize": ".resize(" in raw,
            "pixel_measurement": any(value in raw for value in ("getpixel", "np.where", "np.array", "ImageFont")),
            "normalized_call_sha256": sha256(json.dumps(
                {"name": tool["name"], "arguments": {k: v for k, v in args.items() if k != "description"}},
                sort_keys=True).encode()).hexdigest()}


def live_records(root, tokenizer):
    records, metadata = [], {}
    for label, name in RUNS.items():
        folder = root / "runs/causal-cot-20261122" / name / "evaluation"
        audit = read_rows(folder / "generator-context-requests.jsonl")
        protocol = json.loads((folder / "protocol.json").read_text())
        metadata[label] = {key: protocol.get(key) for key in (
            "reference_sha256", "source_commit", "state_prefix_policy", "generator_sampling",
            "generator_context", "gold_available", "mcp", "build_after_tools", "thinking_loop_recovery")}
        for index, row in enumerate(summarize(folder / "generator-context-requests-raw",
                                              audit=audit, tokenizer=tokenizer), 1):
            reply, state = row["response"], row["state_prefix"]
            reasoning = reply["reasoning_text"]
            repeats = Counter(part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", reasoning)
                              if 8 <= len(part.strip()) <= 200)
            loop = reasoning_loop(reasoning)
            records.append({"run": label, "request_index": index, "request_sha256": row["request_sha256"],
                "request": row["request"], "response_sha256": row["response_sha256"],
                "input_tokens": row["prompt_tokens"], "input_token_source": row["prompt_token_source"],
                "max_output_tokens": row["requested_max_tokens"],
                "completion_tokens": row["completion_tokens"], "thinking_tokens": row["reasoning_tokens"],
                "nonthinking_tokens": (row["completion_tokens"] - row["reasoning_tokens"]
                    if row["completion_tokens"] is not None and row["reasoning_tokens"] is not None else None),
                "retokenized": row["retokenized"], "finish_reason": reply.get("finish_reason"),
                "complete_stream": row["complete_stream"], "loop": loop,
                "repeated_phrases": [{"text": text, "occurrences": count}
                    for text, count in repeats.most_common(3) if count > 1],
                "reasoning_tail_excerpt": reasoning[-400:] if loop or not row["complete_stream"] else "",
                "tools": [tool_detail(tool) for tool in reply["tool_calls"].values()],
                "xml_present_before": any(name.endswith(".drawio") for name in state.get("files_sha256", {})),
                "history_thinking_messages": state.get("history_thinking_messages"),
                "state_policy": state.get("policy"),
                "images": [{"sha256": image.get("sha256"), "bytes": image.get("bytes")}
                           for image in row["image_inputs"]]})
    return records, metadata


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.root / "checks/dsh-midflight-20261123/curriculum-final.jsonl"
    rows = read_rows(args.root / "checks/rewritten-supervision-20261124/train/decisions.jsonl")
    validation = read_rows(args.root / "checks/rewritten-supervision-20261124/validation/decisions.jsonl")
    seen = read_rows(args.root / "checks/stopped218-supervised-actions-20261124/decisions.jsonl")
    indexed = {row["sample_id"]: row for row in rows}
    consumed = [indexed[row["sample_id"]] for row in seen]
    effects = {row["sample_id"]: row for row in read_rows(
        args.root / "checks/teacher180-resource-audit-v1/decisions.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(args.root / "models/Qwen3.5-9B", local_files_only=True)
    records, metadata = live_records(args.root, tokenizer)
    with (args.output / "calls.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in records)
    write_csv(args.output / "live-calls.csv", [{
        **{key: row[key] for key in ("run", "request_index", "input_tokens", "input_token_source",
            "max_output_tokens", "completion_tokens", "thinking_tokens", "nonthinking_tokens", "finish_reason", "complete_stream")},
        "thinking_tokens_retokenized": row["retokenized"]["reasoning"],
        "tool_arguments_tokens_retokenized": row["retokenized"]["tool_arguments"],
        "visible_answer_tokens_retokenized": row["retokenized"]["content"],
        "tools": ",".join(tool["name"] for tool in row["tools"]),
        "loop": bool(row["loop"]), "xml_present_before": row["xml_present_before"]} for row in records])
    export_training(rows + validation, read_rows(manifest), tokenizer, args.output)
    training = {"all_train": training_summary(rows), "consumed_through_218": training_summary(consumed)}
    training["same_image_teacher"] = training_summary([row for row in validation if row["episode"] == "ebf9d337d02a5300"])
    for label, selected in (("before_initial_xml", [row for row in rows if row["stage"] == 1
                and effects[row["sample_id"]]["xml_present_before"] is False]),
            ("xml_changes", [row for row in rows if effects[row["sample_id"]]["artifact_effect"] == "native_xml_changed"])):
        training[label] = training_summary(selected)
    live = {}
    for label in RUNS:
        selected = [row for row in records if row["run"] == label]
        complete = [row for row in selected if row["completion_tokens"] is not None]
        thinking = sum(row["thinking_tokens"] or 0 for row in complete)
        total = sum(row["completion_tokens"] for row in complete)
        live[label] = {"calls": len(selected), "provider_usage_calls": len(complete),
            "input_tokens": quantiles(row["input_tokens"] for row in selected),
            "thinking_tokens": quantiles(row["thinking_tokens"] for row in complete),
            "nonthinking_tokens": quantiles(row["nonthinking_tokens"] for row in complete),
            "thinking_fraction_complete_calls": thinking / total if total else None,
            "loop_request_indices": [row["request_index"] for row in selected if row["loop"]],
            "tools": dict(Counter(tool["name"] for row in selected for tool in row["tools"])),
            "first_request_with_existing_xml": next((row["request_index"] for row in selected if row["xml_present_before"]), None),
            "last_call": selected[-1]}
    report = {"at": time.time(), "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "model_requests": 0, "training_changed": False, "training": training, "live": live,
        "protocols": metadata, "limitations": [
            "Base and SFT are existing single-image trajectories with different generated prefixes and state policy versions.",
            "Complete-call token counts use provider usage; interrupted loops expose only separately retokenized observed text.",
            "Training partitions use native target offsets, while provider nonthinking counts also include native framing.",
            "Length ratios and repeated inspection are diagnostic evidence, not proof of a unique causal mechanism.",
            "Calls copied into the 170 recovery are retained for within-run chronology; do not pool runs as independent samples."]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"calls": len(records), "training_decisions": len(rows), "seen": len(consumed),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
