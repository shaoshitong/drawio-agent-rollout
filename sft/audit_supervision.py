"""Audit supervision density and repetition without changing training examples."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from .audit_actions import category
from .data import ManifestDataset
from .export_sft import normalize
from .train_unsloth import render_completion


def partition_target(processor, row, message):
    completion, _ = normalize([message], decode_images=False)
    completion[0]["content"] = completion[0].get("content") or ""
    example = {
        "prompt": [{"role": "user", "content": "Audit prompt."}],
        "completion": completion,
        "chat_template_kwargs": {**row.get("chat_template_kwargs", {}), "enable_thinking": True},
    }
    _, text = render_completion(processor, example)
    encoded = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) != row["completion_tokens"]:
        raise ValueError("Audit target differs from prepared length: " + row["sample_id"])
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning and not text.startswith(reasoning):
        raise ValueError("Reasoning does not begin the target: " + row["sample_id"])
    spans = [(0, len(reasoning), "reasoning")] if reasoning else []
    calls = list(re.finditer(r"<tool_call>.*?</tool_call>", text[len(reasoning):], re.S))
    if len(calls) != len(message.get("tool_calls", [])):
        raise ValueError("Cannot identify complete tool-call spans: " + row["sample_id"])
    spans.extend((len(reasoning) + m.start(), len(reasoning) + m.end(), "tool_call") for m in calls)
    content = (message.get("content") or "").strip()
    if content:
        start = text.find(content, len(reasoning))
        if start < 0:
            raise ValueError("Cannot locate visible answer: " + row["sample_id"])
        spans.append((start, start + len(content), "visible_content"))
    counts = Counter()
    for start, end in encoded["offset_mapping"]:
        label = next((kind for low, high, kind in spans if start < high and end > low), "format")
        counts[label] += 1
    paragraphs = Counter(part.strip() for part in reasoning.split("\n\n") if part.strip())
    repeated = sum(len(part) * (count - 1) for part, count in paragraphs.items())
    return {
        "target_sha256": sha256(text.encode()).hexdigest(),
        "partition_tokens": dict(counts),
        "reasoning_characters": len(reasoning),
        "max_identical_reasoning_paragraph_repeats": max(paragraphs.values(), default=0),
        "repeated_reasoning_character_fraction": repeated / max(1, len(reasoning)),
    }


def summarize(records):
    tokens = sum(row["target_tokens"] for row in records)
    partitions = Counter()
    categories = defaultdict(Counter)
    for row in records:
        partitions.update(row["partition_tokens"])
        categories[row["category"]].update(row["partition_tokens"])
    ordered = sorted(records, key=lambda row: row["target_tokens"], reverse=True)
    tails = {}
    for fraction in (0.01, 0.05, 0.10):
        count = math.ceil(len(ordered) * fraction)
        tails[str(fraction)] = {
            "decisions": count,
            "target_token_fraction": sum(row["target_tokens"] for row in ordered[:count]) / tokens,
        }
    return {
        "decisions": len(records),
        "episodes": len({row["episode"] for row in records}),
        "target_tokens": tokens,
        "partition_tokens": dict(partitions),
        "partition_fractions": {name: value / tokens for name, value in partitions.items()},
        "category_partition_tokens": {name: dict(value) for name, value in categories.items()},
        "longest_decisions": tails,
        "repetition_candidates": sum(
            row["max_identical_reasoning_paragraph_repeats"] >= 20
            or row["repeated_reasoning_character_fraction"] >= 0.5 for row in records
        ),
        "longest_20_sample_ids": [row["sample_id"] for row in ordered[:20]],
    }


def audit(args):
    from transformers import AutoProcessor

    dataset = ManifestDataset(args.manifest, split=args.split, decode_images=False)
    rows = dataset.rows
    if args.limit:
        rows = rows[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    records = []
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "decisions.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            message = dataset[index]["completion"][0]
            record = {
                **{key: row[key] for key in ("sample_id", "episode", "stage", "prompt_tokens")},
                "target_tokens": row["completion_tokens"],
                "category": category(message),
                **partition_target(processor, row, message),
            }
            stream.write(json.dumps(record) + "\n")
            records.append(record)
            if (index + 1) % 500 == 0:
                stream.flush()
                print(json.dumps({"measured": index + 1, "total": len(rows)}), flush=True)
    report = {
        "schema": "sft-supervision-density/1",
        "at": time.time(),
        "split": args.split,
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        **summarize(records),
        "scope": "Current-completion token offsets; full native target length verified for every row.",
        "limitations": [
            "Short audit prompt used only to render the current target, never as a training input.",
            "Target length equality alone is not proof of complete serving-prefix parity.",
            "Token crossing span boundaries belongs to the first overlapping span.",
            "Repetition flags are review candidates, not automatic rejection decisions.",
            "Length and absence of file changes do not establish whether reasoning is useful.",
        ],
        "training_modified": False,
        "raw_reasoning_exported": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=0)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
