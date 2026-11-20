"""CPU checks for processor parity and real SGLang tool-call round trips."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .data import ManifestDataset, open_store, token_sha256


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def processor_for(model):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this CPU checker with CUDA_VISIBLE_DEVICES=''")
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    return AutoProcessor.from_pretrained(model, local_files_only=True)


def tensor_record(tensor):
    if tensor is None:
        return None
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def prefixes(args):
    processor = processor_for(args.model)
    dataset = ManifestDataset(args.manifest)
    report = {"mode": "cpu_prefixes", "transformers": importlib.metadata.version("transformers"),
              "processor": type(processor).__name__, "model": str(args.model), "samples": []}
    started = time.monotonic()
    for index in range(min(args.limit, len(dataset)) if args.limit else len(dataset)):
        sample = dataset[index]
        kwargs = {"enable_thinking": True, **sample["chat_template_kwargs"]}
        prefix = processor.apply_chat_template(sample["prompt"], tokenize=False,
                                               add_generation_prompt=True, **kwargs)
        inputs = {"text": [prefix], "add_special_tokens": False, "truncation": False,
                  "padding": False, "return_tensors": "pt"}
        if sample["images"]:
            inputs["images"] = sample["images"]
        batch = processor(**inputs)
        record = {"sample_id": sample["sample_id"], "images": len(sample["images"]),
                  "prefix_text_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                  "prefix_token_sha256": token_sha256(batch["input_ids"][0])}
        record.update({key: tensor_record(batch.get(key)) for key in
                       ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")})
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        save(args.output, report)
        print(json.dumps({"prefixes_checked": index + 1, "images": record["images"]}), flush=True)
    return report


def compare(args):
    training, serving = json.loads(args.training.read_text()), json.loads(args.serving.read_text())
    left = {row["sample_id"]: row for row in training["samples"]}
    right = {row["sample_id"]: row for row in serving["samples"]}
    fields = ("prefix_text_sha256", "prefix_token_sha256", "input_ids", "attention_mask",
              "pixel_values", "image_grid_thw")
    mismatches = [{"sample_id": key, "fields": [field for field in fields if left[key][field] != right[key][field]]}
                  for key in left.keys() & right.keys() if any(left[key][field] != right[key][field] for field in fields)]
    missing = sorted(left.keys() ^ right.keys())
    report = {"status": "passed" if not mismatches and not missing and left else "failed",
              "training_transformers": training["transformers"], "serving_transformers": serving["transformers"],
              "compared": len(left.keys() & right.keys()), "missing": missing, "mismatches": mismatches,
              "scope": "same checkpoint, real CPU processor outputs; no server/model inference"}
    save(args.output, report)
    return report


def child_reference(store, reference, key):
    node = json.loads(store.raw(reference))
    return dict(node[1])[key]


def differences(expected, actual, path="$"):
    if type(expected) is not type(actual):
        return [path + ":type"]
    if isinstance(expected, dict):
        missing = [path + "." + key + ":missing" for key in expected.keys() ^ actual.keys()]
        return missing + [item for key in expected.keys() & actual.keys()
                          for item in differences(expected[key], actual[key], path + "." + key)]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [path + ":length"]
        return [item for index, (left, right) in enumerate(zip(expected, actual))
                for item in differences(left, right, f"{path}[{index}]")]
    return [] if expected == actual else [path + ":value"]


def parse_calls(detector, text, schema, chunk_size):
    if not chunk_size:
        return [{"name": call.name, "arguments": json.loads(call.parameters)}
                for call in detector.detect_and_parse(text, schema).calls]
    fragments = {}
    for offset in range(0, len(text), chunk_size):
        result = detector.parse_streaming_increment(text[offset:offset + chunk_size], schema)
        for call in result.calls:
            entry = fragments.setdefault(call.tool_index, {"name": None, "arguments": []})
            if call.name is not None:
                entry["name"] = call.name
            entry["arguments"].append(call.parameters)
    return [{"name": entry["name"], "arguments": json.loads("".join(entry["arguments"]))}
            for _, entry in sorted(fragments.items())]


def tools(args):
    processor = processor_for(args.model)
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

    episodes = {row["episode"]: row for row in map(json.loads, args.episodes.read_text().splitlines())}
    stores, schemas, counts, failures = {}, {}, Counter(), []
    started = time.monotonic()
    for index, row in enumerate(map(json.loads, args.decisions.open()), 1):
        if args.limit and index > args.limit:
            break
        episode = episodes[row["episode"]]
        root = args.data_root / episode["extract_dir"] / "run-01"
        if str(root) not in stores:
            stores[str(root)] = open_store(root)
        store = stores[str(root)]
        choices = store.get(child_reference(store, row["response_ref"], "choices"))
        message = deepcopy(choices[0]["message"])
        calls = message.get("tool_calls") or []
        counts["decisions"] += 1
        if not calls:
            counts["without_tool_calls"] += 1
            continue
        schema_ref = child_reference(store, row["request_ref"], "tools")
        cache_key = (str(root), schema_ref)
        if cache_key not in schemas:
            schema = store.get(schema_ref)
            schema = json.loads(schema) if isinstance(schema, str) else schema
            schemas[cache_key] = (schema, [Tool.model_validate(tool) for tool in schema])
        schema, typed_tools = schemas[cache_key]
        message["content"] = message.get("content") or ""
        for call in calls:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
        messages = [{"role": "user", "content": "Reconstruct the diagram."}]
        kwargs = {"tools": schema, "enable_thinking": True, "tokenize": False}
        prefix = processor.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        full = processor.apply_chat_template(messages + [message], add_generation_prompt=False, **kwargs)
        if not full.startswith(prefix):
            raise ValueError("Official template changed its generation prefix")
        generated = full[len(prefix):].split("</think>", 1)[-1].lstrip("\n")
        expected = [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                    for call in calls]
        actual = parse_calls(Qwen3CoderDetector(), generated, typed_tools, args.stream_chunk_size)
        changed = differences(expected, actual)
        counts["tool_decisions"] += 1
        counts["expected_tool_calls"] += len(expected)
        counts["parsed_tool_calls"] += len(actual)
        counts["mismatched_decisions"] += bool(changed)
        if changed and len(failures) < 32:
            failures.append({"episode": row["episode"], "call_id": row["call_id"], "differences": changed[:32],
                             "tools": [call["name"] for call in expected]})
        if counts["tool_decisions"] % 1000 == 0:
            print(json.dumps(dict(counts)), flush=True)
    report = {"status": "passed" if not counts["mismatched_decisions"] else "failed", "counts": dict(counts),
              "failures": failures, "elapsed_seconds": time.monotonic() - started,
              "transformers": importlib.metadata.version("transformers"),
              "sglang": importlib.metadata.version("sglang"),
              "schema_variants": len({reference for _, reference in schemas}),
              "schema_cache_entries": len(schemas), "reasoning_removed_before_parsing": True,
              "stream_chunk_characters": args.stream_chunk_size,
              "scope": "real teacher outputs -> official9B template -> installed SGLang parser"}
    save(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prefixes", "compare", "tools"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--serving", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--stream-chunk-size", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = globals()[args.mode](args)
    print(json.dumps(report if args.mode != "prefixes" else
                     {"samples": len(report["samples"]), "transformers": report["transformers"]}), flush=True)


if __name__ == "__main__":
    main()
