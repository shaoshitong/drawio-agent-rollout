"""Measure context-clearing candidates without changing training or exporting text."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

from .check_prefix_chains import Reader


def template_messages(messages):
    messages = deepcopy(messages)
    for message in messages:
        message["content"] = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
        content = message["content"]
        if isinstance(content, list):
            message["content"] = [
                {"type": "image"} if part.get("type") == "image_url" else part for part in content
            ]
    return messages


def tool_result(message):
    content = message.get("content")
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content or []
    )
    return message["role"] == "tool" or (
        message["role"] == "user" and text.startswith("<tool_response>")
        and text.endswith("</tool_response>")
    )


def clear_context(messages, reasoning=False, keep_results=None):
    """Counterfactual only: preserve calls, images, roles and result containers."""
    messages = deepcopy(messages)
    results = [index for index, message in enumerate(messages) if tool_result(message)]
    old = set(results[:-keep_results] if keep_results else results) if keep_results is not None else set()
    for index, message in enumerate(messages):
        content = message.get("content") or ""
        if reasoning and message["role"] == "assistant":
            message.pop("reasoning_content", None)
            if isinstance(content, str) and content.lstrip().startswith("<think>") and "</think>" in content:
                message["content"] = content.split("</think>", 1)[1].lstrip("\n")
        if index not in old:
            continue
        if isinstance(content, str):
            message["content"] = clear_text(content)
        else:
            message["content"] = [
                {**part, "text": clear_text(part["text"])} if part.get("type") == "text" else part
                for part in content
            ]
    return messages


def clear_text(text):
    placeholder = "[Earlier tool output omitted; re-read the file if needed.]"
    prefix = "<tool_response>\n" if text.startswith("<tool_response>") else ""
    suffix = "\n</tool_response>" if text.endswith("</tool_response>") else ""
    return prefix + placeholder + suffix


def render(processor, messages, completion, tools, kwargs):
    options = {**kwargs, "tools": tools, "tokenize": False}
    prefix = processor.apply_chat_template(messages, add_generation_prompt=True, **options)
    full = processor.apply_chat_template(messages + completion, add_generation_prompt=False, **options)
    if not full.startswith(prefix):
        raise ValueError("Current target changes its prompt prefix")
    target = full[len(prefix):]
    count = len(processor.tokenizer(prefix, add_special_tokens=False)["input_ids"])
    target_count = len(processor.tokenizer(target, add_special_tokens=False)["input_ids"])
    return count + target_count, sha256(target.encode()).hexdigest(), target_count


def verify_images(processor, reader, row, expected, keep):
    from .data import token_sha256
    from .export_sft import normalize

    request, response = reader.store.get(row["request_ref"]), reader.store.get(row["response_ref"])
    prompt, images = normalize(request["messages"], decode_images=False)
    completion, _ = normalize([response["choices"][0]["message"]], decode_images=False)
    prompt = clear_context(prompt, keep_results=keep)
    kwargs = {**row.get("chat_template_kwargs", {"enable_thinking": True}), "tools": request.get("tools", [])}
    count, target_hash, target_count = render(processor, prompt, completion, request.get("tools", []), kwargs)
    if target_hash != expected["target_sha256"]:
        raise ValueError("Native image materialization changed the target")
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = processor(text=[text], images=images, truncation=False, padding=False,
                        add_special_tokens=False, return_tensors="pt")
    total = encoded["input_ids"].shape[-1] + target_count
    prediction = expected["variants"][f"clear_tool_text_keep_{keep}"]
    if count != prediction["text_tokens"]:
        raise ValueError("Native image materialization changed text tokenization")
    estimate = prediction["estimated_total_with_fixed_image_expansion"]
    if estimate is not None and total != estimate:
        raise ValueError("Fixed-image token estimate differs from native processing")
    result = {"token_count": total, "image_count": len(images), "target_sha256": target_hash,
              "prompt_token_sha256": token_sha256(encoded["input_ids"][0]),
              "image_grid_thw": encoded["image_grid_thw"].tolist(), "keep_results": keep}
    for image in images:
        image.close()
    return result


def audit(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    readers, records = {}, []
    rows = json.loads(args.manifest.read_text())
    variants = [("clear_history_reasoning", True, None)] + [
        (f"{kind}_keep_{keep}", reasoning, keep)
        for keep in args.keep_results
        for kind, reasoning in (("clear_tool_text", False), ("clear_both", True))
    ]
    for row in rows:
        root = row["snapshot_root"]
        if root not in readers:
            readers[root] = Reader(root)
        reader = readers[root]
        request, response = reader.value(row["request_ref"]), reader.value(row["response_ref"])
        prompt = template_messages(request["messages"])
        completion = template_messages([response["choices"][0]["message"]])
        tools = request.get("tools", [])
        kwargs = row.get("chat_template_kwargs", {"enable_thinking": True})
        original, target_hash, target_count = render(processor, prompt, completion, tools, kwargs)
        if target_count != row["completion_tokens"]:
            raise ValueError("Original target length differs: " + row["sample_id"])
        record = {"sample_id": row["sample_id"], "original_text_tokens": original,
                  "original_total_tokens": row.get("token_count"), "target_tokens": target_count,
                  "target_sha256": target_hash, "variants": {}}
        for name, reasoning, keep in variants:
            candidate = clear_context(prompt, reasoning, keep)
            count, candidate_hash, _ = render(processor, candidate, completion, tools, kwargs)
            if candidate_hash != target_hash:
                raise ValueError("Context clearing changed current target")
            total = row["token_count"] + count - original if "token_count" in row else None
            record["variants"][name] = {
                "text_tokens": count, "text_tokens_removed": original - count,
                "estimated_total_with_fixed_image_expansion": total,
                "estimated_fits_context": total <= args.context if total is not None else None,
                "text_only_fits_context": count <= args.context,
            }
        if args.verify_keep_results is not None:
            record["native_image_verification"] = verify_images(
                processor, reader, row, record, args.verify_keep_results)
        records.append(record)
        print(json.dumps({"measured": len(records), "total": len(rows)}), flush=True)
    report = {"at": time.time(), "records": records, "context_limit": args.context,
              "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "known_total_fits": dict(Counter(name for row in records for name, value in
                  row["variants"].items() if value["estimated_fits_context"])),
              "limitations": ["Token-budget counterfactuals, not recovered training examples.",
                  "Image expansion held fixed; pixels and grids not reprocessed.",
                  "Targets may reference cleared history; semantic validity is not established.",
                  "A context policy must also be implemented and verified in inference."],
              "training_modified": False, "raw_text_exported": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "known_total_fits": report["known_total_fits"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--keep-results", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--verify-keep-results", type=int, choices=[4, 8, 16])
    parser.add_argument("--context", type=int, default=262144)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
