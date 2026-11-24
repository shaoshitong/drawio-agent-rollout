"""Export a target-free native replay of one complete captured OpenAI request."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from .check_serving import processor_for
from .data import token_sha256
from .export_sft import normalize


def export(args):
    body = args.request.read_bytes()
    request = json.loads(body)
    messages, images = normalize(request["messages"])
    for message in messages:
        if message.get("content") is None:
            message["content"] = ""
    processor = processor_for(args.model)
    prompt = processor.apply_chat_template(messages, tools=request.get("tools", []),
        tokenize=False, add_generation_prompt=True, **request.get("chat_template_kwargs", {}))
    batch = processor(text=[prompt], images=[image.convert("RGB") for image in images],
        add_special_tokens=False, truncation=False, return_tensors="pt")
    directory = args.output / "images"
    directory.mkdir(parents=True, exist_ok=False)
    paths, assets = [], {}
    for message in request["messages"]:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if part.get("type") != "image_url":
                continue
            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            digest = sha256(raw).hexdigest()
            path = directory / (digest + ".bin")
            path.write_bytes(raw)
            paths.append(str(path.relative_to(args.output)))
            assets[digest] = {"bytes": len(raw)}
    row = {"sample_id": args.sample_id, "prompt": prompt, "images": paths,
           "prompt_token_sha256": token_sha256(batch["input_ids"][0]),
           "prompt_tokens": batch["input_ids"].shape[-1]}
    (args.output / "probes.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"request_sha256": sha256(body).hexdigest(), "images": len(images),
        "original_assets": assets, "prompt_tokens": row["prompt_tokens"],
        "target_included": False, "gold_included": False,
        "scope": "Canonical HF replay of the captured request; not a certification of SG/HF tokenization parity."}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
