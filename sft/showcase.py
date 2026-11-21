"""Export real masked SFT decisions and build an offline inspection page."""

import argparse
import base64
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from .audit_actions import category
from .data import ManifestDataset, open_store, token_sha256
from .train_unsloth import render_completion


def export_examples(args):
    from .check_serving import processor_for

    dataset = ManifestDataset(args.manifest)
    processor = processor_for(args.model)
    stores = {}

    def message(row):
        root = row["snapshot_root"]
        if root not in stores:
            stores[root] = open_store(root)
        return stores[root].get(row["response_ref"])["choices"][0]["message"]

    training = [row for row in dataset.rows if row["split"] == "train"]
    chosen = {"initial": training[0]}
    candidates = sorted((row for row in training if row["image_count"] <= 6 and row["token_count"] <= 65536),
                        key=lambda row: (row["token_count"], row["sample_id"]))
    for row in candidates:
        target = message(row)
        kind = category(target)
        if row["image_count"] and kind in ("image_read", "crop_related"):
            chosen.setdefault(kind, row)
        if kind == "xml_related" and re.search(r"<(?:mxfile|mxGraphModel|mxCell)\b", json.dumps(target)):
            chosen.setdefault("xml_construct", row)
        if row["stage"] == 2 and row["image_count"]:
            chosen.setdefault("after_reviewer", row)
        if len(chosen) == 5:
            break
    assets, samples = {}, []
    for kind, row in chosen.items():
        example = dataset[dataset.rows.index(row)]
        example["chat_template_kwargs"]["enable_thinking"] = True
        prompt, completion = render_completion(processor, example)
        if re.search(r"(?:hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{24,})", prompt + completion):
            raise ValueError("Credential-like text in showcase sample " + row["sample_id"])
        inputs = {"images": example["images"]} if example["images"] else {}
        prefix_ids = processor(text=[prompt], add_special_tokens=False, return_tensors="pt", **inputs)["input_ids"][0]
        target_ids = processor(text=[completion], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if (token_sha256(prefix_ids) != row["prompt_token_sha256"]
                or len(target_ids) != row["completion_tokens"]):
            raise ValueError("Displayed sample differs from the immutable training manifest")
        images = []
        for image in example["images"]:
            rgb = image.convert("RGB")
            digest = sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest not in assets:
                stream = BytesIO()
                rgb.save(stream, format="PNG")
                assets[digest] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
            images.append(digest)
        boundary = [{"position": len(prefix_ids) + offset, "id": int(token),
                     "token": processor.tokenizer.convert_ids_to_tokens(int(token)),
                     "label": -100 if offset < 0 else int(token)}
                    for offset, token in enumerate(prefix_ids[-8:].tolist() + target_ids[:16].tolist(), -8)]
        samples.append({"kind": kind, "row": row, "category": category(example["completion"][0]),
                        "prompt": example["prompt"], "completion": example["completion"],
                        "canonical_prompt": prompt, "canonical_completion": completion,
                        "images": images, "boundary": boundary, "manifest_prefix_hash_verified": True})
    ids = set(json.loads(args.probe_ids.read_text()))
    labels = {row["sample_id"]: category(message(row)) for row in dataset.rows if row["sample_id"] in ids}
    if set(labels) != ids:
        raise ValueError("Fixed validation IDs are missing from the source manifest")
    report = {"manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
              "decisions": dict(Counter(row["split"] for row in dataset.rows)),
              "episodes": {split: len({row["episode"] for row in dataset.rows if row["split"] == split})
                           for split in ("train", "validation")},
              "complexity_source": "Teacher final editable XML vertices + edges; no gold XML in training input",
              "samples": samples, "assets": assets, "loss_categories": labels}
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(samples), "image_assets": len(assets),
                      "bytes": args.output.stat().st_size, "sample_ids": [s["row"]["sample_id"] for s in samples]}))


def build_report(args):
    report = json.loads(args.examples.read_text())
    variants = {name: json.loads((args.loss / f"loss-{name}.json").read_text()) for name in ("base", "peft", "merged")}
    grouped = defaultdict(list)
    values = {name: {row["sample_id"]: row for row in result["per_probe"]} for name, result in variants.items()}
    for sample_id, row in values["base"].items():
        grouped[report["loss_categories"][sample_id]].append(sample_id)
    metrics = []
    for kind in ("image_read", "crop_related", "xml_related", "other_tool", "assistant_only"):
        ids = grouped[kind]
        tokens = sum(values["base"][i]["tokens"] for i in ids)
        ce = {name: sum(values[name][i]["ce"] * values[name][i]["tokens"] for i in ids) / tokens
              for name in variants} if tokens else {}
        metrics.append({"category": kind, "decisions": len(ids), "tokens": tokens, "ce": ce,
                        "drop": ce["base"] - ce["peft"] if tokens else None,
                        "relative_drop": 1 - ce["peft"] / ce["base"] if tokens else None})
    report["loss"] = {"groups": metrics, "overall": {name: {k: v for k, v in value.items() if k != "per_probe"}
                                                     for name, value in variants.items()}}
    report["training"] = json.loads(args.training.read_text())
    report["renders"] = json.loads((args.media / "renders.json").read_text())
    report["media"] = {name: "data:image/png;base64," + base64.b64encode((args.media / f"{name}.png").read_bytes()).decode()
                       for name in ("reference", "capture", "replay", "base")}
    report["xml"] = {name: (args.media / f"{name}.drawio").read_text() for name in ("capture", "replay", "base")}
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    args.output.write_text(template.replace("__REPORT_DATA__", payload))
    print(json.dumps({"html": str(args.output), "bytes": args.output.stat().st_size,
                      "loss_by_action": metrics}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "report"))
    for name in ("manifest", "model", "probe-ids", "examples", "loss", "media", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    args = parser.parse_args()
    {"export": export_examples, "report": build_report}[args.mode](args)


if __name__ == "__main__":
    main()
