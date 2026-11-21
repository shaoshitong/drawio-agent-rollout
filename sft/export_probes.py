"""Export the frozen validation probes for an isolated numerical checkpoint audit."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .train_unsloth import Examples, render_completion
from .data import open_store


def export(manifest, plan, model, output, external_images=False):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    examples = Examples(manifest, "validation", True)
    selected = json.loads(plan.read_text())["probes"]["validation"]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=False)
    records, assets, stores = [], {}, {}
    for probe in selected:
        example = examples({"row_index": probe["index"]})
        row = examples.dataset.rows[probe["index"]]
        assert example["sample_id"] == probe["sample_id"]
        prompt, completion = render_completion(processor, example)
        urls = []
        if external_images:
            root = row["snapshot_root"]
            if root not in stores:
                stores[root] = open_store(root)
            request = stores[root].get(row["request_ref"])
            urls = [part["image_url"]["url"] for message in request["messages"]
                    if isinstance(message.get("content"), list) for part in message["content"]
                    if part.get("type") == "image_url"]
            if len(urls) != len(example["images"]):
                raise ValueError("Original asset order differs from the native training images")
        paths = []
        for position, image in enumerate(example["images"]):
            image = image.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if external_images:
                raw = base64.b64decode(urls[position].split(",", 1)[1])
                raw_digest = hashlib.sha256(raw).hexdigest()
                path = images / (raw_digest + ".bin")
                assets[raw_digest] = {"bytes": len(raw), "rgb_size": list(image.size), "rgb_sha256": digest}
            else:
                path = images / (digest + ".png")
            if not external_images and not path.exists():
                image.save(path, compress_level=1)
            paths.append(str(path.relative_to(output)))
        records.append({"sample_id": example["sample_id"], "prompt": prompt,
                        "completion": completion, "images": paths,
                        "sequence_tokens": row["token_count"], "target_tokens": row["completion_tokens"],
                        "prompt_token_sha256": row["prompt_token_sha256"]})
    (output / "probes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    shutil.copy2(plan, output / "fixed-plan.json")
    if external_images:
        (output / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    report = {"records": len(records), "images": len(assets) if external_images else len(list(images.iterdir())),
              "target_tokens": sum(row["target_tokens"] for row in records),
              "external_original_assets": external_images,
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "scope": "teacher validation contexts/labels for host-only numerical audit; not mounted into DSH",
              "gold_xml_dataset_included": False}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "plan", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--external-images", action="store_true",
                        help="Export original image hashes so the dataset host can supply assets directly")
    args = parser.parse_args()
    report = export(args.manifest, args.plan, args.model, args.output, args.external_images)
    if args.archive:
        with tarfile.open(args.archive, "w:gz") as archive:
            archive.add(args.output, arcname="probes")
        report.update(archive_bytes=args.archive.stat().st_size,
                      archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
