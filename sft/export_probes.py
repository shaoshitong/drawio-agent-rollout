"""Export the frozen validation probes for an isolated numerical checkpoint audit."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .train_unsloth import Examples, render_completion


def export(manifest, plan, model, output):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    examples = Examples(manifest, "validation", True)
    selected = json.loads(plan.read_text())["probes"]["validation"]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=False)
    records = []
    for probe in selected:
        example = examples({"row_index": probe["index"]})
        row = examples.dataset.rows[probe["index"]]
        assert example["sample_id"] == probe["sample_id"]
        prompt, completion = render_completion(processor, example)
        paths = []
        for image in example["images"]:
            image = image.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            path = images / (digest + ".png")
            if not path.exists():
                image.save(path, compress_level=1)
            paths.append(str(path.relative_to(output)))
        records.append({"sample_id": example["sample_id"], "prompt": prompt,
                        "completion": completion, "images": paths,
                        "sequence_tokens": row["token_count"], "target_tokens": row["completion_tokens"],
                        "prompt_token_sha256": row["prompt_token_sha256"]})
    (output / "probes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    shutil.copy2(plan, output / "fixed-plan.json")
    report = {"records": len(records), "images": len(list(images.iterdir())),
              "target_tokens": sum(row["target_tokens"] for row in records),
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
    args = parser.parse_args()
    report = export(args.manifest, args.plan, args.model, args.output)
    if args.archive:
        with tarfile.open(args.archive, "w:gz") as archive:
            archive.add(args.output, arcname="probes")
        report.update(archive_bytes=args.archive.stat().st_size,
                      archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
