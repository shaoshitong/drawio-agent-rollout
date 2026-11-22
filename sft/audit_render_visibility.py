"""Check exact render visibility without treating pixel mismatch as quality failure."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

from .check_prefix_chains import Reader


def measure(reader, row, png):
    from PIL import Image

    with Image.open(BytesIO(reader.store.raw(png["sha256"]))) as target:
        size = target.size
        target_hash = sha256(target.convert("RGBA").tobytes()).hexdigest()
    images = reader.context(row)["images"]
    sizes, matches, same_size = Counter(), [], 0
    for index, (_, _, reference) in enumerate(images):
        with Image.open(BytesIO(reader.store.raw(reference))) as image:
            sizes[f"{image.width}x{image.height}"] += 1
            if image.size != size:
                continue
            same_size += 1
            if sha256(image.convert("RGBA").tobytes()).hexdigest() == target_hash:
                matches.append(index)
    return {"sample_id": row["sample_id"], "render_size": list(size), "input_images": len(images),
            "input_sizes": dict(sizes), "same_size_images": same_size,
            "pixel_identical_input_indices": matches}


def audit(args):
    report = json.loads(args.report.read_text())
    rows = {row["sample_id"]: row for row in json.loads(args.manifest.read_text())}
    readers, records = {}, []
    for record in report["records"]:
        png = next(file for file in record["snapshot_state"]["files"] if file["name"].endswith(".png"))
        if png["raw_render_blob_in_input_images"]:
            continue
        row = rows[record["sample_id"]]
        root = row["snapshot_root"]
        if root not in readers:
            readers[root] = Reader(root)
        records.append(measure(readers[root], row, png))
    result = {"at": time.time(), "records": records,
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "report_sha256": sha256(args.report.read_bytes()).hexdigest(),
              "method": "Exact decoded RGBA pixels; no resizing, rendering or model calls.",
              "limitation": "No exact match does not prove unseen: resized or cropped views are not matched.",
              "training_modified": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
