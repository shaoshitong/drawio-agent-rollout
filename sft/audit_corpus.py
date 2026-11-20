"""Measure XML resources before selecting image-to-code SFT targets."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re

from drawio.utils.xml_models import models


def resources(pages):
    counts, fonts = Counter(), set()
    elements = (element for page in pages for element in page.iter())
    for element in elements:
        for value in element.attrib.values():
            counts["all_attribute_characters"] += len(value)
            font = "data:font/" in value
            opaque = font or "data:image/" in value or "stencil(" in value
            counts["font_attributes"] += font
            if opaque:
                counts["opaque_attribute_characters"] += len(value)
            for payload in re.findall(r"data:font/[^,;]+;base64,([A-Za-z0-9+/=]+)", value):
                fonts.add(sha256(payload.encode()).hexdigest())
    return {**counts, "font_payload_base64_sha256": sorted(fonts)}


def inspect(item):
    root, row = item
    blob = (Path(root) / row["code_file"]).read_bytes()
    _, pages = models(blob)
    counts = Counter()
    cells = (cell for page in pages for cell in page.iter("mxCell"))
    for cell in cells:
        style = cell.get("style", "")
        image = "shape=image" in style or "image=" in style
        counts["cells"] += 1
        counts["vertices"] += cell.get("vertex") == "1"
        counts["edges"] += cell.get("edge") == "1"
        counts["image_cells"] += image
        counts["vector_vertices"] += cell.get("vertex") == "1" and not image
        counts["data_image_cells"] += any("data:image/" in value for value in cell.attrib.values())
        for value in cell.attrib.values():
            counts["attribute_characters"] += len(value)
            if "data:image/" in value:
                counts["image_data_attribute_characters"] += len(value)
    return {
        "id": row["id"],
        "source": row.get("source"),
        "xml_source_bytes": len(blob),
        "pages": len(pages),
        **counts,
        **resources(pages),
    }


def percentiles(values):
    values = sorted(values)
    return {str(p): values[round((len(values) - 1) * p / 100)] for p in (10, 25, 50, 75, 90, 95, 99, 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(inspect, ((str(args.root), row) for row in rows)):
            records.append(record)
            if len(records) % 100 == 0:
                print(json.dumps({"inspected": len(records), "total": len(rows)}), flush=True)
    (args.output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    fractions = [
        row.get("image_data_attribute_characters", 0) / max(1, row["attribute_characters"]) for row in records
    ]
    result = {
        "cases": len(records),
        "source_bytes": sum(row["xml_source_bytes"] for row in records),
        "source_byte_percentiles": percentiles([row["xml_source_bytes"] for row in records]),
        "data_image_cases": sum(row["data_image_cells"] > 0 for row in records),
        "cases_without_image_cells": sum(row["image_cells"] == 0 for row in records),
        "image_attribute_fraction_percentiles": percentiles(fractions),
        "cases_image_attributes_over_90_percent": sum(value > 0.9 for value in fractions),
        "total_image_cells": sum(row["image_cells"] for row in records),
        "total_vector_vertices": sum(row["vector_vertices"] for row in records),
        "image_resource_attribute_characters": sum(
            row.get("image_data_attribute_characters", 0) for row in records
        ),
        "scope": "Source bytes and decoded mxCell attributes; not token lengths, raster coverage or fidelity.",
        "inputs_modified": False,
        "training_run": False,
    }
    result["font_cases"] = sum(row.get("font_attributes", 0) > 0 for row in records)
    result["unique_font_payload_base64_digests"] = len(
        {value for row in records for value in row["font_payload_base64_sha256"]}
    )
    result["cases_without_images_or_font_attributes"] = sum(
        row["image_cells"] == 0 and row.get("font_attributes", 0) == 0 for row in records
    )
    result["opaque_attribute_fraction_percentiles"] = percentiles(
        [
            row.get("opaque_attribute_characters", 0) / max(1, row.get("all_attribute_characters", 0))
            for row in records
        ]
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
