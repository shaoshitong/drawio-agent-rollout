"""Pack accepted images into IR-only or two-stage SFT examples."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from drawio.compiler.syntax import format_style
from drawio.compiler import ir as graph_ir
from drawio.compiler.compiler import compile_bytes
from drawio.compiler.emit import emit
from meanflow_batch import load_rows
from meanflow_two_stage import IR_LINES_PROMPT, XML_PROMPT, compile_visual_ir, require_editable


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ir_lines(document, canvas):
    page, = document["pages"]
    settings = dict(page.get("settings", []))
    if [int(settings[key]) for key in ("pageWidth", "pageHeight")] != canvas:
        raise ValueError("Input image and IR canvas mismatch")
    yield {"canvas": canvas}
    for cell in page["cells"]:
        if cell["kind"] not in {"vertex", "edge"}:
            continue
        vertex = cell["kind"] == "vertex"
        row = {"id": cell["id"], "k": "v" if vertex else "e"}
        if cell.get("parent") not in {None, "1"}:
            row["p"] = cell["parent"]
        if label := cell.get("label"):
            row["l"] = label
        if style := cell.get("style"):
            row["s"] = format_style(style)
        geom = cell.get("geometry", {})
        if vertex:
            row["g"] = [geom.get(key, 0) for key in ("x", "y", "width", "height")]
        else:
            for name, key in (("f", "source"), ("t", "target")):
                if cell.get(key) is not None:
                    row[name] = cell[key]
            for name, key in (("sp", "source_point"), ("tp", "target_point"), ("w", "points")):
                if geom.get(key) is not None:
                    row[name] = geom[key]
        yield row
    yield {"end": True}


def use_file(src, target):
    if target.is_file():
        return
    try:
        target.hardlink_to(src)
    except OSError:
        copyfile(src, target)


def pack(records, target, expected, ir_only=False):
    if len(records) != expected or len({row["sha256"] for row in records}) != expected:
        raise ValueError(f"Expected {expected} unique accepted PNGs")
    target.mkdir(parents=True, exist_ok=True)
    episodes = target / "episodes.jsonl"
    examples = target / "sft.jsonl"
    counts = Counter()
    with episodes.with_suffix(".tmp").open("w") as episode_out, examples.with_suffix(".tmp").open("w") as example_out:
        for record in records:
            ident = record["case_id"]
            name = Path("episodes") / ident
            folder = target / name
            folder.mkdir(parents=True, exist_ok=True)
            source = Path(record["image"])
            with source.open("rb") as stream:
                signature = stream.read(24)
            if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
                raise ValueError(f"Training reference is not a PNG for {ident}")
            canvas = [int.from_bytes(signature[i:i+4], "big") for i in (16, 20)]
            ir = json.loads(Path(record["ir"]).read_text())
            if ir_only and record.get("ir_source") != "canonical_ir":
                ir = require_editable(compile_visual_ir(ir, graph_ir, emit, compile_bytes)).ir_json
            first = "\n".join(map(dump, ir_lines(ir, canvas)))
            use_file(source, folder / "image.png")
            if ir_only:
                temporary_ir = folder / ".ir.json"
                temporary_ir.write_text(dump(ir) + "\n")
                temporary_ir.replace(folder / "ir.json")
            else:
                xml = Path(record["xml"]).read_text()
                use_file(Path(record["ir"]), folder / "ir.json")
                use_file(Path(record["xml"]), folder / "diagram.drawio")
            (folder / "ir.jsonl").write_text(first + "\n")
            paths = {key: str(name / basename) for key, basename in
                     (("image", "image.png"), ("ir", "ir.json"), ("ir_lines", "ir.jsonl"))}
            if not ir_only:
                paths["xml"] = str(name / "diagram.drawio")
            metadata = {key: record[key] for key in
                        ("case_id", "sha256", "source", "xml_source", "arxiv_url", "paper_id", "license",
                         "license_urls", "source_license_status", "dataset", "dataset_revision", "dataset_split",
                         "source_record", "source_row", "source_file", "source_url", "discovery_dataset", "repository",
                         "revision", "path", "quality_tier", "complexity", "structure_hint", "original_source_path",
                         "provenance_path") if record.get(key) is not None}
            if "paper_id" not in metadata and record.get("paper"):
                metadata["paper_id"] = record["paper"]
            episode_out.write(dump({**metadata, "size": canvas, **paths}) + "\n")
            prompt_ir = IR_LINES_PROMPT + f"\nThe input canvas is exactly {canvas[0]} by {canvas[1]} pixels."
            report_path = Path(record["report"])
            report = json.loads(report_path.read_text())
            if record.get("xml_source") in {"author_drawio", "template_drawio", "restored_drawio"}:
                ir_source = "native_xml_projection"
            elif ir_only:
                ir_source = "canonical_ir"
            elif (report_path.parent / "stage1.txt").is_file():
                ir_source = "normalized_model_ir"
            else:
                ir_source = "saved_ir"
            stages = [("image_to_ir", prompt_ir, first, "ir_request", ir_source)]
            if not ir_only:
                prompt_xml = XML_PROMPT + "\n\nIR\n" + dump(ir)
                stages.append(("image_ir_to_xml", prompt_xml, xml, "xml_request", record["xml_source"]))
            for stage, prompt, completion, request, origin in stages:
                example = {
                    "episode_id": ident,
                    "stage": stage,
                    "source": record["source"],
                    "target_source": origin,
                    "images": [paths["image"]],
                    "prompt": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}],
                    "completion": [{"role": "assistant", "content": [{"type": "text", "text": completion}]}],
                }
                example.update({key: report[request][key] for key in ("provider", "model") if key in report.get(request, {})})
                details = report.get(request, {})
                if ir_only and details.get("reasoning_required"):
                    reasoning = (report_path.parent / "reasoning.txt").read_bytes()
                    if (details.get("provider") not in {"hy", "tcodex"} or not reasoning.strip()
                            or sha256(reasoning).hexdigest() != details.get("reasoning_sha256")):
                        raise ValueError(f"Invalid teacher reasoning for {ident}")
                    example["completion"][0]["reasoning_content"] = reasoning.decode("utf-8")
                if "paper_id" in metadata:
                    example["paper_id"] = metadata["paper_id"]
                example_out.write(dump(example) + "\n")
            counts[record["source"]] += 1
    episodes.with_suffix(".tmp").replace(episodes)
    examples.with_suffix(".tmp").replace(examples)
    receipt = {"images": len(records), "sft_rows": len(records) if ir_only else 2 * len(records), "sources": dict(counts),
               "episodes": episodes.name, "training": examples.name}
    temp = target / ".dataset.json"
    temp.write_text(dump(receipt) + "\n")
    temp.replace(target / "dataset.json")
    print(dump(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--ir-only", action="store_true", help="Pack one canonical image-to-IR row per image")
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    pack([row for path in args.manifests for row in load_rows(path)], args.out, args.expected, args.ir_only)
