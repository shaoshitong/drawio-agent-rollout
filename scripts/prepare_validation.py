"""Package the frozen validation inputs; run all actual validation on H20."""

import argparse
from contextlib import ExitStack
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import tarfile
import time


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compare-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume-files",
        type=int,
        default=0,
        help="Continue after a completed-member checkpoint; extract with ignore_zeros=True",
    )
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.out.resolve()
    config = json.loads(args.config.read_text())
    output.mkdir(parents=True, exist_ok=True)
    items, expected = {}, {}

    def add(path, name):
        items[name] = Path(path)

    def manifest(name, rows):
        path = output / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        add(path, name)

    rows = read_rows(root / config["corpus"])
    portable = []
    for row in rows:
        name = "corpus/" + row["id"] + ".drawio"
        add(root / row["code_file"], name)
        expected[name] = row["code_sha256"]
        portable.append({**row, "code_file": name})
    manifest("corpus.jsonl", portable)
    final = root / config["release"]
    variants = []
    for path in sorted((final / "variants/cases").glob("*/status.json")):
        row = json.loads(path.read_text())
        name, base = "variants/" + row["id"] + ".drawio", "bases/" + row["id"] + ".drawio"
        add(row["source"], name)
        add(row["base"], base)
        expected[name] = row["input_sha256"]
        variants.append(
            {"id": row["id"], "code_file": name, "code_sha256": row["input_sha256"], "base": base}
        )
    manifest("variants.jsonl", variants)
    controls = []
    for row in read_rows(final / "controls-input/controls.jsonl"):
        name, base = "controls/" + Path(row["xml"]).name, "bases/" + row["case"] + ".drawio"
        add(row["xml"], name)
        if base not in items:
            add(final / "corpus/cases" / row["case"] / "canonical.drawio", base)
        controls.append({**row, "code_file": name, "xml": name, "base": base})
    manifest("controls.jsonl", controls)
    validation = root / config["validation"]
    for name in ("batch.py", "controls.py", "test_compiler.py"):
        add(validation / "conversion/drawio_semantic_compiler" / name, "validation/" + name)
    add(validation / "boundary_equivalence.py", "validation/boundary_equivalence.py")
    for name in ("acceptance.py", "generalization.py"):
        add(args.compare_source / name, "validation/" + name)
    generalization = root / config["generalization"]
    for folder in ("sources_100", "variants_100"):
        for path in sorted((generalization / folder).rglob("*")):
            if path.is_file() and path.suffix in (".drawio", ".xml", ".json", ".jsonl"):
                add(path, "generalization/" + folder + "/" + str(path.relative_to(generalization / folder)))
    for path in sorted((root / config["acceptance"]).glob("*/input-*.drawio")):
        add(path, "acceptance/" + path.parent.name + "/" + path.name)
    for name, path in items.items():
        if name not in expected:
            with path.open("rb") as stream:
                expected[name] = file_digest(stream, "sha256").hexdigest()
    path = output / "manifest.json"
    path.write_text(json.dumps(expected, indent=2) + "\n")
    add(path, "manifest.json")
    archive = output / "compiler-validation.tar.gz"
    with ExitStack() as stack:
        if args.resume_files:
            stream = stack.enter_context(gzip.open(archive, "ab", compresslevel=1))
            tar = stack.enter_context(tarfile.open(fileobj=stream, mode="w", dereference=True))
        else:
            tar = stack.enter_context(tarfile.open(archive, "w:gz", compresslevel=1, dereference=True))
        for done, (name, path) in enumerate(items.items(), 1):
            if done <= args.resume_files:
                continue
            tar.add(path, arcname=name, recursive=False)
            if done % 128 == 0:
                progress = {
                    "files": done,
                    "total": len(items),
                    "archive_bytes": archive.stat().st_size,
                    "at": time.time(),
                }
                (output / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
    with archive.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    receipt = {
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "files": len(items),
        "corpus": len(rows),
        "variants": len(variants),
        "controls": len(controls),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
