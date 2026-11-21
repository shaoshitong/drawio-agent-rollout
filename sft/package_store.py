"""Package only the archive objects reachable from an unchanged SFT manifest."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
import time

from .data import load_manifest, open_store


def reachable_objects(stores, roots):
    pending = [(reference, True) for reference in roots]
    seen = set()
    while pending:
        reference, structured = pending.pop()
        if reference in seen:
            continue
        relative = Path(reference[:2]) / reference[2:]
        store = next((store for store in stores if (store.objects / relative.with_suffix(".bin")).is_file()
                      or (store.objects / relative.with_suffix(".gz")).is_file()), None)
        if store is None:
            raise FileNotFoundError("Referenced object is absent from the source archives: " + reference)
        raw = store.raw(reference)
        seen.add(reference)
        path = store.objects / reference[:2] / (reference[2:] + ".bin")
        if not path.is_file():
            path = path.with_suffix(".gz")
        if structured:
            node = json.loads(raw)
            if node[0] == "dict":
                pending.extend((child, True) for _, child in node[1])
            elif node[0] == "list":
                pending.extend((child, True) for child in node[1])
            elif node[0] == "image":
                pending.append((node[2], False))
        yield reference, path, len(raw)


def package(manifest, sources, destination):
    rows = load_manifest(manifest)
    roots = {row[key] for row in rows for key in ("request_ref", "response_ref")}
    stores = [open_store(source) for source in sources]
    count = raw_bytes = stored_bytes = 0
    with destination.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as archive:
        archive.add(manifest, arcname="curriculum.jsonl", recursive=False)
        for reference, path, size in reachable_objects(stores, sorted(roots)):
            relative = Path(reference[:2]) / path.name
            archive.add(path, arcname="archive/objects/" + str(relative), recursive=False)
            count += 1
            raw_bytes += size
            stored_bytes += path.stat().st_size
            if count % 10000 == 0:
                print(json.dumps({"objects_verified": count, "stored_bytes": stored_bytes,
                                  "archive_bytes": stream.tell(), "at": time.time()}), flush=True)
        report = {"at": time.time(), "decisions": len(rows), "splits": dict(Counter(row["split"] for row in rows)),
                  "root_references": len(roots), "objects": count, "raw_object_bytes": raw_bytes,
                  "stored_object_bytes": stored_bytes, "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                  "sources": [str(source) for source in sources], "gold_xml_dataset_included": False,
                  "scope": "Unchanged accepted decision index and reachable request/response/image objects; no ledger, raw SSE or filesystem snapshots"}
        payload = (json.dumps(report, indent=2) + "\n").encode()
        header = tarfile.TarInfo("manifest.json")
        header.size = len(payload)
        archive.addfile(header, BytesIO(payload))
    digest = sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    report.update(archive_bytes=destination.stat().st_size, archive_sha256=digest.hexdigest())
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--store", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.manifest, args.store, args.output)), flush=True)


if __name__ == "__main__":
    main()
