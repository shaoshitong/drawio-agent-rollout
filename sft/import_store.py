"""Import a frozen archive into a lossless packed store for shared filesystems."""

import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_members(archive):
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            yield member.name, source.extractfile(member).read()


def insert_object(connection, name, payload):
    path = Path(name)
    reference = path.parent.name + path.stem
    raw = gzip.decompress(payload) if path.suffix == ".gz" else payload
    if sha256(raw).hexdigest() != reference:
        raise ValueError("Corrupt archive object: " + reference)
    connection.execute("insert into objects values(?,?)", (reference, raw))
    return len(raw)


def import_archive(archive, expected_sha256, output):
    output.mkdir(parents=True)
    progress = output / "progress.json"
    progress.write_text(json.dumps({"phase": "verifying_archive", "at": time.time()}) + "\n")
    if file_sha256(archive) != expected_sha256:
        raise ValueError("Frozen archive SHA256 mismatch")
    folder = output / "archive"
    folder.mkdir()
    partial = folder / "objects.sqlite3.partial"
    connection = sqlite3.connect(partial)
    connection.executescript("""
        pragma journal_mode=off;
        pragma synchronous=off;
        pragma page_size=32768;
        pragma cache_size=-262144;
        create table objects(id text primary key, data blob) without rowid;
    """)
    count = raw_bytes = 0
    metadata = None
    for name, payload in archive_members(archive):
        if name == "curriculum.jsonl":
            (output / name).write_bytes(payload)
        elif name == "manifest.json":
            metadata = json.loads(payload)
        else:
            raw_bytes += insert_object(connection, name, payload)
            count += 1
        if count and count % 10000 == 0:
            progress.write_text(json.dumps({"phase": "packing", "objects": count,
                                           "raw_bytes": raw_bytes, "at": time.time()}) + "\n")
    connection.commit()
    connection.close()
    if count != metadata["objects"] or raw_bytes != metadata["raw_object_bytes"]:
        raise ValueError("Packed object inventory differs from the source manifest")
    manifest = output / "curriculum.jsonl"
    if file_sha256(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Packed SFT index differs from the frozen source")
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    progress.write_text(json.dumps({"phase": "fresh_verification", "objects": 0,
                                   "at": time.time()}) + "\n")
    verified = verified_bytes = 0
    with sqlite3.connect(partial.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as fresh:
        for reference, raw in fresh.execute("select id,data from objects"):
            if sha256(raw).hexdigest() != reference:
                raise ValueError("Corrupt packed object: " + reference)
            verified += 1
            verified_bytes += len(raw)
    if (verified, verified_bytes) != (count, raw_bytes):
        raise ValueError("Fresh packed-store inventory mismatch")
    final = partial.with_suffix("")
    partial.rename(final)
    result = {"at": time.time(), "phase": "verified", "objects": count,
              "raw_object_bytes": raw_bytes, "archive_sha256": expected_sha256,
              "manifest_sha256": metadata["manifest_sha256"],
              "database_bytes": final.stat().st_size, "database_sha256": file_sha256(final),
              "fresh_objects_verified": verified, "sample_content_changed": False}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    progress.write_text(json.dumps(result) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.archive_sha256, args.output)))


if __name__ == "__main__":
    main()
