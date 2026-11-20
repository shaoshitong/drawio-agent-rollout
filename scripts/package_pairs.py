"""Freeze XML/PNG pairs without changing their bytes or upstream provenance."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path
import tarfile
import time

import zstandard


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = sha256()

    def read(self, size):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def add_payload(bundle, root, row, kind, name):
    path = root / row[kind + "_file"]
    info = bundle.gettarinfo(str(path), arcname=name)
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    with path.open("rb") as stream:
        reader = DigestReader(stream)
        bundle.addfile(info, reader)
    actual = reader.digest.hexdigest()
    if actual != row[kind + "_sha256"]:
        raise ValueError("Payload differs from frozen manifest: " + str(path))
    return info.size, actual


def add_bytes(bundle, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    bundle.addfile(info, io.BytesIO(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = args.manifest.read_bytes()
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    portable, checksums, total = [], [], 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".building")
    progress = args.output.with_suffix(".progress.json")
    parameters = zstandard.ZstdCompressionParameters.from_level(3, window_log=31, enable_ldm=True, threads=2)
    with ExitStack() as stack:
        destination = stack.enter_context(temporary.open("wb"))
        zipped = stack.enter_context(
            zstandard.ZstdCompressor(compression_params=parameters).stream_writer(destination, closefd=False)
        )
        bundle = stack.enter_context(tarfile.open(fileobj=zipped, mode="w|"))
        for index, row in enumerate(rows, 1):
            item = {**row, "upstream": {key: row[key] for key in ("code_file", "image_file")}}
            for kind, suffix in (("code", "editable.drawio"), ("image", "image.png")):
                name = "pairs/" + row["id"] + "/" + suffix
                size, digest = add_payload(bundle, args.root, row, kind, name)
                item[kind + "_file"] = name
                checksums.append(digest + "  " + name)
                total += size
            portable.append(item)
            if index % 100 == 0 or index == len(rows):
                state = {
                    "verified_pairs": index,
                    "total_pairs": len(rows),
                    "payload_bytes": total,
                    "compressed_bytes": destination.tell(),
                    "at": time.time(),
                }
                progress.write_text(json.dumps(state) + "\n")
                print(json.dumps(state), flush=True)
        metadata = {
            "schema": "drawio-gold-pairs/1",
            "pairs": len(rows),
            "files": 2 * len(rows),
            "sources": dict(Counter(row["source"] for row in rows)),
            "payload_bytes": total,
            "source_manifest_sha256": sha256(original).hexdigest(),
            "payload_bytes_unchanged": True,
        }
        add_bytes(bundle, "pairs.jsonl", "".join(json.dumps(row) + "\n" for row in portable).encode())
        add_bytes(bundle, "source-manifest.jsonl", original)
        add_bytes(bundle, "release.json", (json.dumps(metadata, indent=2) + "\n").encode())
        add_bytes(bundle, "SHA256SUMS", ("\n".join(checksums) + "\n").encode())
    temporary.replace(args.output)
    with args.output.open("rb") as stream:
        receipt = {
            **metadata,
            "bytes": args.output.stat().st_size,
            "sha256": file_digest(stream, "sha256").hexdigest(),
            "archive": args.output.name,
        }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
