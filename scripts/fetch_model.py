"""Download a fixed Hugging Face revision and verify every upstream file hash."""

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path
import struct
import time
from urllib.request import urlopen

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    address = f"{args.endpoint}/api/models/{args.model}/revision/{args.revision}?blobs=true"
    with urlopen(address, timeout=30) as response:
        metadata = json.load(response)
    assert metadata["sha"] == args.revision
    files = [row for row in metadata["siblings"] if row["rfilename"].endswith(
        (".json", ".jinja", ".txt", ".safetensors")) or row["rfilename"] == "LICENSE"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"phase": "download", "revision": args.revision,
                      "expected_bytes": sum(row["size"] for row in files)}), flush=True)
    snapshot_download(args.model, revision=args.revision, endpoint=args.endpoint,
                      local_dir=args.output, allow_patterns=[row["rfilename"] for row in files],
                      max_workers=args.workers, token=False)
    for row in files:
        path = args.output / row["rfilename"]
        assert path.stat().st_size == row["size"], path.name
        digest = sha256() if row.get("lfs") else sha1(f"blob {row['size']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        expected = row["lfs"]["sha256"] if row.get("lfs") else row["blobId"]
        assert digest.hexdigest() == expected, path.name
    index = json.loads((args.output / "model.safetensors.index.json").read_text())["weight_map"]
    observed = {}
    for shard in sorted(set(index.values())):
        with (args.output / shard).open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        observed.update({name: shard for name in header if name != "__metadata__"})
    assert observed == index
    receipt = {"model": args.model, "revision": args.revision, "files": len(files),
               "bytes": sum(row["size"] for row in files), "tensors": len(index),
               "all_upstream_hashes_verified": True, "verified_at": time.time()}
    (args.output / "verification-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
