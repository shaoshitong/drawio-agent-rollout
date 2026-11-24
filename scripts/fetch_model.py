"""Download a fixed Hugging Face revision and verify every upstream file hash."""

import argparse
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from urllib.parse import quote
from urllib.request import urlopen

from huggingface_hub import snapshot_download


def modelscope_weights(model, output, files):
    import requests

    session = requests.Session()
    session.trust_env = False
    response = session.get(f"https://modelscope.cn/api/v1/models/{model}/repo/files",
                           params={"Revision": "master", "Recursive": "true"}, timeout=(15, 30))
    response.raise_for_status()
    upstream = {row["Path"]: row for row in response.json()["Data"]["Files"]}
    weights = [row for row in files if row["rfilename"].endswith(".safetensors")]
    if any(upstream[row["rfilename"]]["Sha256"] != row["lfs"]["sha256"] or
           upstream[row["rfilename"]]["Size"] != row["size"] for row in weights):
        raise ValueError("ModelScope weights differ from the pinned HF revision")
    entries = []
    cache = output / ".cache/huggingface/download"
    for row in weights:
        name = row["rfilename"]
        partial = next(cache.glob(name + ".*.incomplete"), None)
        if partial is not None and not (output / name).exists():
            partial.rename(output / name)
        entries.extend((f"https://modelscope.cn/models/{model}/resolve/master/{quote(name, safe='/')}",
                        f" dir={output}", f" out={name}"))
    listing = output / "modelscope-download.txt"
    listing.write_text("\n".join(entries) + "\n")
    subprocess.run(["aria2c", "--input-file=" + str(listing), "--continue=true",
                    "--auto-file-renaming=false", "--file-allocation=none",
                    "--max-concurrent-downloads=4", "--split=8", "--max-connection-per-server=8",
                    "--min-split-size=4M", "--connect-timeout=15", "--timeout=60",
                    "--max-tries=3", "--retry-wait=3", "--enable-color=false",
                    "--summary-interval=15"], check=True,
                   env={key: value for key, value in os.environ.items()
                        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--modelscope-weights", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files against the saved upstream manifest")
    args = parser.parse_args()
    address = f"{args.endpoint}/api/models/{args.model}/revision/{args.revision}?blobs=true"
    if args.verify_only:
        metadata = json.loads((args.output / "source-manifest.json").read_text())
    else:
        with urlopen(address, timeout=30) as response:
            metadata = json.load(response)
    assert metadata["sha"] == args.revision
    files = [row for row in metadata["siblings"] if row["rfilename"].endswith(
        (".json", ".jinja", ".txt", ".safetensors")) or row["rfilename"] == "LICENSE"]
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        (args.output / "source-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"phase": "download", "revision": args.revision,
                          "expected_bytes": sum(row["size"] for row in files)}), flush=True)
        selected = [row["rfilename"] for row in files
                    if not args.modelscope_weights or not row["rfilename"].endswith(".safetensors")]
        snapshot_download(args.model, revision=args.revision, endpoint=args.endpoint,
                          local_dir=args.output, allow_patterns=selected,
                          max_workers=args.workers, token=False)
        if args.modelscope_weights:
            modelscope_weights(args.model, args.output, files)
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
    receipt["weights_transport"] = ("verified-existing-files" if args.verify_only
                                    else "modelscope-aria2" if args.modelscope_weights else args.endpoint)
    (args.output / "verification-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
