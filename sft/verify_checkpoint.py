"""Verify every file and core tensor header in a portable full checkpoint."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from .merge_lora import sha256
from .prepare_checkpoint import tensor_header


def verify(directory):
    manifest = json.loads((directory / "inference-provenance.json").read_text())
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Checkpoint file differs from its source: " + name)
    tensors = tensor_header(directory / "model.safetensors")
    actual = {"tensors": len(tensors), "parameters": sum(math.prod(value["shape"]) for value in tensors.values()),
              "weight_dtypes": sorted({value["dtype"] for value in tensors.values()})}
    if any(actual[key] != manifest[key] for key in actual):
        raise ValueError("Checkpoint tensor header differs from its source")
    return {"verified": True, "global_step": manifest["global_step"], **actual,
            "files_verified": len(manifest["files"]),
            "model_sha256": manifest["files"]["model.safetensors"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.archive:
        import zstandard

        args.directory.parent.mkdir(parents=True, exist_ok=False)
        with args.archive.open("rb") as source, zstandard.ZstdDecompressor(
                max_window_size=2**31).stream_reader(source) as decoded:
            with subprocess.Popen(["tar", "-xf", "-", "-C", str(args.directory.parent)],
                                  stdin=subprocess.PIPE) as child:
                shutil.copyfileobj(decoded, child.stdin, length=8 * 1024**2)
                child.stdin.close()
                if child.wait():
                    raise RuntimeError("Checkpoint archive extraction failed")
    report = verify(args.directory)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
