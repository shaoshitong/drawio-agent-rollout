"""Stream original content-addressed probe images and verify their native RGB pixels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile


def pack(roots):
    assets = json.load(sys.stdin)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for digest, expected in assets.items():
            relative = Path(digest[:2]) / (digest[2:] + ".bin")
            path = next((root / relative for root in roots if (root / relative).is_file()), None)
            if path is None:
                raise FileNotFoundError("Original probe image is missing: " + digest)
            raw = path.read_bytes()
            if len(raw) != expected["bytes"] or sha256(raw).hexdigest() != digest:
                raise ValueError("Original probe image hash differs: " + digest)
            info = archive.gettarinfo(path, arcname="images/" + digest + ".bin")
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def verify(directory):
    from PIL import Image

    assets = json.loads((directory / "assets.json").read_text())
    total = 0
    for digest, expected in assets.items():
        path = directory / "images" / (digest + ".bin")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or sha256(raw).hexdigest() != digest:
            raise ValueError("Received probe image hash differs: " + digest)
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixels = sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if list(image.size) != expected["rgb_size"] or pixels != expected["rgb_sha256"]:
            raise ValueError("Native training image pixels differ: " + digest)
        total += len(raw)
    report = {"images": len(assets), "original_image_bytes": total,
              "all_original_and_rgb_hashes_verified": True, "raw_rollouts_saved_on_mac": False}
    (directory / "asset-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    source = sub.add_parser("pack")
    source.add_argument("--objects-root", type=Path, action="append", required=True)
    target = sub.add_parser("verify")
    target.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    pack(args.objects_root) if args.mode == "pack" else verify(args.directory)


if __name__ == "__main__":
    main()
