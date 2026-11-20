"""Archive selected remote artifacts with lossless long-window compression."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest
import json
from pathlib import Path
import tarfile

import zstandard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.output.with_suffix(args.output.suffix + ".building")
    parameters = zstandard.ZstdCompressionParameters.from_level(3, window_log=31, enable_ldm=True, threads=2)
    with ExitStack() as stack:
        destination = stack.enter_context(temporary.open("wb"))
        zipped = stack.enter_context(
            zstandard.ZstdCompressor(compression_params=parameters).stream_writer(destination, closefd=False)
        )
        bundle = stack.enter_context(tarfile.open(fileobj=zipped, mode="w|"))
        for name in args.include:
            bundle.add(args.root / name, arcname=name)
            print(json.dumps({"included": name, "compressed_bytes": destination.tell()}), flush=True)
    temporary.replace(args.output)
    with args.output.open("rb") as stream:
        receipt = {
            "archive": args.output.name,
            "bytes": args.output.stat().st_size,
            "sha256": file_digest(stream, "sha256").hexdigest(),
            "includes": args.include,
        }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
