"""Recompress an immutable tar stream with a long Zstandard deduplication window."""

import argparse
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import time

import zstandard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    while not args.receipt.exists():
        time.sleep(2)
    original = json.loads(args.receipt.read_text())
    parameters = zstandard.ZstdCompressionParameters.from_level(3, window_log=31, enable_ldm=True, threads=2)
    compressor = zstandard.ZstdCompressor(compression_params=parameters)
    total = 0
    with gzip.open(args.input, "rb") as source, args.output.open("wb") as destination:
        with compressor.stream_writer(destination, closefd=False) as output:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(block)
                total += len(block)
                if total % (128 * 1024 * 1024) == 0:
                    output.flush(zstandard.FLUSH_BLOCK)
                    print(
                        json.dumps(
                            {"tar_bytes": total, "compressed_bytes": destination.tell(), "at": time.time()}
                        ),
                        flush=True,
                    )
    with args.output.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    receipt = {
        **original,
        "archive": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "original_gzip_sha256": original["sha256"],
        "tar_bytes": total,
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
