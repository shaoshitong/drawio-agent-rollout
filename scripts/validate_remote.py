"""Verify and extract arriving fragments, then resume the original H20 suites."""

import argparse
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from threading import Thread
import time


class Fragments(io.RawIOBase):
    def __init__(self, archive):
        self.archive = archive
        self.parts = Path(str(archive) + ".parts")
        self.index, self.buffer = 0, b""

    def readable(self):
        return True

    def read(self, size=-1):
        while not self.buffer:
            matches = list(self.parts.glob(f"{self.index:08d}.*.part"))
            if matches:
                path = matches[0]
                self.buffer = path.read_bytes()
                if sha256(self.buffer).hexdigest() != path.name.split(".")[1]:
                    raise ValueError("Fragment hash mismatch: " + path.name)
                self.index += 1
            elif self.archive.exists():
                return b""
            else:
                time.sleep(2)
        if size < 0:
            raise ValueError("Unbounded reads are unsupported")
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result


def extract(archive, data, errors):
    manifest = json.loads((data / "manifest.json").read_text())
    done = 0
    try:
        if archive.suffix == ".zst":
            import zstandard

            zipped = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(Fragments(archive))
        else:
            zipped = gzip.GzipFile(fileobj=Fragments(archive))
        with zipped:
            with tarfile.open(fileobj=zipped, mode="r|", ignore_zeros=True) as bundle:
                for member in bundle:
                    if not member.isfile():
                        continue
                    if member.name not in manifest and member.name != "manifest.json":
                        raise ValueError("Unexpected archive member: " + member.name)
                    path = data / member.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + ".extracting")
                    digest = sha256()
                    with temporary.open("wb") as output, bundle.extractfile(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                            digest.update(block)
                    if member.name in manifest and digest.hexdigest() != manifest[member.name]:
                        raise ValueError("Input hash mismatch: " + member.name)
                    temporary.replace(path)
                    done += 1
                    if done % 100 == 0:
                        print(json.dumps({"extracted_verified": done, "at": time.time()}), flush=True)
        if done != len(manifest) + 1:
            raise ValueError(f"Archive members: {done}, expected {len(manifest) + 1}")
    except Exception as error:
        errors.append(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data, output = root / "data", root / "checks"
    rows = [json.loads(line) for line in (data / "corpus.jsonl").read_text().splitlines()]
    early = root / "early-data"
    early.mkdir(exist_ok=True)
    for name in ("baseline", "validation", "corpus"):
        if not (early / name).exists():
            (early / name).symlink_to(data / name, target_is_directory=True)
    errors = []
    thread = Thread(
        target=extract, args=(args.archive or root / "compiler-validation.tar.gz", data, errors), daemon=True
    )
    thread.start()
    last_count = 0
    runner = Path(__file__).resolve().parents[1] / "test/replay.py"
    while thread.is_alive() or last_count < len(rows):
        if errors:
            raise errors[0]
        available = [row for row in rows if (data / row["code_file"]).is_file()]
        if len(available) >= last_count + 100 or len(available) == len(rows):
            (early / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in available))
            command = [
                sys.executable,
                str(runner),
                "--data",
                str(early),
                "--out",
                str(output / "corpus"),
                "--suite",
                "corpus",
                "--workers",
                str(args.workers),
            ]
            subprocess.run(command, check=True)
            last_count = len(available)
        else:
            time.sleep(5)
    thread.join()
    if errors:
        raise errors[0]
    for suite in ("variants", "controls", "generalization"):
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--data",
                str(data),
                "--out",
                str(output / suite),
                "--suite",
                suite,
                "--workers",
                str(args.workers),
            ],
            check=True,
        )
    print(json.dumps({"all_suites_complete": True, "at": time.time()}), flush=True)


if __name__ == "__main__":
    main()
