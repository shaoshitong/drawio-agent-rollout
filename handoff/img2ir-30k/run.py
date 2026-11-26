"""Download the frozen assignment, run 12 Codex lanes, or pack the verified results."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download

from meanflow_batch import load_rows
from meanflow_two_stage import CODEX_CHATGPT, codex_chatgpt_available, write_json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = "sst12345/png-to-editable-drawio-20261120"
BRANCH = "img2ir-codex-30k"
INPUTS = "img2ir-codex-30k/inputs"
TARGET = 30_000
CANDIDATES = 36_000
LANES = 12


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def encode(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def download(work):
    pin = work / "input-revision.json"
    if pin.is_file():
        revision = json.loads(pin.read_text())["revision"]
    else:
        revision = HfApi().dataset_info(REPO, revision=BRANCH).sha

    def get(filename):
        return Path(hf_hub_download(REPO, f"{INPUTS}/{filename}", repo_type="dataset", revision=revision,
                                    local_dir=work / "download"))

    catalog = json.loads(get("catalog.json").read_text())
    if catalog["version"] != 1 or catalog["accepted_target"] != TARGET or catalog["input_images"] != CANDIDATES:
        raise ValueError("Unexpected input catalog")
    if not pin.is_file():
        write_json(pin, {"repository": REPO, "revision": revision})
    shards = work / "shards"
    shards.mkdir(exist_ok=True)

    def unpack(item):
        name = item["file"]
        if Path(name).name != name or not name.endswith(".tar"):
            raise ValueError(f"Invalid input shard name: {name}")
        folder = shards / Path(name).stem
        if not folder.is_dir():
            archive = get(name)
            if archive.stat().st_size != item["bytes"] or digest(archive) != item["sha256"]:
                raise ValueError(f"Input archive differs from the catalog: {name}")
            with TemporaryDirectory(dir=shards) as temporary:
                extracted = Path(temporary) / "contents"
                with tarfile.open(archive) as stream:
                    stream.extractall(extracted, filter="data")
                rows = load_rows(extracted / "manifest.jsonl")
                if len(rows) != item["rows"]:
                    raise ValueError(f"Wrong manifest count in {name}")
                for row in rows:
                    case = row["case_id"]
                    if not case.replace("-", "").isalnum() or row["file"] != f"png/{case}.png":
                        raise ValueError(f"Invalid case path in {name}")
                    if digest(extracted / row["file"]) != row["sha256"]:
                        raise ValueError(f"PNG digest mismatch: {case}")
                extracted.rename(folder)
        rows = load_rows(folder / "manifest.jsonl")
        print(f"Ready {name} ({len(rows)} PNGs)", flush=True)
        return [{**row, "image": str(folder / row["file"]), "input_shard": name} for row in rows]

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for shard in pool.map(unpack, catalog["shards"]) for row in shard]
    if (len(rows) != catalog["input_images"] or len({row["sha256"] for row in rows}) != len(rows)
            or len({row["case_id"] for row in rows}) != len(rows)):
        raise ValueError("The frozen assignment is incomplete or has duplicate inputs")
    manifest = work / "inputs.jsonl"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("".join(map(encode, rows)), encoding="utf-8")
    temporary.replace(manifest)


def run(work):
    if not codex_chatgpt_available(os.environ):
        raise RuntimeError("Sign in to the Codex CLI with ChatGPT before running the batch")
    command = [sys.executable, str(HERE / "meanflow_batch.py"), str(work / "inputs.jsonl"),
               "--compiler", str(ROOT), "--out", str(work / "generated"), "--ir-only",
               "--target-complete", str(TARGET), "--codex-chatgpt-jobs", str(LANES)]
    return subprocess.run(command).returncode


def pack(work):
    accepted = work / "generated" / "accepted.jsonl"
    records = load_rows(accepted)
    if len(records) != TARGET or len({row["sha256"] for row in records}) != TARGET:
        raise ValueError("Packing requires exactly 30,000 unique accepted PNGs")
    for row in records:
        report = json.loads(Path(row["report"]).read_text())
        req = report["ir_request"]
        if (req["provider"] != CODEX_CHATGPT or req["model"] != "gpt-6.2-sol"
                or req["reasoning_effort"] != "medium" or req["completed_turns"] != 1
                or report.get("image_sha256") != row["sha256"] or "xml_request" in report
                or req.get("reasoning_required") or req.get("tool_calls") != 0
                or not all(report["compiler_checks"].values())):
            raise ValueError(f"Unexpected request provenance for {row['case_id']}")
    dataset = work / "dataset"
    subprocess.run([sys.executable, str(HERE / "pack_sft.py"), "--ir-only", "--expected", str(TARGET),
                    "--out", str(dataset), str(accepted)], check=True)
    result = work / "result"
    result.mkdir(exist_ok=True)
    for name in ("dataset.json", "episodes.jsonl", "sft.jsonl"):
        copyfile(dataset / name, result / name)
    index = result / "input-index.jsonl"
    with index.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(encode({k: row[k] for k in ("case_id", "sha256", "input_shard", "file")}))
    shards = []
    for offset in range(0, len(records), 512):
        group = records[offset:offset + 512]
        path = result / f"ir-{offset // 512:03d}.tar"
        temporary = path.with_suffix(".tmp")
        with tarfile.open(temporary, "w") as archive:
            for row in group:
                case = row["case_id"]
                folder = Path("episodes") / case
                for name in ("ir.json", "ir.jsonl"):
                    archive.add(dataset / folder / name, arcname=str(folder / name))
                archive.add(row["report"], arcname=str(folder / "report.json"))
        temporary.replace(path)
        shards.append({"file": path.name, "rows": len(group), "sha256": digest(path), "bytes": path.stat().st_size})
    revision = json.loads((work / "input-revision.json").read_text())["revision"]
    write_json(result / "catalog.json", {"accepted": TARGET, "input_revision": revision, "input_repo": REPO,
                                         "input_prefix": INPUTS, "ir_shards": shards,
                                         "input_index": index.name, "sft": "sft.jsonl", "episodes": "episodes.jsonl"})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "run", "pack"))
    parser.add_argument("--work", type=Path, required=True, help="Persistent local workspace for inputs and outputs")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.exit({"download": download, "run": run, "pack": pack}[args.command](work) or 0)
