"""Check refreshed SFT rows with the actual training processor before staging."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time

from .data import load_manifest
from .prepare_trajectories import initialize_worker, measure
from .rewrite_cot import atomic


IDENTITY = ("episode", "split", "stage", "call_id", "request_ref", "response_ref", "snapshot_root")


def collect(baseline, refresh):
    fingerprint = sha256(baseline.read_bytes()).hexdigest()
    plan = json.loads((refresh / "plan.json").read_text())
    if plan["manifest_sha256"] != fingerprint:
        raise ValueError("Refreshed records were based on a different manifest")
    original = {row["sample_id"]: row for row in load_manifest(baseline)}
    excluded = json.loads((refresh / "failures.json").read_text())
    missed = {item["sample_id"] for item in excluded}
    updated = {}
    for path in refresh.glob("shard*/episodes/*/*.json"):
        if not path.stem.isdigit():
            continue
        receipt = json.loads(path.read_text())
        row, proof = receipt["row"], receipt["proof"]
        ident = row["sample_id"]
        if ident in updated or any(row[key] != original[ident][key] for key in IDENTITY):
            raise ValueError("Refresh identity mismatch: " + ident)
        if not proof["answer_fields_unchanged"] or proof["history_thinking_messages"] != 0:
            raise ValueError("Refresh target or history verification failed: " + ident)
        updated[ident] = row
    if updated.keys() & missed or updated.keys() | missed != original.keys():
        raise ValueError("Refresh records and explicit failures do not partition the baseline")
    return [updated[ident] for ident in original if ident in updated], excluded, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "refresh", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows, prior_failures, baseline_sha = collect(args.baseline, args.refresh)
    args.output.mkdir(parents=True)
    pending = args.output / "staged.building.jsonl"
    rejected = [dict(record, stage="state_refresh") for record in prior_failures]
    atomic(args.output / "rejected.json", rejected)
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.model), 32768, True)) as executor, pending.open("x") as file:
        for index, (row, checked) in enumerate(zip(rows, executor.map(measure, rows, chunksize=8)), 1):
            if checked.get("rejection"):
                rejected.append({**checked, "stage": "native_training_processor"})
            else:
                file.write(json.dumps({**row, **checked}) + "\n")
                counts[row["split"]] += 1
            if index % 100 == 0 or index == len(rows):
                file.flush()
                progress = {"at": time.time(), "processed": index, "expected": len(rows),
                            "accepted": dict(counts), "rejected": len(rejected), "training_started": False}
                atomic(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    staged = args.output / "staged.jsonl"
    pending.replace(staged)
    atomic(args.output / "rejected.json", rejected)
    report = {"at": time.time(), "baseline_sha256": baseline_sha, "expected": len(rows),
              "accepted": dict(counts), "previous_refresh_failures": len(prior_failures),
              "native_measurement_failures": len(rejected) - len(prior_failures),
              "staged_sha256": sha256(staged.read_bytes()).hexdigest(), "training_started": False}
    atomic(args.output / "completed.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
