"""Qualify complete generation phases and select unique supervised decisions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import time

from .causal_cot import digest
from .check_prefix_chains import Reader
from .data import load_manifest
from .rewrite_cot import atomic, select_rows
from .sampling_pool import phase_guards
from .state_prefix import NativeCounter, build_state_prefix, snapshot_before_request

POLICY = "complete-phase-cap64-v1"
PROGRESS = {"native_xml_changed", "code_file_changed", "candidate_render_changed"}


def sample_phase(rows, effects, cap, seed):
    first = rows[0]
    protected, _, guards = phase_guards(Path(first["snapshot_root"]), first["episode"],
                                        first["stage"], rows, effects)
    protected.update(row["sample_id"] for row in rows
                     if effects[row["sample_id"]]["artifact_effect"] in PROGRESS
                     or effects[row["sample_id"]]["category"] == "xml_related")
    others = sorted((row["sample_id"] for row in rows if row["sample_id"] not in protected),
                    key=lambda ident: sha256(f"{seed}:{ident}".encode()).hexdigest())
    selected = protected | set(others[:max(0, cap - len(protected))])
    return selected, {**guards, "protected": len(protected), "retained": len(selected)}


def qualify_episode(rows, processor, effects, cap, seed):
    reader = Reader(rows[0]["snapshot_root"])
    phases = defaultdict(list)
    for row in rows:
        phases[row["stage"]].append(row)
    decisions, receipts, blocked = [], [], False
    for stage, members in sorted(phases.items()):
        members.sort(key=lambda row: row["call_id"])
        terminal = members[-1]
        receipt = {"episode": terminal["episode"], "stage": stage, "split": terminal["split"],
                   "final_components": terminal["final_components"], "decisions": len(members),
                   "terminal_sample_id": terminal["sample_id"], "qualified": False}
        reason = "after_unfit_phase"
        if not blocked:
            request = reader.value(terminal["request_ref"])
            files, snapshot = snapshot_before_request(terminal, reader.store)
            counter = NativeCounter(processor, request.get("tools", []), reader.store.raw)
            prefix, proof = build_state_prefix(request["messages"], files,
                lambda name: reader.store.raw(files[name]), stage, counter)
            blocked = prefix is None
            receipt.update(qualified=not blocked, snapshot=snapshot, state=proof)
            reason = "terminal_protected_state_exceeds_budget"
        selected = set()
        if not blocked:
            if terminal["split"] == "train":
                selected, guards = sample_phase(members, effects, cap, seed)
                receipt.update(guards)
            else:
                selected = {row["sample_id"] for row in members}
            reason = "long_phase_downsampled"
        receipt["retained"] = len(selected)
        receipts.append(receipt)
        decisions.extend({"sample_id": row["sample_id"], "episode": row["episode"],
                          "stage": stage, "split": row["split"], "keep": row["sample_id"] in selected,
                          "reason": "retained" if row["sample_id"] in selected else reason}
                         for row in members)
    return decisions, receipts


def load_selection(path, rows):
    plan = json.loads(Path(path).read_text())
    decisions = {row["sample_id"]: row for row in plan["decisions"]}
    if plan["policy"] != POLICY or plan["source_index_sha256"] != digest(rows):
        raise ValueError("Selection plan does not match this source corpus")
    if len(plan["decisions"]) != len(rows) or len(decisions) != len(rows) or set(decisions) != {row["sample_id"] for row in rows}:
        raise ValueError("Selection plan must cover every source decision exactly once")
    return decisions


def main():
    from setproctitle import setproctitle
    from transformers import AutoProcessor

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "actions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = select_rows(args.data)
    effects = {row["sample_id"]: row for row in load_manifest(args.actions)}
    train_ids = {row["sample_id"] for row in rows if row["split"] == "train"}
    if not train_ids <= effects.keys():
        raise ValueError("Source action audit is missing training decisions")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row)
    decisions, phases, started = [], [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(qualify_episode, group, processor, effects, 64, 3407)
                   for group in episodes.values()]
        for number, future in enumerate(as_completed(pending), 1):
            selected, receipts = future.result()
            decisions.extend(selected)
            phases.extend(receipts)
            progress = {"episodes": number, "total_episodes": len(episodes),
                        "decisions": len(decisions), "retained": sum(row["keep"] for row in decisions),
                        "at": time.time(), "started": started}
            atomic(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    order = {row["sample_id"]: index for index, row in enumerate(rows)}
    decisions.sort(key=lambda row: order[row["sample_id"]])
    phases.sort(key=lambda row: (row["episode"], row["stage"]))
    plan = {"policy": POLICY, "source_commit": args.source_commit,
            "source_index_sha256": digest(rows), "seed": 3407, "phase_cap": 64,
            "actions_sha256": sha256(args.actions.read_bytes()).hexdigest(),
            "offline_selection_only": True, "prefix_or_editor_receives_selection": False,
            "decisions": decisions, "phases": phases}
    atomic(args.output / "selection.json", plan)
    load_selection(args.output / "selection.json", rows)
    totals = Counter((row["split"], row["reason"]) for row in decisions)
    summary = {"episodes": len(episodes), "decisions": len(rows),
               "retained": sum(row["keep"] for row in decisions), "phases": len(phases),
               "qualified_phases": sum(row["qualified"] for row in phases),
               "counts": {f"{split}/{reason}": count for (split, reason), count in sorted(totals.items())},
               "elapsed_seconds": time.time() - started, "all_source_decisions_covered": True}
    atomic(args.output / "completed.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
