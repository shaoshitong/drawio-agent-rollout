"""Prepare paired, all-difficulty pilots without changing archived decisions."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import load_manifest
from .sampling_pool import describe, separate_update_duplicates


def select_slots(rows, pool, effects, boundaries, per_stage, seed):
    stages = defaultdict(list)
    for index, row in enumerate(rows):
        stages[bisect_left(boundaries, row["final_components"])].append(index)
    selected = {}
    for stage, indices in sorted(stages.items()):
        phases = defaultdict(list)
        for index in indices:
            row = rows[index]
            effect = effects[row["sample_id"]]
            observation = effect["category"] in {"image_read", "crop_related", "other_tool"}
            if pool[index] == index and observation and effect["artifact_effect"] == "no_case_file_change":
                phases[(row["episode"], row["stage"])].append(index)
        expected = {(rows[index]["episode"], rows[index]["stage"]) for index in indices}
        if phases.keys() != expected:
            raise ValueError("A pilot phase has no unchanged observation anchor")
        anchors = {min(members, key=lambda index: rows[index]["call_id"]) for members in phases.values()}
        if not len(anchors) <= per_stage <= len(indices):
            raise ValueError("Pilot budget cannot cover all phases in this difficulty stage")
        key = lambda index: sha256(f"{seed}:{rows[index]['sample_id']}".encode()).hexdigest()
        others = sorted(set(indices) - anchors, key=key)
        selected[stage] = sorted(anchors) + others[:per_stage - len(anchors)]
    return selected


def execution_order(selected, pool, rows, seed, update_size, microbatch):
    import torch
    from transformers.trainer_pt_utils import LengthGroupedSampler

    order = []
    for stage, slots in sorted(selected.items()):
        sources = [pool[index] for index in slots]
        sampler = LengthGroupedSampler(
            batch_size=microbatch, lengths=[rows[index]["token_count"] for index in sources],
            generator=torch.Generator(device="cpu").manual_seed(seed + stage),
        )
        grouped = [sources[index] for index in sampler]
        grouped, _ = separate_update_duplicates(grouped, rows, update_size, len(order))
        order.extend(grouped)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pool", "actions", "parts", "curriculum", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--update-size", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    rows = load_manifest(args.manifest, "train")
    pool = json.loads(args.pool.read_text())
    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    if pool["manifest_sha256"] != manifest_hash:
        raise ValueError("Sampling pool belongs to another manifest")
    effects = {row["sample_id"]: row for row in load_manifest(args.actions)}
    parts = {row["sample_id"]: row for row in load_manifest(args.parts)}
    boundaries = json.loads(args.curriculum.read_text())["difficulty_boundaries"]
    stages = len(boundaries) + 1
    config = json.loads(args.config.read_text())
    microbatch = config["batch_size"] * args.world_size
    if args.updates % stages or args.update_size % microbatch:
        parser.error("Updates must cover equal whole stages; update size must cover whole microbatches")
    selected = select_slots(rows, pool["source_indices"], effects, boundaries,
                            args.updates // stages * args.update_size, config["seed"])
    baseline = execution_order(selected, list(range(len(rows))), rows, config["seed"], args.update_size, microbatch)
    candidate = execution_order(selected, pool["source_indices"], rows, config["seed"], args.update_size, microbatch)
    phases = lambda order: Counter((rows[index]["episode"], rows[index]["stage"]) for index in order)
    endings = lambda order: Counter(rows[index]["sample_id"] for index in order
                                   if effects[rows[index]["sample_id"]]["category"] == "assistant_only")
    if phases(baseline) != phases(candidate) or endings(baseline) != endings(candidate):
        raise ValueError("Paired pilot changed episode, phase or closing exposure")
    args.output.mkdir(parents=True)
    report = {"manifest_sha256": manifest_hash, "pool_sha256": sha256(args.pool.read_bytes()).hexdigest(),
              "updates": args.updates, "effective_batch": args.update_size, "world_size": args.world_size,
              "updates_per_difficulty_stage": args.updates // stages, "unchanged_contexts": True,
              "paired_phase_and_closing_exposure": True, "model_training_started": False, "arms": {}}
    for name, order in (("D0", baseline), ("B3", candidate)):
        if any(len(set(order[start:start + args.update_size])) != len(order[start:start + args.update_size])
               for start in range(0, len(order), args.update_size)):
            raise ValueError("A pilot optimizer update repeats a source decision")
        arm = {**config, "gate_indices": order, "max_steps": args.updates,
               "gradient_accumulation_steps": args.update_size // microbatch,
               "evaluation_enabled": True, "save_steps": args.updates // 4,
               "preflight_samples": 4, "warmup_ratio": 0.03,
               "output": str(args.output / name)}
        (args.output / (name + ".json")).write_text(json.dumps(arm, indent=2) + "\n")
        report["arms"][name] = {**describe(order, rows, effects, parts, boundaries),
                                "order_sha256": sha256(json.dumps(order).encode()).hexdigest()}
    (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
