"""Prepare an isolated, budget-matched sampling pool from audited teacher decisions."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from .audit_transitions import archived_calls, event_calls
from .data import load_manifest

PROGRESS = {"native_xml_changed", "code_file_changed"}
ERROR = re.compile(r"(?m)^\s*(?:Error:|Traceback \(most recent call last\):|"
                   r"(?:Process|Command).*exit(?:ed)? (?:with )?code [1-9])")


def phase_guards(root, episode, stage, rows, effects):
    ordered = sorted(rows, key=lambda row: row["call_id"])
    ids = {row["call_id"]: row["sample_id"] for row in ordered}
    numbers = sorted(ids)
    protected = {ordered[0]["sample_id"], ordered[-1]["sample_id"]}
    for before, after in zip(ordered, ordered[1:]):
        if effects[after["sample_id"]]["artifact_effect"] in PROGRESS:
            protected.add(before["sample_id"])
    native = []
    for path in (root / "episodes" / episode / "generator").glob(f"round{stage}*/events.jsonl"):
        native.extend(event_calls(path))
    duplicates = len(native) != len({call["callId"] for call in native})
    results = {call["callId"]: call.get("result") for call in native}
    calls = archived_calls(root, episode, stage)
    uncertain, failed, matched = set(), set(), 0
    for call in calls:
        result = results.get(call["callId"])
        if not result:
            uncertain.add(call["source_decision"])
            continue
        matched += 1
        text = result.get("result", "")
        bad = result.get("status") != "completed" or bool(ERROR.search(text if isinstance(text, str) else ""))
        if bad:
            failed.add(call["source_decision"])
    for number in uncertain | failed:
        position = bisect_right(numbers, number)
        if position < len(numbers):
            protected.add(ids[numbers[position]])
    protected.update(ids[number] for number in uncertain if number in ids)
    if duplicates:
        protected.update(ids.values())
    receipt = {"api_tools": len(calls), "matched_native_results": matched,
               "uncertain_decisions": len(uncertain), "failed_or_error_marked_decisions": len(failed),
               "duplicate_native_tool_ids": duplicates}
    return protected, failed | uncertain, receipt


def resample(rows, effects, parts, fraction, seed, shorter=False, drawing_only=False):
    phases = defaultdict(list)
    thought = [parts[row["sample_id"]]["partition_tokens"].get("reasoning", 0) for row in rows]
    for index, row in enumerate(rows):
        phases[(row["episode"], row["stage"])].append((index, row))
    indices, replacements, audits = list(range(len(rows))), [], []
    for number, ((episode, stage), members) in enumerate(sorted(phases.items()), 1):
        phase = [row for _, row in members]
        protected, unsafe, audit = phase_guards(Path(phase[0]["snapshot_root"]), episode, stage, phase, effects)
        eligible = [index for index, row in members if row["sample_id"] not in protected
                    and effects[row["sample_id"]]["category"] == "other_tool"
                    and effects[row["sample_id"]]["artifact_effect"] == "no_case_file_change"]
        progress = [index for index, row in members
                    if effects[row["sample_id"]]["artifact_effect"] in PROGRESS
                    and row["call_id"] not in unsafe
                    and (not drawing_only or effects[row["sample_id"]]["artifact_effect"] == "native_xml_changed"
                         or effects[row["sample_id"]]["category"] == "xml_related")]
        key = lambda index: sha256(f"{seed}:{rows[index]['sample_id']}".encode()).hexdigest()
        eligible.sort(key=lambda index: (-thought[index], key(index)) if shorter else key(index))
        progress.sort(key=lambda index: (thought[index], key(index)) if shorter else key(index))
        count = min(len(eligible) - math.ceil(len(eligible) * fraction), len(progress))
        starting_replacements = len(replacements)
        for source, target in zip(eligible[:count], progress[:count]):
            if shorter and thought[target] >= thought[source]:
                break
            indices[source] = target
            replacements.append({"removed_index": source, "repeated_index": target,
                                 "removed_id": rows[source]["sample_id"],
                                 "repeated_id": rows[target]["sample_id"],
                                 "removed_reasoning_tokens": thought[source],
                                 "repeated_reasoning_tokens": thought[target]})
        audits.append({"episode": episode, "generation_stage": stage, **audit,
                       "protected": len(protected), "eligible_observations": len(eligible),
                       "replacements": len(replacements) - starting_replacements})
        if number % 20 == 0:
            print(json.dumps({"phases_completed": number, "total_phases": len(phases),
                              "replacements": len(replacements)}), flush=True)
    return indices, replacements, audits


def describe(indices, rows, effects, parts, boundaries):
    partitions, categories, difficulty = Counter(), Counter(), defaultdict(Counter)
    prompts = targets = images = squares = 0
    for index in indices:
        row = rows[index]
        ident = row["sample_id"]
        partitions.update(parts[ident]["partition_tokens"])
        categories[effects[ident]["category"]] += 1
        stage = difficulty[bisect_left(boundaries, row["final_components"])]
        stage["decisions"] += 1
        stage["prompt_tokens"] += row["prompt_tokens"]
        stage["target_tokens"] += row["completion_tokens"]
        prompts += row["prompt_tokens"]
        targets += row["completion_tokens"]
        images += row["image_count"]
        squares += row["token_count"] ** 2
    counts = Counter(indices)
    return {"decisions": len(indices), "unique_source_decisions": len(counts),
            "unique_episodes": len({rows[index]["episode"] for index in indices}),
            "maximum_source_multiplicity": max(counts.values()),
            "multiplicity_histogram": dict(Counter(counts.values())),
            "prompt_tokens": prompts, "target_tokens": targets, "image_inputs": images,
            "sequence_token_squared_proxy": squares,
            "partition_tokens": dict(partitions),
            "partition_fractions": {name: count / targets for name, count in partitions.items()},
            "categories": dict(categories),
            "difficulty_stages": {str(stage): dict(values) for stage, values in difficulty.items()}}


def validate_pool(indices, rows, effects):
    counts = Counter(indices)
    if len(indices) != len(rows) or max(counts.values()) > 2:
        raise ValueError("Sampling pool changed its decision budget or duplicate cap")
    before = Counter((row["episode"], row["stage"]) for row in rows)
    after = Counter((rows[index]["episode"], rows[index]["stage"]) for index in indices)
    if before != after:
        raise ValueError("Sampling changed episode or generation-stage exposure")
    protected = [index for index, row in enumerate(rows)
                 if effects[row["sample_id"]]["artifact_effect"] != "no_case_file_change"
                 or effects[row["sample_id"]]["category"] != "other_tool"]
    if any(counts[index] < 1 for index in protected):
        raise ValueError("Sampling omitted a protected observation or action")
    endings = [index for index, row in enumerate(rows)
               if effects[row["sample_id"]]["category"] == "assistant_only"]
    if any(counts[index] != 1 for index in endings):
        raise ValueError("Sampling changed closing-decision exposure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "actions", "parts", "curriculum", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--observation-keep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shorter-reasoning", action="store_true",
                        help="Replace only with a progress decision that has fewer reasoning tokens")
    parser.add_argument("--drawing-progress-only", action="store_true",
                        help="Repeat only XML changes or code changes with native Draw.io markers")
    args = parser.parse_args()
    if not 0 <= args.observation_keep <= 1:
        parser.error("observation-keep must be between 0 and 1")
    rows = load_manifest(args.manifest, "train")
    effects = {row["sample_id"]: row for row in load_manifest(args.actions)}
    parts = {row["sample_id"]: row for row in load_manifest(args.parts)}
    curriculum = json.loads(args.curriculum.read_text())
    if any(effects[row["sample_id"]]["target_tokens"] != row["completion_tokens"] for row in rows):
        raise ValueError("Action audit differs from the training manifest")
    indices, replacements, audits = resample(rows, effects, parts, args.observation_keep,
                                           args.seed, args.shorter_reasoning, args.drawing_progress_only)
    validate_pool(indices, rows, effects)
    baseline = describe(list(range(len(rows))), rows, effects, parts, curriculum["difficulty_boundaries"])
    candidate = describe(indices, rows, effects, parts, curriculum["difficulty_boundaries"])
    thought_grew = candidate["partition_tokens"]["reasoning"] > baseline["partition_tokens"]["reasoning"]
    if args.shorter_reasoning and thought_grew:
        raise ValueError("Shorter-reasoning sampling increased thought supervision")
    if {k: v["decisions"] for k, v in baseline["difficulty_stages"].items()} != {
            k: v["decisions"] for k, v in candidate["difficulty_stages"].items()}:
        raise ValueError("Difficulty-stage budgets changed")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "sft-sampling-pool/1", "at": time.time(),
            "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
            "action_audit_sha256": sha256(args.actions.read_bytes()).hexdigest(),
            "token_parts_sha256": sha256(args.parts.read_bytes()).hexdigest(),
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_curriculum_sha256": curriculum["source_indices_sha256"],
            "seed": args.seed, "observation_keep": args.observation_keep, "source_indices": indices,
            "shorter_reasoning": args.shorter_reasoning,
            "drawing_progress_only": args.drawing_progress_only,
            "pool_kind": "Source-index multiset; not optimizer execution order",
            "integration_gate": "Apply original length grouping; separate duplicate source IDs across optimizer updates.",
            "scope": "No prefixes, labels, images, parameters or validation decisions are rewritten.",
            "training_modified": False}
    (args.output / "pool.json").write_text(json.dumps(plan) + "\n")
    (args.output / "replacements.json").write_text(json.dumps(replacements) + "\n")
    (args.output / "native-outcome-coverage.json").write_text(json.dumps(audits) + "\n")
    summary = {key: value for key, value in plan.items() if key != "source_indices"}
    summary.update(baseline=baseline, candidate=candidate, replacements=len(replacements),
                   losses_of_overcontext_decisions_recovered=False,
                   quality_improvement_proven=False, pilot_started=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
