"""Describe the actual supervised actions seen by a curriculum checkpoint."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sqlite3
import time

from .data import open_store
from .train_unsloth import Examples, curriculum_order


def category(message):
    tools = message.get("tool_calls", [])
    arguments = "\n".join(str(call["function"].get("arguments", "")) for call in tools)
    names = [call["function"]["name"] for call in tools]
    if any(marker in arguments for marker in (".drawio", "mxCell", "mxGraphModel", "mxGeometry")):
        return "xml_related"
    if re.search(r"\.crop\s*\(", arguments):
        return "crop_related"
    if "read_image" in names:
        return "image_read"
    return "other_tool" if tools else "assistant_only"


def snapshot_pairs(root):
    database = Path(root) / "archive/ledger.sqlite"
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
        calls = connection.execute("select id,episode,stage,artifacts from calls where "
            "role='generator' and artifacts is not null order by id").fetchall()
    phases = defaultdict(list)
    for number, episode, stage, reference in calls:
        phases[(episode, stage)].append((number, reference))
    return {before[0]: (before[1], after[1]) for phase in phases.values()
            for before, after in zip(phase, phase[1:])}


def artifact_effect(store, pairs, call_id):
    if call_id not in pairs:
        return "next_snapshot_unavailable", 0, None
    before_ref, after_ref = pairs[call_id]
    before, after = store.get(before_ref), store.get(after_ref)
    has_xml = any(Path(name).suffix in (".drawio", ".xml") for name in before)
    changed = {name: digest for name, digest in after.items() if before.get(name) != digest}
    paths = [Path(name) for name in changed]
    if any(path.suffix in (".drawio", ".xml") for path in paths):
        return "native_xml_changed", 0, has_xml
    if any(path.suffix in (".py", ".js", ".sh") for path in paths):
        return "code_file_changed", 0, has_xml
    if any(re.fullmatch(r"round\d+\.png", path.name) for path in paths):
        return "candidate_render_changed", 0, has_xml
    images = [digest for name, digest in changed.items() if Path(name).suffix in (".png", ".jpg", ".webp")]
    if images:
        duplicates = sum(digest in before.values() for digest in images)
        return ("only_duplicate_images_written" if duplicates == len(images) else "image_files_changed"), duplicates, has_xml
    return ("other_case_file_changed" if changed else "no_case_file_change"), 0, has_xml


def trajectory_summaries(details):
    episodes = defaultdict(list)
    for row in details:
        episodes[row["sample_id"].split(":")[0]].append(row)
    result = []
    for episode, rows in sorted(episodes.items()):
        first = [row for row in rows if row["generation_stage"] == 1]
        writes = [index + 1 for index, row in enumerate(first) if row["artifact_effect"] == "native_xml_changed"]
        code = [index + 1 for index, row in enumerate(first)
                if row["artifact_effect"] in ("native_xml_changed", "code_file_changed")]
        inspections = [row for row in rows if row["pixel_inspection"]]
        result.append({"episode": episode, "decisions": len(rows), "generation1_decisions": len(first),
            "initial_xml_present": first[0]["xml_present_before"] if first else None,
            "first_xml_change_decision": writes[0] if writes else None,
            "first_xml_change_fraction": writes[0] / len(first) if writes else None,
            "first_code_or_xml_decision": code[0] if code else None,
            "first_code_or_xml_fraction": code[0] / len(first) if code else None,
            "native_xml_change_decisions": len([row for row in rows if row["artifact_effect"] == "native_xml_changed"]),
            "effect_counts": dict(Counter(row["artifact_effect"] for row in rows)),
            "category_counts": dict(Counter(row["category"] for row in rows)),
            "pixel_inspections": len(inspections),
            "pixel_inspections_before_xml": sum(row["xml_present_before"] is False for row in inspections),
            "pixel_inspections_after_xml": sum(row["xml_present_before"] is True for row in inspections),
            "reasoning_characters": sum(row["reasoning_characters"] for row in rows),
            "completion_tokens": sum(row["target_tokens"] for row in rows)})
    return result


def source_rows(path, rows, selected):
    retained = {rows[index]["sample_id"]: rows[index] for index in selected}
    episodes = {row["episode"]: row for row in retained.values()}
    output = []
    with path.open() as stream:
        for line in stream:
            original = json.loads(line)
            episode = original["episode"]
            if episode not in episodes:
                continue
            ident = f"{episode}:{original['call_id']}"
            if ident in retained:
                output.append({**retained[ident], "used_for_sft": True})
            else:
                output.append({**episodes[episode], **original, "sample_id": ident,
                    "completion_tokens": 0, "used_for_sft": False,
                    "excluded_reason": "context_limit" if original.get("supervision_eligible", True)
                                       else "protocol_ineligible"})
    output.sort(key=lambda row: (row["episode"], row["call_id"]))
    return output


def audit(args):
    config = json.loads(args.config.read_text())
    examples = Examples(args.manifest, "train", True)
    order, plan = curriculum_order(examples, config, args.world_size)
    expected = json.loads(args.plan.read_text())["source_indices_sha256"]
    if plan["source_indices_sha256"] != expected:
        raise ValueError("Reconstructed order differs from the actual training sampler")
    limit = args.updates * args.world_size * config["batch_size"] * config["gradient_accumulation_steps"]
    selected = order[:limit]
    sampling = {"mode": "actual_checkpoint_prefix", "updates": args.updates}
    complete_episodes = args.episodes or args.seen_episodes
    if complete_episodes:
        pool = sorted({row["episode"] for row in examples.dataset.rows})
        chosen = ({examples.dataset.rows[index]["episode"] for index in selected} if args.seen_episodes
                  else set(random.Random(args.seed).sample(pool, min(args.episodes, len(pool)))))
        selected = [index for index, row in enumerate(examples.dataset.rows) if row["episode"] in chosen]
        selected.sort(key=lambda index: (examples.dataset.rows[index]["episode"], examples.dataset.rows[index]["call_id"]))
        sampling = {"mode": "seen_episode_census" if args.seen_episodes else "uniform_complete_training_episode_sample",
                    "seed": args.seed, "seen_through_update": args.updates if args.seen_episodes else None,
                    "pool_episodes": len(pool), "sampled_episodes": sorted(chosen),
                    "includes_validation_episodes": False}
    rows = [{**examples.dataset.rows[index], "used_for_sft": True} for index in selected]
    if args.source_decisions:
        if not complete_episodes:
            raise ValueError("Full source audit needs an episode sample or census, not a checkpoint prefix")
        rows = source_rows(args.source_decisions, examples.dataset.rows, selected)
        sampling["full_source_decisions_included"] = True
        sampling["source_decisions_sha256"] = __import__("hashlib").sha256(args.source_decisions.read_bytes()).hexdigest()
    stores, pairs, counts, tokens, episode_counts = {}, {}, Counter(), Counter(), Counter()
    effects, effect_tokens, duplicate_images = Counter(), Counter(), 0
    excluded_effects = defaultdict(Counter)
    tool_names = defaultdict(Counter)
    components, generation_stages, details = [], Counter(), []
    per_episode = defaultdict(Counter)
    for position, row in enumerate(rows):
        root = row["snapshot_root"]
        if root not in stores:
            stores[root] = open_store(root)
            pairs[root] = snapshot_pairs(root)
        reply = stores[root].get(row["response_ref"])["choices"][0]["message"]
        kind = category(reply)
        counts[kind] += 1
        tokens[kind] += row["completion_tokens"]
        tool_names[kind].update(call["function"]["name"] for call in reply.get("tool_calls", []))
        effect, duplicates, has_xml = artifact_effect(stores[root], pairs[root], row["call_id"])
        effects[effect] += 1
        effect_tokens[effect] += row["completion_tokens"]
        if not row["used_for_sft"]:
            excluded_effects[row["excluded_reason"]][effect] += 1
        duplicate_images += duplicates
        per_episode[row["episode"]][kind] += 1
        episode_counts[row["episode"]] += 1
        components.append(row["final_components"])
        generation_stages[row["stage"]] += 1
        details.append({"exposure": position, "sample_id": row["sample_id"], "category": kind,
                        "target_tokens": row["completion_tokens"], "generation_stage": row["stage"],
                        "artifact_effect": effect, "duplicate_images_written": duplicates,
                        "used_for_sft": row["used_for_sft"],
                        "xml_present_before": has_xml,
                        "reasoning_characters": len(reply.get("reasoning_content") or ""),
                        "pixel_inspection": effect == "no_case_file_change" and any(
                            marker in str(reply.get("tool_calls", [])) for marker in
                            ("Image.open", "np.array", "np.where", "cv2.", "getpixel", "ImageFont", "ndimage"))})
        if (position + 1) % 200 == 0:
            print(json.dumps({"audited": position + 1, "total": len(rows)}), flush=True)
    total_tokens = sum(tokens.values())
    report = {"at": time.time(), "sampling": sampling, "decisions_audited": len(rows),
        "decisions_used_for_sft": sum(row["used_for_sft"] for row in rows),
        "excluded_decision_effects": {reason: dict(values) for reason, values in excluded_effects.items()},
        "updates": args.updates if not complete_episodes else None,
        "exposures": len(selected) if not complete_episodes else None,
        "unique_samples": len({row["sample_id"] for row in details}), "unique_episodes": len(episode_counts),
        "curriculum_order_sha256": expected, "actual_training_order_verified": True,
        "components_range": [min(components), max(components)],
        "generation_stages": dict(generation_stages), "supervised_target_tokens": total_tokens,
        "categories": {name: {"decisions": counts[name], "decision_fraction": counts[name] / len(rows),
                              "target_tokens": tokens[name], "target_token_fraction": tokens[name] / total_tokens}
                       for name in sorted(counts)},
        "tool_names_by_category": {name: dict(values) for name, values in tool_names.items()},
        "artifact_effects": {name: {"decisions": effects[name], "decision_fraction": effects[name] / len(rows),
            "target_tokens": effect_tokens[name], "target_token_fraction": effect_tokens[name] / total_tokens}
            for name in sorted(effects)},
        "byte_identical_image_files_written_again": duplicate_images,
        "token_scope": "Target-token totals cover only the accepted SFT decisions. Rejected source decisions are included in process counts but contribute zero to actual training supervision.",
        "artifact_method": "Compare the case snapshot at this request with the next recorded generator request in the same episode and generation stage, including failed requests to avoid crossing unobserved actions. Missing next snapshots stay unknown. No-case-file-change may still acquire information and is not automatically a bad step. Exact SHA image duplication is a conservative lower bound.",
        "definition": "Mutually exclusive: XML-related argument markers first; then .crop calls; then read_image; other tools; assistant-only. XML-related includes inspecting/rendering, not only XML writes. Token totals include each selected decision's full reasoning and completion, not just crop code.",
        "limitations": ["String-based semantic categories; mixed calls assigned by the stated priority",
                        "Exposure distribution is evidence of supervision, not a causal explanation of one rollout"],
        "per_episode": {episode: {"exposures": episode_counts[episode], "categories": dict(kinds)}
                        for episode, kinds in sorted(per_episode.items())}}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    if complete_episodes:
        (args.output / "trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trajectory_summaries(details)))
    print(json.dumps({key: value for key, value in report.items() if key not in ("per_episode", "sampling")}), flush=True)


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=35)
    parser.add_argument("--world-size", type=int, default=4)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--episodes", type=int, default=0)
    sampling.add_argument("--seen-episodes", action="store_true")
    parser.add_argument("--source-decisions", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
