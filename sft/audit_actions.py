"""Describe the actual supervised actions seen by a curriculum checkpoint."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
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


def audit(args):
    config = json.loads(args.config.read_text())
    examples = Examples(args.manifest, "train", True)
    order, plan = curriculum_order(examples, config, args.world_size)
    expected = json.loads(args.plan.read_text())["source_indices_sha256"]
    if plan["source_indices_sha256"] != expected:
        raise ValueError("Reconstructed order differs from the actual training sampler")
    limit = args.updates * args.world_size * config["batch_size"] * config["gradient_accumulation_steps"]
    selected = order[:limit]
    stores, counts, tokens, episode_counts = {}, Counter(), Counter(), Counter()
    components, generation_stages, details = [], Counter(), []
    per_episode = defaultdict(Counter)
    for position, index in enumerate(selected):
        row = examples.dataset.rows[index]
        root = row["snapshot_root"]
        if root not in stores:
            stores[root] = open_store(root)
        reply = stores[root].get(row["response_ref"])["choices"][0]["message"]
        kind = category(reply)
        counts[kind] += 1
        tokens[kind] += row["completion_tokens"]
        per_episode[row["episode"]][kind] += 1
        episode_counts[row["episode"]] += 1
        components.append(row["final_components"])
        generation_stages[row["stage"]] += 1
        details.append({"exposure": position, "sample_id": row["sample_id"], "category": kind,
                        "target_tokens": row["completion_tokens"], "generation_stage": row["stage"]})
        if (position + 1) % 200 == 0:
            print(json.dumps({"audited": position + 1, "total": len(selected)}), flush=True)
    total_tokens = sum(tokens.values())
    report = {"at": time.time(), "updates": args.updates, "exposures": len(selected),
        "unique_samples": len({row["sample_id"] for row in details}), "unique_episodes": len(episode_counts),
        "curriculum_order_sha256": expected, "actual_training_order_verified": True,
        "components_range": [min(components), max(components)],
        "generation_stages": dict(generation_stages), "supervised_target_tokens": total_tokens,
        "categories": {name: {"decisions": counts[name], "decision_fraction": counts[name] / len(selected),
                              "target_tokens": tokens[name], "target_token_fraction": tokens[name] / total_tokens}
                       for name in sorted(counts)},
        "definition": "Mutually exclusive: XML-related argument markers first; then .crop calls; then read_image; other tools; assistant-only. XML-related includes inspecting/rendering, not only XML writes. Token totals include each selected decision's full reasoning and completion, not just crop code.",
        "limitations": ["String-based semantic categories; mixed calls assigned by the stated priority",
                        "Exposure distribution is evidence of supervision, not a causal explanation of one rollout"],
        "per_episode": {episode: {"exposures": episode_counts[episode], "categories": dict(kinds)}
                        for episode, kinds in sorted(per_episode.items())}}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    print(json.dumps({key: value for key, value in report.items() if key != "per_episode"}), flush=True)


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=35)
    parser.add_argument("--world-size", type=int, default=4)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
