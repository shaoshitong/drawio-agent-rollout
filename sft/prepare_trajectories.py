"""Build a paper-disjoint, simple-to-complex curriculum of teacher decisions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

from drawio.utils.xml_models import models

if __package__:
    from .data import materialize, open_store, token_sha256
    from .history import HISTORY_POLICY
else:
    from data import materialize, open_store, token_sha256
    from history import HISTORY_POLICY


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def final_artifact(root, record):
    folder = root / record["extract_dir"]
    run = folder / "run-01"
    episode = run / "episodes" / record["episode"]
    passed = json.loads((folder / "audits" / record["episode"] / "passed.json").read_text())
    completed = json.loads((episode / "completed.json").read_text())
    protocol = json.loads((episode / "protocol.json").read_text())
    if not completed["completed"] or completed["id"] != record["episode"] or passed["episode"] != record["episode"]:
        raise ValueError("Episode lacks a matching positive completion and integrity audit")
    if not completed["decision_export"]["with_reasoning"]:
        raise ValueError("Episode has no recorded teacher reasoning")
    if any(protocol.get(key) is not False for key in ("gold_available", "gold_feedback", "mcp")):
        raise ValueError("Teacher protocol does not establish absence of privileged gold feedback")
    if digest(episode / "case/reference.png") != record["reference_sha256"]:
        raise ValueError("Teacher input image differs from the frozen release")
    xml, png = episode / "case/round2.drawio", episode / "case/round2.png"
    if digest(xml) != record["final_xml_sha256"] or digest(png) != record["final_png_sha256"]:
        raise ValueError("Final artifacts differ from the frozen release")
    render = json.loads((episode / "render2.json").read_text())
    if (
        render["source_sha256"] != record["final_xml_sha256"]
        or render["png_sha256"] != record["final_png_sha256"]
        or render.get("unresolved")
        or not render["renderer"].startswith("official_drawio")
    ):
        raise ValueError("Final official render receipt does not match the artifact")
    _, pages = models(xml.read_bytes())
    cells = [cell for page in pages for cell in page.iter("mxCell")]
    vertices = [cell for cell in cells if cell.get("vertex") == "1"]
    edges = sum(cell.get("edge") == "1" for cell in cells)
    image_cells = sum("shape=image" in cell.get("style", "") or "image=" in cell.get("style", "")
                      for cell in vertices)
    if not vertices or (image_cells and image_cells == len(vertices)):
        raise ValueError("Final diagram has no editable non-image vertices")
    return {
        "snapshot_root": str(run.resolve()),
        "final_components": len(vertices) + edges,
        "final_vertices": len(vertices),
        "final_edges": edges,
        "final_image_cells": image_cells,
        "audited_decisions": passed["generator_decisions"],
        "archived_final_render_verified": True,
        "fidelity_verified": False,
    }


def paper_split(records, seed, fraction):
    papers = sorted({re.sub(r"v\d+$", "", row["paper"]) for row in records})
    parents, references = {paper: paper for paper in papers}, {}

    def group(paper):
        while parents[paper] != paper:
            paper = parents[paper]
        return paper

    for row in records:
        paper, image = re.sub(r"v\d+$", "", row["paper"]), row.get("reference_sha256")
        if image in references:
            parents[group(paper)] = group(references[image])
        if image:
            references[image] = paper
    groups = {group(paper) for paper in papers}
    ordered = sorted(groups, key=lambda paper: sha256(f"{seed}:{paper}".encode()).hexdigest())
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction))) if len(groups) > 1 else 0
    validation = set(ordered[:count])
    return {paper: "validation" if group(paper) in validation else "train" for paper in papers}


def initialize_worker(processor_path, max_tokens, thinking):
    global PROCESSOR, MAX_TOKENS, THINKING, STORES, REWRITES
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    PROCESSOR = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    MAX_TOKENS, THINKING = max_tokens, thinking
    STORES, REWRITES = {}, {}


def measure(row):
    try:
        root = row["snapshot_root"]
        if root not in STORES:
            STORES[root] = open_store(root)
        rewrite = row.get("reasoning_rewrite")
        if rewrite and rewrite["root"] not in REWRITES:
            REWRITES[rewrite["root"]] = open_store(rewrite["root"])
        sample = materialize(row, Path("/"), STORES[root], REWRITES[rewrite["root"]] if rewrite else None)
        message = sample["completion"][0]
        reasoning = bool((message.get("reasoning_content") or "").strip())
        kwargs = {**sample["chat_template_kwargs"], "enable_thinking": THINKING, "tokenize": False}
        prefix = PROCESSOR.apply_chat_template(sample["prompt"], add_generation_prompt=True, **kwargs)
        full = PROCESSOR.apply_chat_template(sample["prompt"] + sample["completion"],
                                             add_generation_prompt=False, **kwargs)
        if not full.startswith(prefix):
            raise ValueError("Current completion changes its serving prompt prefix")
        completion = full[len(prefix):]
        target_tokens = len(PROCESSOR.tokenizer(completion, add_special_tokens=False)["input_ids"])
        if not target_tokens:
            raise ValueError("Current completion has no target tokens")
        if reasoning and (message["reasoning_content"].strip() not in completion):
            raise ValueError("Student chat template discarded recorded teacher reasoning")
        text_count = len(PROCESSOR.tokenizer(prefix, add_special_tokens=False)["input_ids"]) + target_tokens
        if text_count > MAX_TOKENS:
            return {"sample_id": row["sample_id"], "rejection": "text_exceeds_context",
                    "text_tokens": text_count, "completion_tokens": target_tokens}
        inputs = {"text": [prefix], "truncation": False, "padding": False,
                  "add_special_tokens": False, "return_tensors": "pt"}
        if sample["images"]:
            inputs["images"] = sample["images"]
        encoded = PROCESSOR(**inputs)
        encoded_completion = PROCESSOR(text=[completion], truncation=False, padding=False,
                                       add_special_tokens=False, return_tensors="pt")
        target_tokens = encoded_completion["input_ids"].shape[-1]
        prompt_tokens = encoded["input_ids"].shape[-1]
        if row.get("state_prefix_input_tokens", prompt_tokens) != prompt_tokens:
            raise ValueError("State prefix budget differs from native multimodal processing")
        total = prompt_tokens + target_tokens
        result = {
            "sample_id": row["sample_id"], "token_count": total,
            "completion_tokens": target_tokens, "prompt_tokens": prompt_tokens,
            "image_count": len(sample["images"]), "has_reasoning": reasoning,
            "tool_calls": len(message.get("tool_calls", [])), "has_content": bool(message.get("content")),
            "tokenization": "unsloth_pc_split",
            "prompt_token_sha256": token_sha256(encoded["input_ids"][0]),
        }
        if total > MAX_TOKENS:
            result["rejection"] = "multimodal_sequence_exceeds_context"
        return result
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"sample_id": row["sample_id"], "rejection": type(error).__name__, "error": str(error)}


def curriculum(args):
    records = read_jsonl(args.episodes)
    split = paper_split(records, args.seed, args.validation_fraction)
    rewrites = None
    if getattr(args, "rewrites", None):
        rewrites = {row["sample_id"]: row for row in read_jsonl(args.rewrites)}
    decisions, duplicates = {}, 0
    for row in read_jsonl(args.decisions):
        ident = row.get("decision_id", f"{row['episode']}:{row['call_id']}")
        if ident in decisions:
            previous = decisions[ident]
            if any(previous[key] != row[key] for key in ("episode", "request_ref", "response_ref")):
                raise ValueError("Conflicting duplicate decision: " + ident)
            duplicates += 1
        decisions[ident] = {**row, "sample_id": ident}
    by_episode = defaultdict(list)
    for row in decisions.values():
        by_episode[row["episode"]].append(row)
    ready, rejections = [], []
    for record in records:
        try:
            meta = final_artifact(args.root, record)
            rows = by_episode[record["episode"]]
            if len(rows) != meta["audited_decisions"]:
                raise ValueError("Global decision index does not cover the audited episode")
            paper = re.sub(r"v\d+$", "", record["paper"])
            ready.append(({**record, **meta, "paper_group": paper, "split": split[paper]}, rows))
        except (ValueError, KeyError, OSError) as error:
            rejections.append({"episode": record["episode"], "rejection": type(error).__name__,
                               "error": str(error)})
    ready.sort(key=lambda item: (item[0]["final_components"], item[0]["episode"]))
    selected, counters = [], Counter()
    for index, (episode, rows) in enumerate(ready):
        limit = args.max_train_episodes if episode["split"] == "train" else args.max_validation_episodes
        if limit and counters[episode["split"]] >= limit:
            continue
        counters[episode["split"]] += 1
        for order, row in enumerate(sorted(rows, key=lambda value: value["call_id"])):
            if not row.get("supervision_eligible", True):
                rejections.append({"sample_id": row["sample_id"], "rejection": "protocol_ineligible"})
                continue
            if rewrites is not None:
                rewritten = rewrites[row["sample_id"]]
                if any(rewritten[key] != row[key] for key in ("request_ref", "response_ref")):
                    raise ValueError("Rewrite source references changed: " + row["sample_id"])
                if rewritten.get("state_rejection"):
                    rejections.append({"sample_id": row["sample_id"], **rewritten["state_rejection"]})
                    continue
                row = {**row, **{key: rewritten[key] for key in
                    ("reasoning_rewrite", "state_prefix_input_tokens", "state_prefix_policy") if key in rewritten}}
            if row.get("data_part") != episode["data_part"]:
                raise ValueError("Decision and episode refer to different data parts")
            selected.append({
                "schema": "drawio-sft/1", "kind": "teacher_decision",
                **{key: row[key] for key in ("sample_id", "episode", "call_id", "stage",
                                            "request_ref", "response_ref")},
                **{key: row[key] for key in ("reasoning_rewrite", "state_prefix_input_tokens", "state_prefix_policy") if key in row},
                **{key: episode[key] for key in ("snapshot_root", "paper_group", "split",
                    "final_components", "final_vertices", "final_edges", "final_image_cells",
                    "archived_final_render_verified", "fidelity_verified", "data_part")},
                "episode_order": index, "decision_order": order, "supervision_eligible": True,
                "history_policy": HISTORY_POLICY,
                "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
            })
    return selected, rejections, {"source_episodes": len(records), "source_unique_decisions": len(decisions),
                                 "duplicate_index_rows": duplicates, "selected_episodes": dict(counters)}


def checkpoint_records(path):
    """Keep complete JSONL records; an interrupted final write is not measured."""
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(stream.tell() - len(line))
                break
            records.append(json.loads(line))
    return records


def resume_checkpoint(rows, initial_rejections, pending, rejects_path):
    accepted = checkpoint_records(pending)
    rejected = checkpoint_records(rejects_path)
    selected = {row["sample_id"]: row for row in rows}
    fixed = {json.dumps(row, sort_keys=True): row for row in initial_rejections}
    seen_fixed, measured, seen = set(), [], set()
    for row in accepted:
        ident = row["sample_id"]
        if ident in seen or ident not in selected or any(row.get(key) != value for key, value in selected[ident].items()):
            raise ValueError("Accepted checkpoint identity differs from curriculum: " + ident)
        if row.get("tokenization") != "unsloth_pc_split" or not row.get("prompt_token_sha256"):
            raise ValueError("Accepted checkpoint lacks the current token policy: " + ident)
        if any(key not in row for key in ("token_count", "prompt_tokens", "completion_tokens", "has_reasoning", "image_count")):
            raise ValueError("Accepted checkpoint has an incomplete measurement: " + ident)
        seen.add(ident)
    for row in rejected:
        signature = json.dumps(row, sort_keys=True)
        if signature in fixed:
            if signature in seen_fixed:
                raise ValueError("Duplicate initial rejection in checkpoint")
            seen_fixed.add(signature)
            continue
        ident = row.get("sample_id")
        if ident in seen or ident not in selected:
            raise ValueError("Rejected checkpoint identity differs from curriculum: " + str(ident))
        if any(key in row and row[key] != selected[ident][key]
               for key in ("episode", "call_id", "request_ref", "response_ref")):
            raise ValueError("Rejected checkpoint call differs from curriculum: " + ident)
        seen.add(ident)
        measured.append(row)
    prefix = rows[:len(seen)]
    if {row["sample_id"] for row in prefix} != seen:
        raise ValueError("Measured checkpoint is not a contiguous curriculum prefix")
    measured_ids = {row["sample_id"] for row in measured}
    expected = [row["sample_id"] for row in prefix if row["sample_id"] not in measured_ids]
    if [row["sample_id"] for row in accepted] != expected:
        raise ValueError("Accepted checkpoint order differs from curriculum")
    missing = [value for key, value in fixed.items() if key not in seen_fixed]
    return accepted, measured, missing, len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "episodes", "decisions", "processor", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-validation-episodes", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--resume-preparation", action="store_true")
    parser.add_argument("--rewrites", type=Path)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.max_tokens < 1:
        parser.error("validation-fraction must be between 0 and 1; max-tokens must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    rows, rejected, summary = curriculum(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = args.output.with_suffix(".rejected.jsonl")
    pending = args.output.with_suffix(".building.jsonl")
    state_path = args.output.with_suffix(".preparation.json")
    processor_files = {path.name: digest(path) for path in args.processor.iterdir()
                       if path.suffix in (".json", ".jinja")}
    policy = {
        "source_episodes_sha256": digest(args.episodes), "source_decisions_sha256": digest(args.decisions),
        "curriculum_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "processor": str(args.processor.resolve()), "processor_files_sha256": processor_files,
        "processor_versions": {name: version(name) for name in ("transformers", "tokenizers", "Pillow", "torch")},
        "max_tokens": args.max_tokens, "tokenization": "unsloth_pc_split",
        "processor_image_settings": "unchanged model defaults",
    }
    if args.rewrites:
        policy["rewrites_sha256"] = digest(args.rewrites)
    previous, prior_accepted, prior_rejected, missing, resumed = {}, [], [], rejected, 0
    if args.resume_preparation:
        if not pending.exists():
            raise FileNotFoundError("No building manifest to resume: " + str(pending))
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous and previous["policy"] != policy:
            raise ValueError("Preparation source, processor or token policy changed")
        prior_accepted, prior_rejected, missing, resumed = resume_checkpoint(rows, rejected, pending, rejects_path)
    elif any(path.exists() for path in (pending, rejects_path, state_path, args.output)):
        raise FileExistsError("Existing preparation output; choose --resume-preparation or a new output")
    code_sha = digest(Path(__file__))
    history = previous.get("history", []) + [{"code_sha256": code_sha, "resumed_decisions": resumed,
        "legacy_checkpoint_adopted": args.resume_preparation and not previous, "at": time.time()}]
    state = {"schema": "drawio-sft-preparation/1", "policy": policy, "history": history}
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(state_path)
    counts, accepted, reasoning_counts, episode_counts = Counter(), Counter(), Counter(), defaultdict(Counter)
    counts.update(row["rejection"] for row in rejected)
    for row in prior_accepted:
        accepted[row["split"]] += 1
        reasoning_counts["with_reasoning" if row["has_reasoning"] else "without_reasoning"] += 1
        episode_counts[row["episode"]]["accepted"] += 1
    by_id = {row["sample_id"]: row for row in rows}
    for row in prior_rejected:
        counts[row["rejection"]] += 1
        episode_counts[by_id[row["sample_id"]]["episode"]][row["rejection"]] += 1
    initializer = (str(args.processor), args.max_tokens, not args.no_thinking)
    started = time.monotonic()
    remaining = rows[resumed:]
    mode = "a" if args.resume_preparation else "w"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=initializer) as pool, pending.open(mode) as output, rejects_path.open(mode) as reject:
        for item in missing:
            reject.write(json.dumps(item) + "\n")
        for index, (row, result) in enumerate(zip(remaining, pool.map(measure, remaining, chunksize=1)), resumed + 1):
            if result.get("rejection"):
                identity = {key: row[key] for key in ("episode", "call_id", "request_ref", "response_ref")}
                reject.write(json.dumps({**identity, **result}) + "\n")
                counts[result["rejection"]] += 1
                episode_counts[row["episode"]][result["rejection"]] += 1
            else:
                output.write(json.dumps({**row, **result}) + "\n")
                accepted[row["split"]] += 1
                reasoning_counts["with_reasoning" if result["has_reasoning"] else "without_reasoning"] += 1
                episode_counts[row["episode"]]["accepted"] += 1
            if index % 25 == 0 or index == len(rows):
                output.flush()
                reject.flush()
                os.fsync(output.fileno())
                os.fsync(reject.fileno())
                print(json.dumps({"measured": index, "selected": len(rows), "accepted": dict(accepted),
                                  "rejected": dict(counts), "elapsed_seconds": time.monotonic() - started,
                                  "resumed_decisions": resumed,
                                  "decisions_per_minute": (index - resumed) * 60 / (time.monotonic() - started)}), flush=True)
    pending.replace(args.output)
    summary.update({
        "accepted_decisions": dict(accepted), "rejections": dict(counts), "measured_decisions": len(rows),
        "manifest_sha256": digest(args.output), "source_episodes_sha256": digest(args.episodes),
        "source_decisions_sha256": digest(args.decisions), "processor": str(args.processor.resolve()),
        "processor_files_sha256": processor_files, "preparation_code_sha256": code_sha,
        "preparation_history": history, "resumed_decisions": resumed,
        "max_tokens": args.max_tokens, "reasoning_decisions": dict(reasoning_counts),
        "episode_decision_retention": {key: dict(value) for key, value in episode_counts.items()},
        "fully_retained_eligible_episodes": sum(set(value) == {"accepted"} for value in episode_counts.values()),
        "partially_retained_episodes": sum(value["accepted"] > 0 and len(value) > 1
                                          for value in episode_counts.values()),
        "reasoning_policy": "Preserve recorded reasoning; retain valid tool-only and final decisions without it",
        "tokenization": "unsloth_pc_split",
        "tokenization_details": "Actual processor prompt with images plus separately encoded current completion",
        "history_conditioning": "Recorded per-call messages rendered through the native student chat template",
        "split": "arxiv_paper_without_version_and_duplicate_reference_image",
        "curriculum": "final_vertices_plus_edges_then_original_call_id",
        "images_resized_by_preparation": False, "processor_image_settings": "unchanged model defaults",
        "fidelity_verified": False, "semantic_quality": "unknown_without_matched_gold_or_independent_fidelity_audit",
        "targets_truncated": False, "gold_pairs_used": False, "training_run": False,
    })
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not accepted["train"]:
        raise ValueError("No training decisions passed preparation")


if __name__ == "__main__":
    main()
