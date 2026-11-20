"""CPU-only sampled native-prefix proofs; never changes training inputs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

from .check_prefix_chains import Reader
from .data import ManifestDataset, token_sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metadata(manifest, output, cap, samples):
    rows = ManifestDataset(manifest).rows
    episodes, readers = defaultdict(list), {}
    for index, row in enumerate(rows):
        episodes[row["episode"]].append(index)
    chains, pairs, counts = [], [], Counter()
    for number, indices in enumerate(episodes.values(), 1):
        indices.sort(key=lambda index: rows[index]["call_id"])
        previous = None
        chain = []
        for index in indices:
            row = rows[index]
            root = str(manifest.parent / row["snapshot_root"])
            if root not in readers:
                readers[root] = Reader(root)
            reader = readers[root]
            current = reader.context(row)
            if previous is not None:
                old_index, old, response = previous
                length = len(old["refs"])
                prefix = old["refs"] == current["refs"][:length]
                if not prefix:
                    prefix = tuple(reader.message(ref)["digest"] for ref in old["refs"]) == \
                        tuple(reader.message(ref)["digest"] for ref in current["refs"][:length])
                images = old["images"] == current["images"][:len(old["images"])]
                appended = len(current["refs"]) > length and \
                    reader.message(current["refs"][length])["digest"] == response
                same_user = old["last_user"] == current["last_user"]
                candidate = prefix and images and appended and old["tools"] == current["tools"]
                added_images = len(current["images"]) > len(old["images"])
                strict = candidate and same_user and old["projected"] == current["projected"][:length]
                counts["links"] += 1
                counts["strict_links"] += strict
                counts["strict_appended_image_links"] += strict and added_images
                counts["append_image_new_user_controls"] += candidate and added_images and not same_user
                if candidate:
                    pairs.append({"early": old_index, "late": index, "strict": strict,
                                  "added_images": added_images, "same_last_user": same_user,
                                  "generation_stage": row["stage"], "length": row["token_count"]})
                if not strict or len(chain) == cap:
                    chains.append(chain)
                    chain = []
            chain.append(index)
            previous = index, current, reader.completion(row)
        if chain:
            chains.append(chain)
        if number % 25 == 0:
            print(json.dumps({"metadata_episodes": number, "decisions": sum(map(len, chains)),
                              "strict_links": counts["strict_links"]}), flush=True)
    original = sum(row["token_count"] for row in rows)
    retained = sum(rows[chain[-1]]["token_count"] for chain in chains)
    chosen = []
    for strict, quota in ((True, samples * 3 // 4), (False, samples - samples * 3 // 4)):
        buckets = defaultdict(list)
        for pair in pairs:
            if pair["strict"] == strict and (strict or pair["added_images"]):
                buckets[(pair["added_images"], pair["generation_stage"], pair["length"] // 65536)].append(pair)
        for bucket in buckets.values():
            bucket.sort(key=lambda pair: pair["length"])
        while buckets and sum(pair["strict"] == strict for pair in chosen) < quota:
            for key in sorted(list(buckets)):
                bucket = buckets[key]
                chosen.append(bucket.pop(len(bucket) // 2))
                if not bucket:
                    del buckets[key]
                if sum(pair["strict"] == strict for pair in chosen) >= quota:
                    break
    report = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "decisions": len(rows), "episodes": len(episodes), "chain_cap": cap,
              "candidate_chains": len(chains), "max_chain": max(map(len, chains)),
              "chain_histogram": dict(Counter(map(len, chains))), "counts": dict(counts),
              "original_input_tokens": original, "candidate_chain_end_tokens": retained,
              "candidate_reduction_factor": original / retained, "selected_pairs": chosen,
              "scope": "Metadata candidates only; new-user controls are never eligible merges"}
    save(output, report)
    print(json.dumps(report), flush=True)
    return report


def initialize(manifest, model):
    global DATA, PROCESSOR, CONFIG, ROPE, TORCH
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only checker requires CUDA_VISIBLE_DEVICES=''")
    os.nice(5)
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    torch.set_num_threads(1)
    TORCH, DATA = torch, ManifestDataset(manifest)
    PROCESSOR = AutoProcessor.from_pretrained(model, local_files_only=True)
    CONFIG = AutoConfig.from_pretrained(model, local_files_only=True)
    stub = SimpleNamespace(config=CONFIG)
    stub.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, stub)
    ROPE = MethodType(Qwen3_5Model.get_rope_index, stub)


def encode(index):
    sample, row = DATA[index], DATA.rows[index]
    kwargs = {"enable_thinking": True, **sample["chat_template_kwargs"]}
    prompt = PROCESSOR.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True, **kwargs)
    full = PROCESSOR.apply_chat_template(sample["prompt"] + sample["completion"], tokenize=False,
                                         add_generation_prompt=False, **kwargs)
    if not full.startswith(prompt):
        raise ValueError("Native full rendering changes its generation prefix")
    inputs = {"text": [prompt], "add_special_tokens": False, "padding": False,
              "truncation": False, "return_tensors": "pt"}
    if sample["images"]:
        inputs["images"] = sample["images"]
    prefix = PROCESSOR(**inputs)
    completion = PROCESSOR(text=[full[len(prompt):]], add_special_tokens=False,
                           padding=False, truncation=False, return_tensors="pt")
    ids = TORCH.cat((prefix["input_ids"], completion["input_ids"]), dim=1)
    types = TORCH.zeros_like(ids)
    types[ids == CONFIG.image_token_id] = 1
    types[ids == CONFIG.video_token_id] = 2
    positions, _ = ROPE(ids, types, image_grid_thw=prefix.get("image_grid_thw"),
                        attention_mask=TORCH.ones_like(ids))
    return {"ids": ids[0], "prompt_tokens": prefix["input_ids"].shape[1],
            "prompt_hash_ok": token_sha256(prefix["input_ids"][0]) == row["prompt_token_sha256"],
            "length_ok": ids.shape[1] == row["token_count"], "full_text": full,
            "pixels": prefix.get("pixel_values"), "grid": prefix.get("image_grid_thw"),
            "positions": positions, "sample_id": sample["sample_id"]}


def prefix_tensor(early, late):
    if early is None:
        return True
    return late is not None and late.shape[0] >= early.shape[0] and TORCH.equal(early, late[:early.shape[0]])


def prove(pair):
    started = time.monotonic()
    result = {**pair, "status": "rejected"}
    try:
        early, late = encode(pair["early"]), encode(pair["late"])
        n = len(early["ids"])
        token_prefix = len(late["ids"]) >= n and TORCH.equal(early["ids"], late["ids"][:n])
        spans = (early["prompt_tokens"], n, late["prompt_tokens"], len(late["ids"]))
        union = TORCH.full_like(late["ids"], -100)
        union_ok = token_prefix and spans[0] < spans[1] <= spans[2] < spans[3]
        if union_ok:
            union[spans[0]:spans[1]] = early["ids"][spans[0]:spans[1]]
            union[spans[2]:spans[3]] = late["ids"][spans[2]:spans[3]]
            union_ok = int(union.ne(-100).sum()) == spans[1] - spans[0] + spans[3] - spans[2]
        old_row, new_row = DATA.rows[pair["early"]], DATA.rows[pair["late"]]
        checks = {"same_episode": old_row["episode"] == new_row["episode"],
                  "same_split": old_row["split"] == new_row["split"],
                  "same_final_components": old_row["final_components"] == new_row["final_components"],
                  "same_last_real_user": pair["same_last_user"], "exact_pc_token_prefix": token_prefix,
                  "manifest_prompt_hashes": early["prompt_hash_ok"] and late["prompt_hash_ok"],
                  "manifest_lengths": early["length_ok"] and late["length_ok"],
                  "target_spans_disjoint_unique": union_ok,
                  "pixel_prefix": prefix_tensor(early["pixels"], late["pixels"]),
                  "image_grid_prefix": prefix_tensor(early["grid"], late["grid"]),
                  "official_mrope_prefix": late["positions"].shape[-1] >= n and
                  TORCH.equal(early["positions"], late["positions"][..., :n])}
        result.update(status="passed" if all(checks.values()) else "rejected", checks=checks,
                      early_id=early["sample_id"], late_id=late["sample_id"], target_spans=spans,
                      character_prefix=late["full_text"].startswith(early["full_text"]))
        if not token_prefix and len(late["ids"]) >= n:
            result["first_token_mismatch"] = int((early["ids"] != late["ids"][:n]).nonzero()[0])
    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--cap", type=int, default=8)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      TOKENIZERS_PARALLELISM="false")
    os.nice(5)
    started = time.monotonic()
    plan = metadata(args.manifest, args.output.with_suffix(".metadata.json"), args.cap, args.pairs)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.manifest), str(args.model))) as pool:
        futures = [pool.submit(prove, pair) for pair in plan["selected_pairs"]]
        for future in as_completed(futures):
            results.append(future.result())
            save(args.output, {"scope": "Sampled CPU processor/PC token, label union, pixels and official weight-free mRoPE",
                               "metadata": plan, "results": results, "pairs_completed": len(results),
                               "statuses": dict(Counter(row["status"] for row in results)),
                               "elapsed_seconds": time.monotonic() - started, "no_training_changes": True,
                               "label_path": "Reconstructed native PC completion-only labels; no collator/model weights loaded",
                               "mrope_path": "Official get_rope_index/get_vision_position_ids on a config-only object"})
            print(json.dumps({"pair_complete": len(results), "result": results[-1]}), flush=True)


if __name__ == "__main__":
    main()
