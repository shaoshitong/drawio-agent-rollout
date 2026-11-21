"""Compare true PEFT and merged Qwen checkpoints on unchanged masked validation targets."""

import argparse
from bisect import bisect_right
import faulthandler
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

from .data import token_sha256
from .train_unsloth import use_linear_patch_embedding


def native_prompt(processor, row, directory):
    from PIL import Image

    images = [Image.open(directory / name).convert("RGB") for name in row["images"]]
    kwargs = {"images": images} if images else {}
    batch = processor(text=[row["prompt"]], add_special_tokens=False, return_tensors="pt", **kwargs)
    if token_sha256(batch["input_ids"][0]) != row["prompt_token_sha256"]:
        raise ValueError(f"{row['sample_id']}: native prompt token IDs changed")
    return batch


def native_batch(processor, row, directory):
    import torch

    batch = native_prompt(processor, row, directory)
    prompt = batch["input_ids"]
    target = processor(text=[row["completion"]], add_special_tokens=False, return_tensors="pt")["input_ids"]
    if prompt.shape[1] + target.shape[1] != row["sequence_tokens"] or target.shape[1] != row["target_tokens"]:
        raise ValueError(f"{row['sample_id']}: native sequence or loss-mask length changed")
    batch["input_ids"] = torch.cat((prompt, target), dim=1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in batch:
            batch[name] = torch.cat((batch[name], torch.zeros_like(target)), dim=1)
    return batch, target[0], prompt.shape[1]


def load_checkpoint(args):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.cuda.set_device(0)
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model or args.base, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    )
    if args.variant == "peft":
        # Dense BF16 adapters do not use optional quantization backends.
        sys.modules["torchao"] = None
        from peft import PeftModel, get_peft_model_state_dict

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        expected = load_file(args.adapter / "adapter_model.safetensors")
        loaded = get_peft_model_state_dict(model)
        if expected.keys() != loaded.keys() or any(not torch.equal(value.cpu(), loaded[key].cpu())
                                                   for key, value in expected.items()):
            raise ValueError("Not every saved LoRA tensor was loaded exactly")
    model.eval()
    use_linear_patch_embedding(model)
    root = model.get_base_model() if args.variant == "peft" else model
    return processor, model, root


def verify_saved_probe(path, row, target, prompt_tokens, width):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    state = load_file(path)
    if (metadata["sample_id"] != row["sample_id"]
            or int(metadata["prompt_tokens"]) != prompt_tokens
            or not torch.equal(state["target"], target)):
        raise ValueError("Saved probe does not match the unchanged validation decision")
    if state["hidden"].shape != (len(target), width) or not torch.isfinite(state["hidden"]).all():
        raise ValueError("Saved probe contains invalid prediction states")


def harvest(args):
    import torch
    from safetensors.torch import save_file

    processor, model, root = load_checkpoint(args)
    if args.layer_progress:
        for index, layer in enumerate(root.model.language_model.layers):
            layer.register_forward_pre_hook(lambda _module, _inputs, index=index: print(json.dumps(
                {"event": "layer_start", "variant": args.variant, "layer": index, "at": time.time()}), flush=True))
            layer.register_forward_hook(lambda _module, _inputs, _output, index=index: print(json.dumps(
                {"event": "layer_done", "variant": args.variant, "layer": index, "at": time.time()}), flush=True))
    directory = args.output / args.variant
    directory.mkdir(parents=True, exist_ok=bool(args.start_index))
    width = root.lm_head.weight.shape[1]
    head = root.lm_head.weight.detach().to(torch.bfloat16).contiguous().view(torch.uint8).cpu()
    digest = hashlib.sha256(head.numpy().tobytes()).hexdigest()
    head_path = directory / "head-sha256.txt"
    if args.start_index:
        if head_path.read_text().strip() != digest:
            raise ValueError("Resume head differs from the saved checkpoint head")
    else:
        head_path.write_text(digest + "\n")
    del head
    if args.offload_layers:
        from accelerate import cpu_offload_with_hook

        root.lm_head.to("cpu")
        for layer in root.model.language_model.layers:
            _, hook = cpu_offload_with_hook(layer, execution_device=torch.device("cuda:0"))
            layer.register_forward_hook(lambda _module, _inputs, _output, hook=hook: hook.offload())
        torch.cuda.empty_cache()
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(rows):
            path = directory / f"{index:03}.safetensors"
            if index < args.start_index:
                target = processor(text=[row["completion"]], add_special_tokens=False,
                                   return_tensors="pt")["input_ids"][0]
                verify_saved_probe(path, row, target, row["sequence_tokens"] - len(target), width)
                print(json.dumps({"event": "probe_reused", "sample_id": row["sample_id"],
                                  "index": index, "at": time.time()}), flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            batch, target, prompt_tokens = native_batch(processor, row, args.probes)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            print(json.dumps({"event": "probe_start", "variant": args.variant,
                              "sample_id": row["sample_id"], "sequence_tokens": row["sequence_tokens"],
                              "images": len(row["images"]), "at": time.time()}), flush=True)
            # Qwen's language backbone already applies its final RMSNorm.
            result = root.model(**batch, use_cache=False, return_dict=True)
            hidden = result.last_hidden_state[0, prompt_tokens - 1:-1].contiguous().cpu()
            if len(hidden) != len(target) or not torch.isfinite(hidden).all():
                raise ValueError("Masked prediction states differ from shifted target positions")
            temporary = path.with_suffix(".safetensors.tmp")
            save_file({"hidden": hidden, "target": target}, temporary,
                      metadata={"sample_id": row["sample_id"], "prompt_tokens": str(prompt_tokens)})
            temporary.replace(path)
            del result, hidden, batch
            progress = {"variant": args.variant, "completed_probes": index + 1, "total_probes": len(rows),
                        "last_sample_id": row["sample_id"], "at": time.time(),
                        "full_native_prefixes_verified": True, "training": False,
                        "decoder_weights_cpu_offloaded": args.offload_layers,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "released_cuda_allocated_bytes": torch.cuda.memory_allocated()}
            (directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)


def checkpoint_head(checkpoint):
    import torch
    from safetensors import safe_open

    path = checkpoint / "model.safetensors"
    if not path.is_file():
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        path = checkpoint / index["lm_head.weight"]
    with safe_open(path, framework="pt", device="cpu") as source:
        return source.get_tensor("lm_head.weight").to(torch.bfloat16).contiguous()


def scores(args):
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file

    head = checkpoint_head(args.base).to("cuda")
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    directories = {name: args.output / name for name in ("base", "peft", "merged")}
    if args.reference_base:
        directories["original_conv3d_base"] = args.reference_base
    variants = tuple(directories)
    heads = [(directory / "head-sha256.txt").read_text().strip() for directory in directories.values()]
    if len(set(heads)) != 1:
        raise ValueError("Shared-head parity applies to LoRA merges, not changed full-finetuning heads")
    labels = ("0-32K", "32-64K", "64-128K", "128K+")
    groups = {name: {"tokens": 0, "ce_sum": [0.0] * len(variants), "kl_sum": 0.0, "agreements": 0}
              for name in ("overall", *labels)}
    probe_groups = {}
    with torch.inference_mode():
        for position, row in enumerate(rows):
            active_groups = groups
            if args.per_probe:
                probe = {"tokens": 0, "ce_sum": [0.0] * len(variants), "kl_sum": 0.0, "agreements": 0}
                probe_groups[row["sample_id"]] = probe
                active_groups = {**groups, "_probe": probe}
            states = [load_file(directory / f"{position:03}.safetensors") for directory in directories.values()]
            target = states[0]["target"]
            if any(not torch.equal(target, state["target"]) for state in states[1:]):
                raise ValueError("Compared checkpoints have different supervised tokens")
            prompt_tokens = row["sequence_tokens"] - len(target)
            for start in range(0, len(target), 128):
                stop = min(start + 128, len(target))
                hidden = [state["hidden"][start:stop].to("cuda") for state in states]
                logp = [functional.log_softmax(functional.linear(value, head).float(), dim=-1) for value in hidden]
                ids = target[start:stop].to("cuda")
                ce = torch.stack([-value.gather(1, ids[:, None])[:, 0] for value in logp])
                kl = (logp[1].exp() * (logp[1] - logp[2])).sum(-1)
                same = logp[1].argmax(-1).eq(logp[2].argmax(-1))
                bins = torch.tensor([bisect_right((32768, 65536, 131072), prompt_tokens + offset)
                                     for offset in range(start, stop)], device="cuda")
                for name, group in active_groups.items():
                    mask = torch.ones_like(same) if name in ("overall", "_probe") else bins.eq(labels.index(name))
                    group["tokens"] += int(mask.sum())
                    group["ce_sum"] = [a + b for a, b in zip(group["ce_sum"], ce[:, mask].double().sum(1).tolist())]
                    group["kl_sum"] += float(kl[mask].double().sum())
                    group["agreements"] += int(same[mask].sum())
    metrics = {}
    for name, group in {**groups, **probe_groups}.items():
        count = group["tokens"]
        if count:
            ce = dict(zip(variants, [value / count for value in group["ce_sum"]]))
            metrics[name] = {"tokens": count, "ce": ce, "export_ce_gap": abs(ce["merged"] - ce["peft"]),
                             "lora_ce_gain": ce["base"] - ce["peft"],
                             "peft_to_merged_kl": group["kl_sum"] / count,
                             "peft_merged_argmax_agreement": group["agreements"] / count}
    overall = metrics["overall"]
    gate = (overall["export_ce_gap"] <= min(0.005, 0.05 * abs(overall["lora_ce_gain"]))
            and overall["peft_to_merged_kl"] <= 0.002 and overall["peft_merged_argmax_agreement"] >= 0.99)
    bin_gate = all(value["export_ce_gap"] <= 2 * overall["export_ce_gap"] for name, value in metrics.items()
                   if name in labels)
    report = {"metrics": {name: value for name, value in metrics.items() if name in groups},
              "overall_export_gate_passed": gate,
              "position_bin_gate_passed": bin_gate, "export_gate_passed": gate and bin_gate, "probes": len(rows),
              "scope": "HF true PEFT vs BF16 merged; serving parity and reconstruction assessed separately",
              "all_native_prefixes_preserved": True, "gold_feedback_to_dsh": False}
    if args.per_probe:
        report["per_probe"] = {name: metrics[name] for name in probe_groups}
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def score_loss(args):
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from safetensors.torch import load_file

    end_id = None
    if args.end_token_metrics:
        from transformers import AutoTokenizer

        end_id = AutoTokenizer.from_pretrained(args.base, local_files_only=True).convert_tokens_to_ids("<|im_end|>")
    checkpoint = args.model or args.base
    head = checkpoint_head(checkpoint)
    digest = hashlib.sha256(head.view(torch.uint8).numpy().tobytes()).hexdigest()
    directory = args.output / args.variant
    if digest != (directory / "head-sha256.txt").read_text().strip():
        raise ValueError("Loss head does not belong to the harvested checkpoint")
    head = head.to("cuda")
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    records, total_loss, total_tokens = [], 0.0, 0
    with torch.inference_mode():
        for position, row in enumerate(rows):
            path = directory / f"{position:03}.safetensors"
            with safe_open(path, framework="pt", device="cpu") as source:
                if source.metadata()["sample_id"] != row["sample_id"]:
                    raise ValueError("Harvested decision does not match the fixed probe")
            state = load_file(path)
            target, loss = state["target"], 0.0
            endings = []
            for start in range(0, len(target), 128):
                hidden = state["hidden"][start:start + 128].to("cuda")
                ids = target[start:start + 128].to("cuda")
                logp = functional.log_softmax(functional.linear(hidden, head).float(), dim=-1)
                loss += float(-logp.gather(1, ids[:, None])[:, 0].double().sum())
                if end_id is not None:
                    mask = ids.eq(end_id)
                    endings.extend({"position": start + int(position), "probability": float(logp[position, end_id].exp()),
                                    "top1": int(logp[position].argmax()) == end_id}
                                   for position in mask.nonzero().flatten())
            record = {"sample_id": row["sample_id"], "tokens": len(target), "ce": loss / len(target)}
            if end_id is not None:
                record["end_of_turn"] = endings
            records.append(record)
            total_loss += loss
            total_tokens += len(target)
    report = {"variant": args.variant, "checkpoint": str(checkpoint), "head_sha256": digest,
              "probe_manifest_sha256": hashlib.sha256((args.probes / "probes.jsonl").read_bytes()).hexdigest(),
              "probes": len(records), "target_tokens": total_tokens,
              "target_token_mean_ce": total_loss / total_tokens,
              "decision_macro_ce": sum(row["ce"] for row in records) / len(records),
              "per_probe": records, "all_native_prefixes_preserved": True,
              "scope": "Fixed masked targets with this checkpoint's own output head; no rollout quality claim"}
    if end_id is not None:
        report["end_token_scope"] = "End-of-turn probabilities at teacher-forced target positions; not free-generation stopping accuracy"
    (args.output / f"loss-{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_probe"}), flush=True)


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("harvest", "score", "loss"))
    for name in ("base", "probes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--variant", choices=("base", "peft", "merged", "full"))
    parser.add_argument("--layer-progress", action="store_true")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Resume the same experiment after validating earlier saved probes")
    parser.add_argument("--offload-layers", action="store_true",
                        help="Move decoder weights between CPU and CUDA without changing native prefixes")
    parser.add_argument("--per-probe", action="store_true", help="Also report each unchanged target decision separately")
    parser.add_argument("--end-token-metrics", action="store_true",
                        help="Report end-of-turn probabilities under unchanged teacher-forced prefixes")
    parser.add_argument("--reference-base", type=Path,
                        help="Also score preserved baseline states from the original Conv3d projection")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative")
    {"harvest": harvest, "score": scores, "loss": score_loss}[args.mode](args)


if __name__ == "__main__":
    main()
