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


def native_batch(processor, row, directory):
    import torch
    from PIL import Image

    images = [Image.open(directory / name).convert("RGB") for name in row["images"]]
    kwargs = {"images": images} if images else {}
    batch = processor(text=[row["prompt"]], add_special_tokens=False, return_tensors="pt", **kwargs)
    prompt = batch["input_ids"]
    target = processor(text=[row["completion"]], add_special_tokens=False, return_tensors="pt")["input_ids"]
    if token_sha256(prompt[0]) != row["prompt_token_sha256"]:
        raise ValueError(f"{row['sample_id']}: native prompt token IDs changed")
    if prompt.shape[1] + target.shape[1] != row["sequence_tokens"] or target.shape[1] != row["target_tokens"]:
        raise ValueError(f"{row['sample_id']}: native sequence or loss-mask length changed")
    batch["input_ids"] = torch.cat((prompt, target), dim=1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in batch:
            batch[name] = torch.cat((batch[name], torch.zeros_like(target)), dim=1)
    return batch, target[0], prompt.shape[1]


def harvest(args):
    import torch
    from safetensors.torch import save_file, load_file
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
    root = model.get_base_model() if args.variant == "peft" else model
    if args.layer_progress:
        for index, layer in enumerate(root.model.language_model.layers):
            layer.register_forward_pre_hook(lambda _module, _inputs, index=index: print(json.dumps(
                {"event": "layer_start", "variant": args.variant, "layer": index, "at": time.time()}), flush=True))
            layer.register_forward_hook(lambda _module, _inputs, _output, index=index: print(json.dumps(
                {"event": "layer_done", "variant": args.variant, "layer": index, "at": time.time()}), flush=True))
    directory = args.output / args.variant
    directory.mkdir(parents=True, exist_ok=False)
    head = root.lm_head.weight.detach().to(torch.bfloat16).contiguous().view(torch.uint8).cpu()
    (directory / "head-sha256.txt").write_text(hashlib.sha256(head.numpy().tobytes()).hexdigest() + "\n")
    del head
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(rows):
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
            save_file({"hidden": hidden, "target": target}, directory / f"{index:03}.safetensors",
                      metadata={"sample_id": row["sample_id"], "prompt_tokens": str(prompt_tokens)})
            progress = {"variant": args.variant, "completed_probes": index + 1, "total_probes": len(rows),
                        "last_sample_id": row["sample_id"], "at": time.time(),
                        "full_native_prefixes_verified": True, "training": False}
            (directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
            del result, hidden, batch


def scores(args):
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from safetensors import safe_open

    index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.base / index["lm_head.weight"], framework="pt", device="cpu") as source:
        head = source.get_tensor("lm_head.weight").to("cuda", dtype=torch.bfloat16)
    rows = [json.loads(line) for line in (args.probes / "probes.jsonl").read_text().splitlines()]
    variants = ("base", "peft", "merged")
    heads = [(args.output / name / "head-sha256.txt").read_text().strip() for name in variants]
    if len(set(heads)) != 1:
        raise ValueError("Shared-head parity applies to LoRA merges, not changed full-finetuning heads")
    labels = ("0-32K", "32-64K", "64-128K", "128K+")
    groups = {name: {"tokens": 0, "ce_sum": [0.0] * 3, "kl_sum": 0.0, "agreements": 0}
              for name in ("overall", *labels)}
    with torch.inference_mode():
        for position, row in enumerate(rows):
            states = [load_file(args.output / variant / f"{position:03}.safetensors") for variant in variants]
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
                for name in groups:
                    mask = torch.ones_like(same) if name == "overall" else bins.eq(labels.index(name))
                    group = groups[name]
                    group["tokens"] += int(mask.sum())
                    group["ce_sum"] = [a + b for a, b in zip(group["ce_sum"], ce[:, mask].double().sum(1).tolist())]
                    group["kl_sum"] += float(kl[mask].double().sum())
                    group["agreements"] += int(same[mask].sum())
    metrics = {}
    for name, group in groups.items():
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
                   if name != "overall")
    report = {"metrics": metrics, "overall_export_gate_passed": gate,
              "position_bin_gate_passed": bin_gate, "export_gate_passed": gate and bin_gate, "probes": len(rows),
              "scope": "HF true PEFT vs BF16 merged; serving parity and reconstruction assessed separately",
              "all_native_prefixes_preserved": True, "gold_feedback_to_dsh": False}
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("harvest", "score"))
    for name in ("base", "probes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--variant", choices=("base", "peft", "merged"))
    parser.add_argument("--layer-progress", action="store_true")
    args = parser.parse_args()
    (harvest if args.mode == "harvest" else scores)(args)


if __name__ == "__main__":
    main()
