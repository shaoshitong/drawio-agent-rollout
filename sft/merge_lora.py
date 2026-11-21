"""Merge an ordinary LoRA into FP32 master weights without a BF16 round trip."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_tensor(name, tensor, adapter, config):
    import torch

    prefix = "base_model.model." + name.removesuffix(".weight")
    key = prefix + ".lora_A.weight"
    master = tensor.float()
    if key not in adapter:
        return master, None
    left = adapter[prefix + ".lora_B.weight"].float()
    right = adapter[key].float()
    delta = (left @ right) * (config["lora_alpha"] / config["r"])
    merged = master + delta
    changed = merged.ne(master)
    rounded = merged.to(tensor.dtype).float()
    report = {"name": name, "elements": tensor.numel(),
              "delta_max": float(delta.abs().max()),
              "fp32_changed": int(changed.sum()),
              "bf16_changed": int(rounded.ne(master).sum()),
              "bf16_rounding_max": float((merged - rounded).abs().max())}
    if not torch.isfinite(merged).all():
        raise FloatingPointError(f"Nonfinite merged tensor: {name}")
    return merged, report


def merge(base, checkpoint, output):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    unsupported = (config.get("use_dora"), config.get("use_rslora"), config.get("fan_in_fan_out"),
                   config.get("rank_pattern"), config.get("alpha_pattern"), config.get("modules_to_save"))
    if any(unsupported):
        raise ValueError("This merger accepts standard, uniform-rank linear LoRA only")
    adapter = load_file(checkpoint / "adapter_model.safetensors", device="cpu")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    reports, files, size = [], {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(base / shard, device="cpu")
        for name, tensor in tensors.items():
            tensors[name], report = merge_tensor(name, tensor, adapter, config)
            size += tensors[name].numel() * tensors[name].element_size()
            if report is not None:
                reports.append(report)
        temporary = output / (shard + ".tmp")
        save_file(tensors, temporary, metadata={"format": "pt"})
        temporary.rename(output / shard)
        files[shard] = {"bytes": (output / shard).stat().st_size, "sha256": sha256(output / shard)}
        print(json.dumps({"merged_shard": shard, "lora_tensors": len(reports)}), flush=True)
        del tensors
    if len(reports) * 2 != len(adapter):
        raise ValueError("Some adapter tensors were not merged into a base tensor")
    for source in base.iterdir():
        if source.is_file() and source.suffix in (".json", ".jinja", ".txt", ".model"):
            shutil.copy2(source, output / source.name)
    model_config = json.loads((output / "config.json").read_text())
    model_config["dtype"] = model_config["torch_dtype"] = "float32"
    (output / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    index["metadata"]["total_size"] = size
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    report = {"status": "merged", "precision": "float32", "base": str(base),
              "adapter": str(checkpoint), "adapter_sha256": sha256(checkpoint / "adapter_model.safetensors"),
              "global_step": json.loads((checkpoint / "trainer_state.json").read_text())["global_step"],
              "tensors": reports, "files": files, "model_bytes": size}
    (output / "merge-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.base, args.checkpoint, args.output)
    print(json.dumps({key: report[key] for key in ("status", "global_step", "precision", "model_bytes")}))


if __name__ == "__main__":
    main()
