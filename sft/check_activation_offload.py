"""Compare native optimizer updates before testing CPU-offloaded long contexts."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_trial(args, config, name, offload, indices):
    output = args.output / name
    output.mkdir()
    config = {**config, "output": str(output), "gate_indices": indices,
              "activation_offloading": offload, "max_steps": len(indices),
              "save_steps": 2, "warmup_ratio": 0.0, "preflight_samples": 4}
    write_json(output / "config.json", config)
    env = dict(os.environ, DRAWIO_SFT_ROOT=str(args.root), DRAWIO_BASE_RUNTIME=str(args.runtime),
               DRAWIO_GPUS=os.environ["CUDA_VISIBLE_DEVICES"], DRAWIO_PROCESSES="1",
               UNSLOTH_COMPILE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
    command = ["bash", str(args.source / "scripts/run_sft.sh"), "train", "--config",
               str(output / "config.json"), "--manifest", config["manifest"], "--output", str(output)]
    with (output / "run.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output / "active.json", {"at": time.time(), "trial": name, "pid": process.pid})
        print(json.dumps({"event": "trial_started", "trial": name, "pid": process.pid}), flush=True)
        code = process.wait()
    (output / "exit-code").write_text(str(code) + "\n")
    if code:
        raise subprocess.CalledProcessError(code, command)
    return output


def difference(pairs):
    import torch

    dot = left_norm = right_norm = error = maximum = 0.0
    elements = 0
    for left, right in pairs:
        if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("Tensor shape or finite-value mismatch")
        left, right = left.double().flatten(), right.double().flatten()
        delta = left - right
        dot += torch.dot(left, right).item()
        left_norm += torch.dot(left, left).item()
        right_norm += torch.dot(right, right).item()
        error += torch.dot(delta, delta).item()
        maximum = max(maximum, delta.abs().max().item())
        elements += left.numel()
    return {"elements": elements, "relative_l2": math.sqrt(error / max(left_norm, 1e-30)),
            "cosine": dot / math.sqrt(max(left_norm * right_norm, 1e-60)), "max_abs": maximum}


def compare(baseline, offload):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    paths = (baseline, offload)
    states = [json.loads((p / "checkpoint-2/trainer_state.json").read_text()) for p in paths]
    logs = [[row for row in state["log_history"] if "loss" in row] for state in states]
    updates = [json.loads((p / "adapter-update.json").read_text()) for p in paths]
    configs = [json.loads((p / "config.json").read_text()) for p in paths]
    config_keys = set(configs[0]) | set(configs[1])
    if any(configs[0].get(k) != configs[1].get(k) for k in config_keys - {"output", "activation_offloading"}):
        raise ValueError("Unexpected configuration difference")
    if updates[0]["initial"] != updates[1]["initial"]:
        raise ValueError("Initial trainable parameters differ")
    if any(len(log) != 2 for log in logs) or any(u["first_update_step"] != 1 for u in updates):
        raise ValueError("Each variant must complete two nonzero-learning-rate updates")
    loss_diff = max(abs(a["loss"] - b["loss"]) for a, b in zip(*logs))
    grad_diff = max(abs(a["grad_norm"] - b["grad_norm"]) / max(abs(a["grad_norm"]), 1e-12)
                    for a, b in zip(*logs))
    adapters = [load_file(str(p / "checkpoint-2/adapter_model.safetensors")) for p in paths]
    if adapters[0].keys() != adapters[1].keys():
        raise ValueError("Adapter parameter names differ")
    adapter = difference((adapters[0][k], adapters[1][k]) for k in adapters[0])
    bmatrix = difference((adapters[0][k], adapters[1][k]) for k in adapters[0] if "lora_B" in k)
    optimizers = [torch.load(p / "checkpoint-2/optimizer.pt", map_location="cpu", weights_only=True)["state"]
                  for p in paths]
    if optimizers[0].keys() != optimizers[1].keys():
        raise ValueError("Optimizer parameter IDs differ")
    moments = {field: difference((optimizers[0][k][field], optimizers[1][k][field]) for k in optimizers[0])
               for field in ("exp_avg", "exp_avg_sq")}
    memory = [json.loads((p / "memory-rank-0.json").read_text()) for p in paths]
    passed = (loss_diff < 0.001 and grad_diff < 0.01 and bmatrix["cosine"] > 0.999
              and moments["exp_avg"]["relative_l2"] < 0.01
              and moments["exp_avg_sq"]["relative_l2"] < 0.02)
    return {"at": time.time(), "passed": passed, "two_real_updates_per_variant": True,
            "initial_trainable_hashes_identical": True, "samples_and_contexts_unchanged": True,
            "max_loss_absolute_difference": loss_diff, "max_gradient_norm_relative_difference": grad_diff,
            "all_adapter_parameters": adapter, "zero_initialized_lora_B": bmatrix,
            "optimizer_moments": moments, "peak_allocated_bytes": [m["max_allocated_bytes"] for m in memory],
            "thresholds": {"loss_absolute": 0.001, "gradient_norm_relative": 0.01,
                           "lora_B_cosine_min": 0.999, "optimizer_exp_avg_relative_l2": 0.01,
                           "optimizer_exp_avg_sq_relative_l2": 0.02},
            "scope": "Two short native decisions; not a long-context or data-repair quality proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "runtime", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--offload", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    indices = config["gate_indices"]
    code = 1
    try:
        baseline = args.baseline or run_trial(args, config, "baseline", False, indices[:2])
        offload = args.offload or run_trial(args, config, "offload", True, indices[:2])
        result = compare(baseline, offload)
        write_json(args.output / "comparison.json", result)
        if not result["passed"]:
            raise ValueError("Activation offloading changes optimizer updates beyond the existing parity gate")
        run_trial(args, config, "capacity", True, indices)
        code = 0
    finally:
        (args.output / "exit-code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
