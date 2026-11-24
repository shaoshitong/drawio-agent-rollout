"""Prepare the fixed fresh-model SFT and its full-parameter execution check."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .data import load_manifest
from .history import HISTORY_POLICY
from .state_prefix import CONTEXT_BUDGET, POLICY


def fixed_config(base, manifest, model, output, rows):
    policies = {row.get("state_prefix_policy") for row in rows}
    if any(row["token_count"] > CONTEXT_BUDGET or row.get("history_policy") != HISTORY_POLICY
           for row in rows) or len(policies) != 1 or not policies <= {POLICY, "own-state-prefix-v2"}:
        raise ValueError("Fresh SFT requires bounded decisions with one supported prefix policy")
    updates = math.ceil(len(rows) / 32)
    inherited = {key: value for key, value in base.items() if key not in (
        "bridge_checkpoint", "gate_indices", "sampling_pool", "max_train_samples", "store_path")}
    return {**inherited, "manifest": str(manifest.resolve()), "model_name": str(model.resolve()),
            "output": str(output.resolve()), "max_seq_length": CONTEXT_BUDGET,
            "split": "train", "seed": 3407,
            "full_finetuning": True, "thinking_loss_weight": 0.7,
            "batch_size": 1, "gradient_accumulation_steps": 8, "difficulty_stages": 8,
            "num_train_epochs": 1.0, "max_steps": -1, "learning_rate": 5e-6,
            "vision_lr_ratio": 0.2, "save_steps": 10, "phase_start_step": 0,
            "phase_total_updates": updates, "phase_warmup_updates": max(10, math.ceil(updates * 0.03)),
            "phase_decay_updates": math.ceil(updates * 0.2), "phase_min_lr_ratio": 0.1,
            "resume_from_checkpoint": None}


def check_indices(rows):
    priorities = [0, len(rows) - 1]
    priorities.extend(max(range(len(rows)), key=lambda i: rows[i][field])
                      for field in ("token_count", "image_count", "completion_tokens", "final_components"))
    priorities.extend(round(i * (len(rows) - 1) / 63) for i in range(64))
    priorities.extend(range(len(rows)))
    return list(dict.fromkeys(priorities))[:64]


def prepare(args):
    rows = load_manifest(args.manifest, "train")
    if len(rows) < 64 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Fresh SFT needs at least 64 unique measured training decisions")
    base = json.loads(args.base_config.read_text())
    config = fixed_config(base, args.manifest, args.model, args.output, rows)
    indices = check_indices(rows)
    check = {**config, "output": str(args.output.parent / (args.output.name + "-check")),
             "max_steps": 2, "phase_total_updates": 2, "phase_warmup_updates": 1,
             "phase_decay_updates": 1, "evaluation_enabled": False, "preflight_samples": 8,
             "gate_indices": indices, "save_steps": 2}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("config.json", config), ("check-config.json", check)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    report = {"manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
              "train_decisions": len(rows), "training_episodes": len({r["episode"] for r in rows}),
              "world_size": 4, "decisions_per_update": 32,
              "updates_per_epoch": config["phase_total_updates"],
              "full_config": str(args.output / "config.json"),
              "check_config": str(args.output / "check-config.json"),
              "check_unique_decisions": len(indices), "check_gradient_accumulation": 8,
              "model_training_started": False}
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/sft-9b.json")
    print(json.dumps(prepare(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
