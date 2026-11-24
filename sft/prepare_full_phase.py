"""Prepare a separate full-parameter run from a frozen LoRA checkpoint."""

import argparse
import json
from pathlib import Path
import shutil


def prepare(previous, checkpoint, model, output):
    output.mkdir(parents=True, exist_ok=False)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    start, total = state["global_step"], state["max_steps"]
    bridge = output / f"checkpoint-{start}-bridge"
    bridge.mkdir()
    state["best_model_checkpoint"] = None
    (bridge / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
    for source in checkpoint.glob("rng_state*.pth"):
        shutil.copy2(source, bridge / source.name)
    for name in ("training-input.json", "curriculum-order.json"):
        shutil.copy2(previous / name, output / name)
    shutil.copytree(previous / "fixed-loss", output / "fixed-loss",
                    ignore=shutil.ignore_patterns("failure-*"))
    config = json.loads((previous / "runtime-config.json").read_text())
    config.update(model_name=str(model), output=str(output), full_finetuning=True,
                  bridge_checkpoint=str(bridge), resume_from_checkpoint=str(bridge),
                  phase_start_step=start, phase_total_updates=total - start,
                  phase_warmup_updates=30, phase_decay_updates=178, phase_min_lr_ratio=0.1,
                  learning_rate=5e-6, vision_lr_ratio=0.2, save_steps=10)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = [json.loads(line) for line in Path(config["manifest"]).open()]
    rows = [row for row in rows if row["split"] == config["split"]]
    short = min(range(len(rows)), key=lambda i: rows[i]["token_count"] if rows[i]["image_count"] else float("inf"))
    text = min(range(len(rows)), key=lambda i: rows[i]["token_count"])
    extremes = [max(range(len(rows)), key=lambda i: rows[i][field])
                for field in ("token_count", "image_count", "completion_tokens")]
    preflight = json.loads((previous / "preflight.json").read_text())
    largest_image = max(preflight["samples"], key=lambda sample: sample["image_tokens"])["sample_id"]
    extremes.append(next(i for i, row in enumerate(rows) if row["sample_id"] == largest_image))
    gate = {**config, "output": str(output.parent / (output.name + "-gate")), "resume_from_checkpoint": None,
            "max_steps": 2, "gradient_accumulation_steps": 1, "preflight_samples": 8,
            "phase_start_step": 0, "phase_total_updates": 2, "phase_warmup_updates": 1,
            "phase_decay_updates": 1, "evaluation_enabled": False, "save_steps": 2,
            "gate_indices": [short, text, short, text, *extremes]}
    (output / "gate-config.json").write_text(json.dumps(gate, indent=2) + "\n")
    return {"bridge": str(bridge), "start_step": start, "total_steps": total,
            "full_config": str(output / "config.json"), "gate_config": str(output / "gate-config.json"),
            "gate_cases": [{key: rows[i][key] for key in ("sample_id", "token_count", "image_count", "completion_tokens")}
                           for i in gate["gate_indices"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "checkpoint", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.previous, args.checkpoint, args.model, args.output)
    (args.output / "transition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
