"""Fixed full-prefix decision loss probes, independent of training presentation."""

from bisect import bisect_left
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path


def select_probes(rows, count, boundaries):
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[(bisect_left(boundaries, row["final_components"]), row["stage"])].append(index)
    for indices in buckets.values():
        indices.sort(key=lambda index: (rows[index]["token_count"], rows[index]["sample_id"]))
    selected, episodes, turn = [], set(), 0
    count = min(count, len(rows))
    rounds = max(1, math.ceil(count / max(1, len(buckets))))
    while len(selected) < count:
        for key in sorted(buckets):
            available = [index for index in buckets[key] if index not in selected]
            if not available or len(selected) == count:
                continue
            diverse = [index for index in available if rows[index]["episode"] not in episodes]
            candidates = diverse or available
            position = min(len(candidates) - 1, int((turn + 0.5) * len(candidates) / rounds))
            chosen = candidates[position]
            selected.append(chosen)
            episodes.add(rows[chosen]["episode"])
        turn += 1
    return selected


def make_evaluation_callback(trainer, train_examples, validation_examples, train_collator,
                             validation_collator, manifest_path, output_dir,
                             validation_samples=32, train_samples=16, difficulty_boundaries=None):
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback

    directory = Path(output_dir) / "fixed-loss"
    directory.mkdir(parents=True, exist_ok=True)
    main = trainer.accelerator.is_main_process
    sources = {"train_probe": (train_examples, train_collator),
               "validation": (validation_examples, validation_collator)}
    boundaries = list(difficulty_boundaries if difficulty_boundaries is not None
                      else trainer.curriculum_plan["difficulty_boundaries"])
    manifest_hash = sha256(Path(manifest_path).read_bytes()).hexdigest()
    plan_path, history_path = directory / "plan.json", directory / "metrics.jsonl"
    baseline_path = directory / "baseline.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["manifest_sha256"] != manifest_hash or plan["difficulty_boundaries"] != boundaries:
            raise ValueError("Fixed-loss manifest or difficulty boundaries changed")
    else:
        plan = {"schema": "drawio-fixed-loss/1", "manifest_sha256": manifest_hash,
                "difficulty_boundaries": boundaries, "probes": {}}
        for split, count in (("train_probe", train_samples), ("validation", validation_samples)):
            rows = sources[split][0].dataset.rows
            if not rows or count < 1:
                raise ValueError("Fixed loss requires nonempty validation and training probes")
            plan["probes"][split] = [{"index": index, "sample_id": rows[index]["sample_id"]}
                                      for index in select_probes(rows, count, boundaries)]
        if main:
            temporary = plan_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(plan, indent=2) + "\n")
            temporary.replace(plan_path)
    for split, probes in plan["probes"].items():
        rows = sources[split][0].dataset.rows
        if any(rows[probe["index"]]["sample_id"] != probe["sample_id"] for probe in probes):
            raise ValueError("Fixed-loss probe IDs no longer match the original manifest")
    plan_hash = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    if any(record["plan_sha256"] != plan_hash for record in history):
        raise ValueError("Existing loss metrics belong to a different fixed probe plan")
    completed = {record["global_step"] for record in history}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if baseline["plan_sha256"] != plan_hash or baseline["global_step"] != 0:
            raise ValueError("Existing fixed-loss baseline does not match this plan")
        completed.add(0)
    groups = [(split, dimension, stage) for split in sources
              for dimension, stages in (("overall", [0]), ("generation_stage", [1, 2]),
                                        ("difficulty_stage", range(len(boundaries) + 1))) for stage in stages]

    class FixedLoss(TrainerCallback):
        def evaluate(self, step):
            if step in completed:
                return
            if any(os.environ.get(name) == "1" for name in
                   ("UNSLOTH_RETURN_LOGITS", "UNSLOTH_RETURN_HIDDEN_STATES")):
                raise ValueError("Fixed loss requires the fused loss-only forward path")
            model = trainer.accelerator.unwrap_model(trainer.model)
            device = trainer.accelerator.device
            rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
            values = torch.zeros(len(groups) * 4 + 1, dtype=torch.float64, device=device)
            stats = values[:-1].view(len(groups), 4)
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                     enabled=device.type == "cuda"):
                    for split, probes in plan["probes"].items():
                        examples, collator = sources[split]
                        for probe in probes[rank::world]:
                            index = probe["index"]
                            batch = collator([{"row_index": index}])
                            tokens = int(batch["labels"][..., 1:].ne(-100).sum())
                            if not tokens:
                                raise ValueError("Fixed-loss decision has no shifted target tokens")
                            batch = {key: value.to(device) if torch.is_tensor(value) else value
                                     for key, value in batch.items()}
                            output = model(**batch, use_cache=False)
                            loss = output.loss.detach().double()
                            if loss.numel() != 1 or not torch.isfinite(loss).all():
                                raise FloatingPointError("Non-finite fixed-decision cross entropy")
                            row = examples.dataset.rows[index]
                            keys = [(split, "overall", 0), (split, "generation_stage", row["stage"]),
                                    (split, "difficulty_stage", bisect_left(boundaries, row["final_components"]))]
                            for key in keys:
                                stats[groups.index(key)] += torch.stack((loss * tokens, loss.new_tensor(tokens),
                                                                        loss, loss.new_tensor(1)))
                            del output, batch
            except Exception as error:
                values[-1] = 1
                (directory / f"failure-rank{rank}.json").write_text(json.dumps(
                    {"global_step": step, "error": f"{type(error).__name__}: {error}"}) + "\n")
            finally:
                model.train(was_training)
            if dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            if values[-1].item():
                raise RuntimeError("Fixed loss evaluation failed; see per-rank failure records")
            metrics = {}
            for key, (total, tokens, macro, decisions) in zip(groups, stats.cpu().tolist()):
                split, dimension, stage = key
                label = dimension if dimension == "overall" else f"{dimension}/{stage + (dimension == 'difficulty_stage')}"
                metrics[f"{split}/{label}"] = {"target_token_mean_ce": total / tokens if tokens else None,
                    "decision_macro_ce": macro / decisions if decisions else None,
                    "target_tokens": int(tokens), "decisions": int(decisions)}
            record = {"global_step": step, "plan_sha256": plan_hash, "metrics": metrics,
                      "baseline_available": step == 0 or baseline_path.exists()}
            if main:
                if step == 0 and not baseline_path.exists():
                    with baseline_path.open("x") as stream:
                        stream.write(json.dumps(record, indent=2) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                with history_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps({"fixed_loss_step": step, "metrics": metrics}), flush=True)
            completed.add(step)

        def on_train_begin(self, args, state, control, **kwargs):
            self.checkpoints = {math.ceil(state.max_steps * fraction) for fraction in (0.25, 0.5, 0.75, 1)}
            if state.global_step == 0:
                self.evaluate(0)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.checkpoints:
                self.evaluate(state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            self.evaluate(state.global_step)

    return FixedLoss()
