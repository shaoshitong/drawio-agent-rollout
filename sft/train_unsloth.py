"""Audit actual multimodal labels, then train generator decisions in curriculum order."""

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from types import MethodType

from .data import ManifestDataset, token_sha256


def write_json(path, value):
    if int(os.environ.get("RANK", "0")) == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Examples:
    def __init__(self, manifest, split, enable_thinking, limit=None, store_path=None):
        self.dataset = ManifestDataset(manifest, split=split, store_path=store_path)
        self.length = min(len(self.dataset), limit) if limit else len(self.dataset)
        self.enable_thinking = enable_thinking

    def __len__(self):
        return self.length

    def __call__(self, row):
        index = int(row["row_index"])
        example = self.dataset[index]
        if example["kind"] != "teacher_decision":
            raise ValueError("This curriculum accepts teacher trajectories only")
        for message in example["completion"]:
            message["content"] = message.get("content") or ""
        kwargs = example.setdefault("chat_template_kwargs", {})
        kwargs.setdefault("enable_thinking", self.enable_thinking)
        example["expected_tokens"] = self.dataset.rows[index].get("token_count")
        example["expected_prompt_token_sha256"] = self.dataset.rows[index].get("prompt_token_sha256")
        return example


def render_completion(processor, example):
    kwargs = example["chat_template_kwargs"]
    prompt = processor.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True, **kwargs
    )
    full = processor.apply_chat_template(
        example["prompt"] + example["completion"], tokenize=False,
        add_generation_prompt=False, **kwargs,
    )
    if not full.startswith(prompt):
        raise ValueError("Training full rendering changes its serving prompt prefix")
    return prompt, full[len(prompt):]


def patch_projection(self, inputs, weight, bias):
    import torch.nn.functional as functional

    values = functional.linear(inputs.flatten(1), weight.flatten(1), bias)
    return values[:, :, None, None, None]


def use_linear_patch_embedding(model):
    # Each input is exactly one non-overlapping image patch, as in training.
    projection = next(module.proj for module in model.modules()
                      if type(module).__name__ == "Qwen3_5VisionPatchEmbed")
    projection._conv_forward = MethodType(patch_projection, projection)


def audit_batch(model, processor, examples, batch, max_length):
    attention = batch["attention_mask"].bool()
    if not batch["labels"][~attention].eq(-100).all():
        raise ValueError("Padding positions contribute to loss")
    grids = batch.get("image_grid_thw")
    image_count = sum(len(example["images"]) for example in examples)
    if (0 if grids is None else len(grids)) != image_count:
        raise ValueError("Image grids differ from the number of materialized input images")
    merge = processor.image_processor.merge_size
    image_offset, reports = 0, []
    for index, example in enumerate(examples):
        count = len(example["images"])
        image_tokens = 0 if not count else int(
            (grids[image_offset:image_offset + count].prod(dim=-1) // (merge * merge)).sum()
        )
        image_offset += count
        ids = batch["input_ids"][index][attention[index]]
        labels = batch["labels"][index][attention[index]]
        reports.append(audit_target(model, processor, example, ids, labels, image_tokens, max_length))
    return reports


def audit_target(model, processor, example, ids, labels, image_tokens, max_length):
    import torch

    if any(message.get("reasoning_content") for message in example["prompt"] if message["role"] == "assistant"):
        raise ValueError("Historical thinking reached the training batch")
    prompt, completion = render_completion(processor, example)
    target = processor(
        text=[completion], add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    start = len(ids) - len(target)
    if len(ids) > max_length:
        raise ValueError(f"{example['sample_id']}: expanded length {len(ids)} > {max_length}")
    if example["expected_tokens"] is not None and len(ids) != example["expected_tokens"]:
        raise ValueError(f"{example['sample_id']}: runtime length differs from the prepared manifest")
    if start < 1 or not torch.equal(ids[start:].cpu(), target.cpu()):
        raise ValueError(f"{example['sample_id']}: collated completion token suffix differs")
    prompt_token_sha256 = token_sha256(ids[:start])
    if example.get("expected_prompt_token_sha256") not in (None, prompt_token_sha256):
        raise ValueError(f"{example['sample_id']}: conditioning token IDs differ from native preparation")
    if not labels[:start].eq(-100).all() or not torch.equal(labels[start:].cpu(), target.cpu()):
        raise ValueError(f"{example['sample_id']}: loss leaks into history or masks a target token")
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not target.eq(end_id).any():
        raise ValueError(f"{example['sample_id']}: completion has no supervised end-of-turn token")
    image_id = model.config.image_token_id
    if int(ids.eq(image_id).sum()) != image_tokens:
        raise ValueError(f"{example['sample_id']}: expanded visual placeholders do not match image grids")
    reasoning = (example["completion"][0].get("reasoning_content") or "").strip()
    if reasoning and not completion.startswith(reasoning):
        raise ValueError(f"{example['sample_id']}: current reasoning start is outside the target")
    return {
        "sample_id": example["sample_id"], "sequence_tokens": len(ids),
        "prompt_tokens": start, "target_tokens": len(target), "image_tokens": image_tokens,
        "images": len(example["images"]), "has_reasoning": bool(reasoning),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in example["completion"]),
        "history_thinking_messages": 0,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_token_sha256": prompt_token_sha256,
        "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
    }


def parameter_audit(model, target_modules, full=False):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    visual = [name for name, _ in trainable if "visual" in name or "merger" in name]
    if full:
        import torch
        from .full_finetune import parameter_group

        groups = Counter()
        for name, parameter in trainable:
            groups[parameter_group(name)] += parameter.numel()
        if set(groups) != {"vision", "merger", "embedding", "head", "language"}:
            raise ValueError(f"Full finetuning is missing active parameter groups: {groups}")
        if any(parameter.dtype != torch.float32 for _, parameter in trainable):
            raise ValueError("Full finetuning requires FP32 master parameters")
        return {"trainable_parameters": sum(groups.values()), "parameter_groups": dict(groups),
                "master_dtype": "float32", "frozen_names": [name for name, p in model.named_parameters()
                                                               if not p.requires_grad]}
    coverage = {suffix: sum(f".{suffix}." in name for name, _ in trainable) for suffix in target_modules}
    if visual or not all(coverage.values()):
        raise ValueError(f"LoRA scope mismatch: visual={visual}, projection coverage={coverage}")
    return {
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "dtypes": dict(Counter(str(p.dtype) for _, p in trainable)),
        "projection_coverage": coverage,
        "trainable_names": [name for name, _ in trainable],
        "visual_and_merger_frozen": True,
    }


def trainable_hash(model):
    import torch

    digest, tensors, parameters = hashlib.sha256(), 0, 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}\n".encode())
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        digest.update(local.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        tensors += 1
        parameters += parameter.numel()
    return {"sha256": digest.hexdigest(), "tensors": tensors, "parameters": parameters,
            "scope": "rank-local trainable storage (shards for FSDP2)"}


def make_collator(model, processor, examples, config, output):
    from unsloth.trainer import UnslothVisionDataCollator

    class AuditedCollator(UnslothVisionDataCollator):
        def __call__(self, rows):
            materialized = [examples(row) for row in rows]
            try:
                batch = super().__call__(materialized)
                self.last_report = audit_batch(model, processor, materialized, batch, config["max_seq_length"])
            except Exception as error:
                write_json(output / "runtime-audit-failure.json", {
                    "status": "failed", "sample_ids": [x["sample_id"] for x in materialized],
                    "error": f"{type(error).__name__}: {error}",
                })
                raise
            self.audited_batches += 1
            self.audited_examples += len(materialized)
            if getattr(self, "record_batches", False):
                path = output / f"batches-rank-{os.environ.get('RANK', '0')}.jsonl"
                with path.open("a") as stream:
                    stream.write(json.dumps({"batch": self.audited_batches, "samples": self.last_report}) + "\n")
            if self.audited_batches % 25 == 0:
                write_json(output / "runtime-audit.json", self.audit_progress())
            return batch

        def audit_progress(self):
            return {"batches": self.audited_batches, "examples": self.audited_examples,
                    "scope": "every collated batch, before model forward", "recollation": False}

    collator = AuditedCollator(model, processor, resize="max", completion_only_loss=True)
    # resize=max disables only the collator's extra resize; the unchanged official
    # processor still applies its default image policy, exactly as manifest preparation.
    collator.max_seq_length = None
    collator.truncation = False
    collator.audited_batches = collator.audited_examples = 0
    collator.last_report = []
    return collator


def setup(config):
    from unsloth import FastVisionModel
    import torch

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    full = config.get("full_finetuning", False)
    model, processor = FastVisionModel.from_pretrained(
        model_name=config["model_name"], max_seq_length=config["max_seq_length"],
        dtype=torch.float32 if full else torch.bfloat16, load_in_4bit=False, load_in_16bit=True,
        **({"full_finetuning": True, "float32_mixed_precision": True,
            "use_gradient_checkpointing": "unsloth", "attn_implementation": "sdpa"} if full else {}),
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
    )
    existing = {name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and "visual" not in name}
    targets = sorted(existing.intersection(config["target_modules"]))
    if not any(name.startswith("in_proj_") for name in targets) or "out_proj" not in targets:
        raise ValueError("Configured LoRA targets do not cover the loaded GatedDeltaNet projections")
    config["resolved_target_modules"] = targets
    if not full:
        model = FastVisionModel.get_peft_model(
            model, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=0, bias="none", target_modules=targets,
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth", random_state=config["seed"],
        )
    FastVisionModel.for_training(model)
    if config.get("mlp_tile_rows"):
        from unsloth import tile_mlp

        tiled = tile_mlp(model, rows=config["mlp_tile_rows"], modules="*language_model.layers.*.mlp")
        if not tiled.tiled:
            raise RuntimeError("No language MLPs were tiled")
        write_json(Path(config["output"]) / "tiled-mlp.json", {
            "rows": config["mlp_tile_rows"], "tiled": tiled.tiled, "skipped": tiled.skipped,
        })
    use_linear_patch_embedding(model)
    examples = Examples(config["manifest"], config["split"], config["enable_thinking"],
                        config.get("max_train_samples"), config.get("store_path"))
    collator = make_collator(model, processor, examples, config, Path(config["output"]))
    return model, processor, examples, collator


def preflight_indices(examples, limit):
    rows = examples.dataset.rows[:len(examples)]
    if not rows:
        raise ValueError("No trajectory examples in the requested split")
    priorities = [0, len(rows) - 1]
    for field in ("token_count", "image_count", "final_components"):
        priorities.append(max(range(len(rows)), key=lambda index: rows[index].get(field, 0)))
    strata = {
        "reasoning": lambda row: row.get("has_reasoning", False),
        "without_reasoning": lambda row: not row.get("has_reasoning", False),
        "tool_only": lambda row: row.get("tool_calls", 0) > 0 and not row.get("has_content", False),
        "no_tool_call": lambda row: row.get("tool_calls", 0) == 0,
        "zero_images": lambda row: row.get("image_count", 0) == 0,
        "multiple_images": lambda row: row.get("image_count", 0) > 1,
    }
    available = {}
    for name, predicate in strata.items():
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        available[name] = len(indices)
        if indices:
            priorities.extend((indices[0], indices[len(indices) // 2], indices[-1]))
    count = min(limit, len(rows))
    priorities.extend(round(index * (len(rows) - 1) / max(count - 1, 1)) for index in range(count))
    priorities.extend(range(len(rows)))
    selected = sorted(list(dict.fromkeys(priorities))[:count])
    return selected, available


def preflight(config, model, processor, examples, collator, output):
    started = time.monotonic()
    versions = {name: importlib.metadata.version(name) for name in
                ("unsloth", "unsloth_zoo", "transformers", "trl", "torch", "peft", "datasets")}
    import unsloth_zoo.vision_utils as vision_utils

    source = Path(vision_utils.__file__)
    report = {
        "status": "running", "config": config, "versions": versions,
        "manifest_sha256": hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest(),
        "collator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameters": parameter_audit(model, config["resolved_target_modules"], config.get("full_finetuning", False)),
        "samples": [], "runtime_only": True,
    }
    selected, strata = preflight_indices(examples, config["preflight_samples"])
    report.update(total_manifest_examples=len(examples), selected_indices=selected, available_strata=strata,
                  scope="stratified preflight; remaining examples audited in their actual training batch")
    try:
        for count, index in enumerate(selected, 1):
            collator([{"row_index": index}])
            report["samples"].extend(collator.last_report)
            if count % 8 == 0:
                write_json(output / "preflight.json", report)
                print(json.dumps({"preflight_audited": count, "selected": len(selected)}), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "preflight.json", report)
        raise
    report.update(status="passed", elapsed_seconds=time.monotonic() - started,
                  audited_examples=len(selected),
                  sampled_supervised_tokens=sum(x["target_tokens"] for x in report["samples"]))
    write_json(output / "preflight.json", report)
    return report


def curriculum_order(examples, config, world_size):
    import torch
    from transformers.trainer_pt_utils import LengthGroupedSampler

    rows = examples.dataset.rows[:len(examples)]
    pool = list(range(len(rows)))
    sampling_path = config.get("sampling_pool")
    if sampling_path:
        sampling = json.loads(Path(sampling_path).read_text())
        manifest_hash = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
        if sampling["manifest_sha256"] != manifest_hash:
            raise ValueError("Sampling pool belongs to a different training manifest")
        pool = sampling["source_indices"]
        if len(pool) != len(rows) or any(index < 0 or index >= len(rows) for index in pool):
            raise ValueError("Sampling pool differs from the complete training split")
    episodes = {row["episode"]: row["final_components"] for row in rows}
    components = sorted(episodes.values())
    count = config["difficulty_stages"]
    boundaries = [components[math.ceil(stage * len(components) / count) - 1]
                  for stage in range(1, count)]
    order, stages = [], []
    duplicate_swaps = 0
    global_batch = world_size * config["batch_size"]
    for stage in range(count):
        indices = [index for index in pool
                   if bisect_left(boundaries, rows[index]["final_components"]) == stage]
        if not indices:
            continue
        seed = config["seed"] + stage
        sampler = LengthGroupedSampler(
            batch_size=global_batch, lengths=[rows[index]["token_count"] for index in indices],
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        selected = [indices[index] for index in sampler]
        if sampling_path:
            from .sampling_pool import separate_update_duplicates

            selected, swaps = separate_update_duplicates(
                selected, rows, global_batch * config["gradient_accumulation_steps"], len(order))
            duplicate_swaps += swaps
        order.extend(selected)
        stages.append({
            "difficulty_stage": stage, "seed": seed,
            "episodes": len({rows[index]["episode"] for index in indices}), "decisions": len(indices),
            "components_min": min(rows[index]["final_components"] for index in indices),
            "components_max": max(rows[index]["final_components"] for index in indices),
            "generation_stage_counts": dict(Counter(str(rows[index]["stage"]) for index in indices)),
            "source_indices_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
        })
    plan = {
        "policy": config["sampler_policy"], "seed": config["seed"], "world_size": world_size,
        "global_microbatch_size": global_batch, "difficulty_boundaries": boundaries,
        "difficulty_stage_rule": "bisect_left: components <= boundary belong to the lower stage",
        "quantile_unit": "unique training episode, not decision",
        "generation_stage_field": "original row.stage (generation1 / repair2), independent of difficulty",
        "stages": stages, "decisions": len(order),
        "source_indices_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        "sample_ids_sha256": hashlib.sha256(json.dumps([rows[index]["sample_id"] for index in order]).encode()).hexdigest(),
        "sample_contents": "unchanged full prefixes and current completions; order only",
    }
    if sampling_path:
        plan["sampling_pool"] = {
            "path": str(sampling_path), "sha256": hashlib.sha256(Path(sampling_path).read_bytes()).hexdigest(),
            "duplicate_separation_swaps": duplicate_swaps,
            "optimizer_update_decisions": global_batch * config["gradient_accumulation_steps"],
        }
        plan["sample_contents"] = "unchanged full prefixes and current completions; audited decision resampling"
    return order, plan


def train(config, model, processor, examples, collator, output):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    order, plan = curriculum_order(examples, config, int(os.environ.get("WORLD_SIZE", "1")))
    if config.get("gate_indices"):
        order = config["gate_indices"]
        plan["execution_subset"] = {
            "decisions": len(order),
            "source_indices_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            "difficulty_stages": dict(Counter(
                bisect_left(plan["difficulty_boundaries"], examples.dataset.rows[index]["final_components"])
                for index in order)),
        }
    if config.get("resume_from_checkpoint"):
        previous = json.loads((Path(config["resume_from_checkpoint"]).parent / "curriculum-order.json").read_text())
        if previous["source_indices_sha256"] != plan["source_indices_sha256"]:
            raise ValueError("Resume sampler order differs from the saved curriculum")
    write_json(output / "curriculum-order.json", plan)

    class CurriculumTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return order

        def log(self, logs, *args, **kwargs):
            logs["step"] = self.state.global_step
            for key in ("loss", "train_loss", "grad_norm"):
                if key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Non-finite training metric: {key}={logs[key]}")
            return super().log(logs, *args, **kwargs)

    evidence = {}

    class UpdateEvidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            import torch

            torch.cuda.reset_peak_memory_stats()
            if state.is_world_process_zero:
                evidence.update(initial=trainable_hash(trainer.accelerator.unwrap_model(trainer.model)),
                                initial_global_step=state.global_step,
                                loss_runtime={
                                    "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
                                    "average_tokens_across_devices": args.average_tokens_across_devices,
                                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                    "world_size": args.world_size,
                                    "activation_offloading": args.activation_offloading,
                                })
                write_json(output / "adapter-update.json", evidence)

        def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
            if not state.is_world_process_zero or "first_update" in evidence:
                return
            rates = [group["lr"] for group in optimizer.param_groups]
            # Warmup can intentionally give the first optimizer step zero learning rate.
            if not any(rate != 0 for rate in rates):
                return
            current = trainable_hash(trainer.accelerator.unwrap_model(trainer.model))
            if current["sha256"] == evidence["initial"]["sha256"]:
                raise RuntimeError("A nonzero-LR optimizer step did not change trainable weights")
            evidence.update(first_update=current, first_update_step=state.global_step + 1,
                            first_update_learning_rates=rates)
            write_json(output / "adapter-update.json", evidence)

        def on_train_end(self, args, state, control, **kwargs):
            import torch

            (output / f"memory-rank-{args.process_index}.json").write_text(json.dumps({
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_total_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                "global_step": state.global_step,
            }, indent=2) + "\n")
            if state.is_world_process_zero:
                evidence.update(final=trainable_hash(trainer.accelerator.unwrap_model(trainer.model)),
                                final_global_step=state.global_step)
                write_json(output / "adapter-update.json", evidence)
                if "first_update" not in evidence:
                    raise RuntimeError("Training ended without an observed optimizer update")

    dataset = Dataset.from_dict({"row_index": list(range(len(examples)))})
    args = SFTConfig(
        output_dir=str(output), num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=True, fp16=False, optim="adamw_torch", seed=config["seed"],
        warmup_ratio=config["warmup_ratio"], lr_scheduler_type=config["lr_scheduler_type"],
        max_grad_norm=config["max_grad_norm"],
        max_length=None, packing=False, padding_free=False, assistant_only_loss=False,
        completion_only_loss=True, remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}, dataloader_num_workers=0,
        dataloader_drop_last=False, logging_steps=1, logging_nan_inf_filter=False, report_to="none",
        save_strategy="steps", save_steps=config["save_steps"], save_total_limit=2,
        save_only_model=False, gradient_checkpointing=True, ddp_find_unused_parameters=False,
        activation_offloading=config.get("activation_offloading", False),
    )
    trainer_class = CurriculumTrainer
    if config.get("full_finetuning"):
        from .full_finetune import FullFinetuneTrainer

        trainer_class = type("FullCurriculumTrainer", (FullFinetuneTrainer, CurriculumTrainer), {"full_config": config})
    trainer = trainer_class(
        model=model, processing_class=processor, train_dataset=dataset,
        data_collator=collator, args=args, callbacks=[UpdateEvidence()],
    )
    if config.get("activation_offloading"):
        from trl.models.activation_offloading import OffloadActivations

        # Trainer already installed the output-head exclusion hooks.
        offload = OffloadActivations(use_streams=False)
        offload.update_model_params(trainer.model)
        trainer.maybe_activation_offload_context = offload
    if config.get("full_finetuning"):
        # The fused Qwen forward consumes this count, but the bare HF model
        # declares False. Set the Trainer flag after all Unsloth init patches.
        trainer.model_accepts_loss_kwargs = True
    # Apply after Unsloth initializes its compiler defaults. Runtime tuning
    # synchronizes all streams and can wait on another rank's DDP reduction.
    if args.world_size > 1:
        import torch._inductor.config as inductor_config

        inductor_config.triton.autotune_pointwise = False
        inductor_config.dynamic_scale_rblock = False
    trainer.curriculum_plan = plan
    if config.get("full_finetuning"):
        from .full_finetune import make_full_callback

        trainer.add_callback(make_full_callback(trainer, config, output))
    if config.get("evaluation_enabled", True):
        from .evaluation import make_evaluation_callback

        validation = Examples(config["manifest"], "validation", config["enable_thinking"],
                              store_path=config.get("store_path"))
        probe_collator = make_collator(model, processor, examples, config, output / "train-probe")
        validation_collator = make_collator(model, processor, validation, config, output / "validation")
        trainer.add_callback(make_evaluation_callback(
            trainer, examples, validation, probe_collator, validation_collator,
            config["manifest"], output, validation_samples=config["validation_samples"],
            train_samples=config["train_probe_samples"],
        ))
    collator.audited_batches = collator.audited_examples = 0
    collator.record_batches = config.get("full_finetuning", False)
    result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    # A short explicit smoke run also needs optimizer and RNG state, even before save_steps.
    if not (output / f"checkpoint-{trainer.state.global_step}" / "trainer_state.json").is_file():
        trainer._save_checkpoint(trainer.model, trial=None)
    trainer.save_model(str(output / "final"))
    if trainer.state.is_world_process_zero:
        processor.save_pretrained(output / "final")
    trainer.save_state()
    write_json(output / "result.json", {
        "status": "completed", "global_step": trainer.state.global_step,
        "metrics": result.metrics, "last_checkpoint": str(output / f"checkpoint-{trainer.state.global_step}"),
        "curriculum": plan,
        "runtime_audit": collator.audit_progress(),
        "adapter_update": evidence,
    })


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/sft-9b.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--preflight-samples", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--split")
    parser.add_argument("--sampling-pool", type=Path)
    parser.add_argument("--store-path", type=Path, help="Read accepted decisions from a relocated content-addressed archive")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update({key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if value is not None and key not in ("config", "mode")})
    if config.get("allocator_conf"):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["PYTORCH_ALLOC_CONF"] = config["allocator_conf"]
    if config.get("compile_enabled"):
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "0"
        os.environ["UNSLOTH_COMPILE_LOCATION"] = config["compile_location"]
    if config["preflight_samples"] < 1:
        parser.error("preflight-samples must be positive")
    output = args.output.resolve()
    if args.mode == "train" and not config.get("resume_from_checkpoint"):
        existing = list(output.glob("checkpoint-*")) + list(output.glob("final"))
        if existing:
            raise ValueError("Existing training output: resume explicitly or choose a pristine output directory")
    if args.mode == "train":
        identity = {key: config.get(key) for key in
                    ("model_name", "split", "max_train_samples", "max_seq_length", "num_train_epochs", "max_steps",
                     "lora_rank", "lora_alpha", "target_modules",
                     "batch_size", "gradient_accumulation_steps", "difficulty_stages", "sampler_policy", "seed",
                     "lr_scheduler_type", "learning_rate", "warmup_ratio", "max_grad_norm")}
        if config.get("full_finetuning"):
            identity["full_phase"] = {key: config[key] for key in
                                      ("phase_start_step", "phase_total_updates", "phase_warmup_updates",
                                       "phase_decay_updates", "phase_min_lr_ratio", "vision_lr_ratio")}
        identity["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        identity["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if config.get("activation_offloading"):
            identity["activation_offloading"] = True
        if config.get("sampling_pool"):
            identity["sampling_pool_sha256"] = hashlib.sha256(Path(config["sampling_pool"]).read_bytes()).hexdigest()
        if config.get("gate_indices"):
            identity["execution_subset_sha256"] = hashlib.sha256(json.dumps(config["gate_indices"]).encode()).hexdigest()
        if config.get("mlp_tile_rows"):
            identity["mlp_tile_rows"] = config["mlp_tile_rows"]
        if config.get("compile_enabled"):
            identity["compile_enabled"] = True
        if config.get("resume_from_checkpoint"):
            checkpoint = Path(config["resume_from_checkpoint"])
            bridge = config.get("full_finetuning") and checkpoint.resolve() == Path(config["bridge_checkpoint"]).resolve()
            previous = json.loads((checkpoint.parent / "training-input.json").read_text())
            immutable = ("split", "max_seq_length", "batch_size", "gradient_accumulation_steps", "difficulty_stages",
                         "sampler_policy", "seed", "world_size", "manifest_sha256", "num_train_epochs", "max_steps")
            if (any(previous[key] != identity[key] for key in immutable) if bridge else previous != identity):
                raise ValueError("Resume input or LoRA scope differs from the saved curriculum")
            required = ("trainer_state.json",) if bridge else ("scheduler.pt", "trainer_state.json")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rng = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
            if not all((checkpoint / name).is_file() for name in (*required, *rng)):
                raise ValueError("Resume requires a complete optimizer, scheduler, trainer and RNG checkpoint")
            if not bridge and not ((checkpoint / "optimizer.pt").is_file() or
                                   (checkpoint / "optimizer_dcp").is_dir()):
                raise ValueError("Resume optimizer state is missing")
        write_json(output / "training-input.json", identity)
    write_json(output / "runtime-config.json", config)
    model, processor, examples, collator = setup(config)
    preflight(config, model, processor, examples, collator, output)
    if args.mode == "train":
        train(config, model, processor, examples, collator, output)


if __name__ == "__main__":
    main()
