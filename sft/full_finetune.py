"""Full-parameter phase transitions with a fresh optimizer and unchanged data cursor."""

from collections import defaultdict
import json
from pathlib import Path


def parameter_group(name):
    if ".visual." in name:
        return "merger" if ".merger." in name else "vision"
    if "embed_tokens" in name:
        return "embedding"
    return "head" if "lm_head" in name else "language"


def lr_factor(update, total, warmup, decay, minimum):
    if update < warmup:
        return update / warmup
    start = total - decay
    if update <= start:
        return 1.0
    return max(minimum, 1 - (1 - minimum) * (update - start) / decay)


def is_bridge_checkpoint(config, checkpoint):
    bridge = config.get("bridge_checkpoint")
    return bool(checkpoint and bridge and Path(checkpoint).resolve() == Path(bridge).resolve())


class FullFinetuneTrainer:
    """Mixin: only the explicit LoRA-to-full bridge skips optimizer/model restore."""

    def _is_bridge(self, checkpoint):
        return is_bridge_checkpoint(self.full_config, checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if not self._is_bridge(resume_from_checkpoint):
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint, *args, **kwargs):
        if not self._is_bridge(checkpoint):
            return super()._load_optimizer_and_scheduler(checkpoint, *args, **kwargs)

    def create_optimizer(self):
        import torch

        if self.optimizer is not None:
            return self.optimizer
        decay_names = set(self.get_decay_parameter_names(self.model))
        grouped = defaultdict(list)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                grouped[(parameter_group(name) == "vision", name in decay_names,
                         hasattr(parameter, "to_local"))].append(parameter)
        groups = [{"params": parameters,
                   "lr": self.args.learning_rate * (self.full_config["vision_lr_ratio"] if vision else 1),
                   "weight_decay": self.args.weight_decay if decay else 0}
                  for (vision, decay, sharded), parameters in sorted(grouped.items())]
        self.optimizer = torch.optim.AdamW(
            groups, betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon, foreach=False,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        from functools import partial
        from torch.optim.lr_scheduler import LambdaLR

        if self.lr_scheduler is None:
            config = self.full_config
            factor = partial(lr_factor, total=config["phase_total_updates"],
                             warmup=config["phase_warmup_updates"], decay=config["phase_decay_updates"],
                             minimum=config["phase_min_lr_ratio"])
            self.lr_scheduler = LambdaLR(optimizer or self.optimizer, factor)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is None:
            raise ValueError("Full-phase loss requires the global accumulated target-token count")
        if not hasattr(self, "loss_count_audited"):
            report = {"global_step": self.state.global_step + 1,
                      "local_microbatch_target_tokens": int(inputs["labels"][..., 1:].ne(-100).sum()),
                      "global_accumulated_target_tokens": int(num_items_in_batch),
                      "world_size": self.args.world_size,
                      "gradient_accumulation_steps": self.args.gradient_accumulation_steps}
            path = Path(self.args.output_dir) / f"loss-count-rank-{self.args.process_index}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            self.loss_count_audited = True
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            directory = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            (directory / "full-phase.json").write_text(json.dumps(self.full_config, indent=2) + "\n")


def make_full_callback(trainer, config, output):
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback

    class FullPhaseEvidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if not trainer.model_accepts_loss_kwargs:
                raise ValueError("Trainer has disabled global token normalization")

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if hasattr(self, "checked"):
                return
            counts = torch.zeros(5, dtype=torch.float64, device=trainer.accelerator.device)
            groups = ("vision", "merger", "embedding", "head", "language")
            for name, parameter in trainer.model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Nonfinite full-parameter gradient: {name}")
                norm = torch.linalg.vector_norm(gradient.detach().float()).double()
                counts[groups.index(parameter_group(name))] += norm.to(counts.device).square()
            if dist.is_initialized():
                dist.all_reduce(counts)
            if not counts.gt(0).all():
                raise RuntimeError(f"Missing full-parameter gradients: {dict(zip(groups, counts.tolist()))}")
            self.checked = True
            if state.is_world_process_zero:
                report = {"global_step": state.global_step + 1,
                          "gradient_squared_sums": dict(zip(groups, counts.tolist())),
                          "master_dtypes": sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}),
                          "scope": "all ranks, after accumulation and global clipping"}
                (output / "full-gradient-audit.json").write_text(json.dumps(report, indent=2) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            update = state.global_step - config["phase_start_step"]
            milestones = {state.max_steps}
            if config.get("bridge_checkpoint"):
                milestones.update((231, 462, 693))
            control.should_save = update > 0 and (update % config["save_steps"] == 0
                                                       or state.global_step in milestones)

    return FullPhaseEvidence()
