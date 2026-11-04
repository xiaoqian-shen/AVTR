"""Hugging Face Trainer with per-group learning rates and AVTR checkpoint export.

Includes FP32 FSDP gradient reduction, non-finite-gradient checks, loss logging,
and optional visualizations. Checkpoint formats are documented in docs/TRAINING.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names

from . import fileio
from .model import group_of

logger = logging.getLogger(__name__)

_FIRE_KEYS = ("fire_pos_sum", "fire_pos_cnt", "fire_neg_sum", "fire_neg_cnt", "fire_correct")
# Fixed accumulator set: every rank gathers the same tensors at each log step.
_ACC_KEYS = ("diff_num", "diff_den", "ce_sum", "ce_cnt", *_FIRE_KEYS)
_GEN_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw",
             "pixel_values_videos", "video_grid_thw", "source_pixel_values")


def _local_finite(grad: torch.Tensor) -> bool:
    local = grad.to_local() if hasattr(grad, "to_local") else grad
    return bool(torch.isfinite(local).all())


class LatentQueryTrainer(Trainer):
    def __init__(self, *args, group_lrs: dict[str, float], decay_all_params: bool = False,
                 viz_every: int = 0, mirror_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_lrs = group_lrs
        self.decay_all_params = decay_all_params
        self.viz_every = viz_every
        self.mirror_dir = mirror_dir
        self.last_batch: dict | None = None
        self._nan_skips = 0
        self._acc: dict[str, torch.Tensor] = {}
        self._configure_fsdp()

    # -- FSDP --------------------------------------------------------------------------------
    def _configure_fsdp(self) -> None:
        """Reduce-scatter gradients in fp32 (bf16 reduction NaNs Qwen3.5's GatedDeltaNet on the first step) and
        use_orig_params (flat params mix frozen and trainable tensors)."""
        plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        if plugin is None:
            return
        plugin.use_orig_params = True
        policy = getattr(plugin, "mixed_precision_policy", None)
        param_dtype = getattr(policy, "param_dtype", None)
        if param_dtype not in (torch.bfloat16, torch.float16):
            param_dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else None
        if param_dtype is None or getattr(policy, "reduce_dtype", None) == torch.float32:
            return
        if getattr(plugin, "fsdp_version", 1) == 2:
            from torch.distributed.fsdp import MixedPrecisionPolicy

            plugin.mixed_precision_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32,
                                                                 output_dtype=param_dtype)
        else:
            from torch.distributed.fsdp import MixedPrecision

            plugin.mixed_precision_policy = MixedPrecision(param_dtype=param_dtype, reduce_dtype=torch.float32,
                                                           buffer_dtype=torch.float32)

    def get_train_dataloader(self) -> DataLoader:
        # The stream shards itself over ranks and workers: no IterableDatasetShard / accelerate prepare.
        return DataLoader(self.train_dataset, batch_size=self._train_batch_size, collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers, pin_memory=True,
                          persistent_workers=(self.args.dataloader_persistent_workers
                                              and self.args.dataloader_num_workers > 0))

    # -- optimizer ---------------------------------------------------------------------------
    def create_optimizer(self, model=None):
        """AdamW with a (decay, no-decay) pair per trainable group at that group's lr. No decay on biases
        and LayerNorm weights unless ``decay_all_params``."""
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model if model is None else model
        decay = set(n for n in get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS) if "bias" not in n)
        groups = []
        for g, lr in self.group_lrs.items():
            members = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad and group_of(_strip(n)) == g]
            for wd_on in (True, False):
                params = [p for n, p in members if (self.decay_all_params or n in decay) == wd_on]
                if params:
                    groups.append({"params": params, "lr": lr,
                                   "weight_decay": self.args.weight_decay if wd_on else 0.0})
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = cls(groups, **kwargs)
        return self.optimizer

    # -- step --------------------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        self._accumulate(outputs, inputs)
        if self.viz_every > 0:
            self.last_batch = {k: v.detach().cpu() for k, v in inputs.items() if torch.is_tensor(v)}
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Drop an update whose gradients went non-finite (grad clipping does not catch NaN, and one NaN
        step poisons Adam's moments for good). The decision is all-reduced so every rank agrees."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        bad = next((n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None and not _local_finite(p.grad)), None)
        flag = torch.tensor(float(bad is not None), device=self.args.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            model.zero_grad(set_to_none=True)
            self._nan_skips += 1
            logger.error("non-finite gradient: skipped update #%d at step %d (%s)", self._nan_skips,
                         self.state.global_step, bad or "flagged by a peer rank")
        return loss

    def _accumulate(self, outputs: dict, inputs: dict) -> None:
        def add(key, value):
            self._acc[key] = self._acc.get(key, torch.zeros((), device=self.args.device)) + value.detach().float()

        tgt = inputs.get("target_pixel_values")
        # Weighted by the real block count so target-less micro-batches do not dilute the curve.
        n = outputs["diffusion_loss"].new_tensor(float(tgt.shape[0]) if tgt is not None else 1.0)
        add("diff_num", outputs["diffusion_loss"] * n)
        add("diff_den", n)
        if "ce_loss" in outputs:
            add("ce_sum", outputs["ce_loss"])
            add("ce_cnt", torch.ones((), device=self.args.device))
        for k in _FIRE_KEYS:
            if k in outputs:
                add(k, outputs[k])

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            zero = torch.zeros((), device=self.args.device)
            s = {k: float(self.accelerator.gather(self._acc.get(k, zero).reshape(1)).sum()) for k in _ACC_KEYS}
            self._acc = {}
            logs["diffusion_loss"] = s["diff_num"] / max(s["diff_den"], 1.0)
            if s["ce_cnt"] > 0:
                logs["ce_loss"] = s["ce_sum"] / max(s["ce_cnt"], 1.0)
            n_fire = s["fire_pos_cnt"] + s["fire_neg_cnt"]
            if n_fire > 0:
                logs["fire_acc"] = s["fire_correct"] / n_fire
                if s["fire_pos_cnt"] > 0:
                    logs["fire_margin_pos"] = s["fire_pos_sum"] / s["fire_pos_cnt"]
                if s["fire_neg_cnt"] > 0:
                    logs["fire_margin_neg"] = s["fire_neg_sum"] / s["fire_neg_cnt"]
            logs["nan_skips"] = float(self._nan_skips)
        return super().log(logs, *args, **kwargs)

    # -- checkpoints -------------------------------------------------------------------------
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Weights are restored before the FSDP wrap (train.py); HF restores optimizer/scheduler/RNG/step.
        pass

    def _load_optimizer_and_scheduler(self, checkpoint):
        # accelerate upcasts trainable FSDP flat params to fp32 after the wrap but leaves the
        # use_orig_params views on the old bf16 storage until FSDP's lazy init: load_state_dict would
        # cast the Adam moments to bf16 and the first step fails on mixed dtypes. Rebuild the views.
        if checkpoint is not None and self.is_fsdp_enabled:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            wrapped = self.model_wrapped if self.model_wrapped is not None else self.model
            for module in FSDP.fsdp_modules(wrapped):
                if module._handle is not None:
                    module._handle._use_sharded_views()
        super()._load_optimizer_and_scheduler(checkpoint)

    def _save_checkpoint(self, model, trial):
        out = os.path.join(self._get_output_dir(trial=trial), f"checkpoint-{self.state.global_step}")
        os.makedirs(out, exist_ok=True)
        save_checkpoint(self, out)
        self._save_optimizer_and_scheduler(out)
        self._save_rng_state(out)
        for p in os.listdir(out):  # HF's FSDP model dump: weights are in our own layout already
            if p.startswith("pytorch_model_fsdp"):
                fileio.remove(os.path.join(out, p))
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(out, "trainer_state.json"))
            if self.mirror_dir:
                fileio.copy(out, f"{self.mirror_dir.rstrip('/')}/{os.path.basename(out)}")
        rotate_checkpoints(self._get_output_dir(trial=trial), self.args.save_total_limit)


def _strip(name: str) -> str:
    """Parameter name without FSDP / compile wrapper prefixes."""
    name = re.sub(r"(_fsdp_wrapped_module|_checkpoint_wrapped_module|_orig_mod)\.", "", name)
    return name[len("module."):] if name.startswith("module.") else name


def save_checkpoint(trainer: Trainer, out: str) -> None:
    full_sd = trainer.accelerator.get_state_dict(trainer.model)  # collective under FSDP
    if not trainer.is_world_process_zero():
        return
    model = trainer.accelerator.unwrap_model(trainer.model)
    full_sd = {_strip(k): v for k, v in full_sd.items()}
    torch.save(model.latent_query_state(full_sd), os.path.join(out, "latent_query.pt"))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(out)
    if any(p.requires_grad for p in model.backbone.parameters()):
        bb = {k[len("backbone."):]: v for k, v in full_sd.items() if k.startswith("backbone.")}
        # under the model's own names: by default transformers saves Qwen3.5's model.visual.* as
        # model.language_model.visual.*, which resume would skip and vLLM cannot load
        model.backbone.save_pretrained(out, state_dict=bb, safe_serialization=True, save_original_format=False)
    if any(p.requires_grad for p in model.decoder.transformer.parameters()):
        _save_diffusion(model.sana_dir, out, {k[len("decoder.transformer."):]: v.contiguous()
                                              for k, v in full_sd.items() if k.startswith("decoder.transformer.")})


def _save_diffusion(base: str, out: str, tf_sd: dict) -> None:
    """A self-contained ``diffusion/`` (trained transformer + base vae / scheduler / model_index),
    built in a temp dir and renamed so a partial save is never picked up."""
    from safetensors.torch import save_file

    final, tmp = os.path.join(out, "diffusion"), os.path.join(out, "diffusion.incomplete")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "transformer"))
    with open(os.path.join(base, "transformer", "config.json")) as fh:
        cfg = json.load(fh)
    cfg["in_channels"] = int(tf_sd["patch_embed.proj.weight"].shape[1])  # widened by edit_cond
    with open(os.path.join(tmp, "transformer", "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    save_file(tf_sd, os.path.join(tmp, "transformer", "diffusion_pytorch_model.safetensors"))
    for name in ("vae", "scheduler"):
        shutil.copytree(os.path.join(base, name), os.path.join(tmp, name), dirs_exist_ok=True)
    if os.path.isfile(os.path.join(base, "model_index.json")):
        shutil.copy2(os.path.join(base, "model_index.json"), os.path.join(tmp, "model_index.json"))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)


def list_checkpoints(output_dir: str) -> list[str]:
    """Complete ``checkpoint-N`` dirs (``trainer_state.json`` is written last), oldest first."""
    if not os.path.isdir(output_dir):
        return []
    matches = (re.fullmatch(r"checkpoint-(\d+)", n) for n in os.listdir(output_dir))
    found = [(int(m.group(1)), m.group(0)) for m in matches
             if m and os.path.isfile(os.path.join(output_dir, m.group(0), "trainer_state.json"))]
    return [os.path.join(output_dir, n) for _, n in sorted(found)]


def rotate_checkpoints(output_dir: str, limit: int | None) -> None:
    """Keep the newest ``limit`` checkpoints on this node's disk (every rank: each node has its own)."""
    if not limit or limit <= 0:
        return
    ckpts = sorted((p for p in os.listdir(output_dir) if re.fullmatch(r"checkpoint-\d+", p)),
                   key=lambda p: int(p.split("-")[1]))
    for stale in ckpts[:-limit]:
        shutil.rmtree(os.path.join(output_dir, stale), ignore_errors=True)


class VizCallback(TrainerCallback):
    """Every ``every`` steps render the last micro-batch's blocks and write ``viz/step_N.png``
    ([target | generated] per row). Every rank runs the sampling (FSDP gathers in lockstep)."""

    def __init__(self, trainer: LatentQueryTrainer, every: int, steps: int, resolution: int, max_rows: int = 4) -> None:
        self.trainer, self.every, self.steps, self.res, self.max_rows = trainer, every, steps, resolution, max_rows

    def on_step_end(self, args, state, control, **kwargs):
        batch = self.trainer.last_batch
        if self.every <= 0 or state.global_step % self.every or batch is None:
            return control
        dev = self.trainer.args.device
        model = self.trainer.model_wrapped if self.trainer.model_wrapped is not None else self.trainer.model
        with torch.no_grad():
            gen = model(mode="viz", height=self.res, width=self.res, num_inference_steps=self.steps,
                                     **{k: batch[k].to(dev) for k in _GEN_KEYS if k in batch})
        tgt = batch.get("target_pixel_values")
        if self.trainer.is_world_process_zero() and tgt is not None and tgt.shape[0] > 0:
            from PIL import Image

            rows = min(self.max_rows, tgt.shape[0], gen.shape[0])
            pair = torch.cat([torch.cat([tgt[i].float(), gen[i].float().cpu()], dim=2) for i in range(rows)], dim=1)
            arr = ((pair.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            os.makedirs(os.path.join(args.output_dir, "viz"), exist_ok=True)
            Image.fromarray(arr).save(os.path.join(args.output_dir, "viz", f"step_{state.global_step}.png"))
        return control
