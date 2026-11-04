"""Train reconstruction, visual reflection, or SFT from a YAML recipe.

Usage: python -m avtr.train --config scripts/configs/stage1.yaml [key.path=value ...]
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import torch
import yaml

from . import fileio
from .fla_patches import apply_fla_autotune_tile_cap, apply_fla_bwd_patch

logger = logging.getLogger("avtr.train")

_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^${}]*))?\}")
_UNSET_RE = re.compile(r"\0(\w+)\0")


def _env(m: re.Match) -> str:
    value = os.environ.get(m.group(1))
    if m.group(2) is not None:  # ${VAR:-default}: unset or empty takes the default, as in the shell
        return value or m.group(2)
    # An unset ${VAR} becomes a marker that fails only if it survives: an enclosing ${A:-...} with A
    # set drops it.
    return f"\0{m.group(1)}\0" if value is None else value


def _expand(value, key: str = ""):
    if isinstance(value, str) and "${" in value:
        for _ in range(5):  # innermost first, so defaults may reference other variables
            new = _ENV_RE.sub(_env, value)
            if new == value:
                break
            value = new
        unset = _UNSET_RE.search(value)
        if unset:
            raise ValueError(f"{key}: environment variable {unset.group(1)} is not set "
                             f"(export it, or override {key}=... on the command line)")
        return yaml.safe_load(value) if value.strip() else None  # a numeric string -> int, "" -> None
    if isinstance(value, dict):
        return {k: _expand(v, f"{key}.{k}" if key else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, f"{key}.{v['name']}" if isinstance(v, dict) and "name" in v else f"{key}[{i}]")
                for i, v in enumerate(value)]
    return value


def _set(cfg, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(node, list):
            match = [e for e in node if isinstance(e, dict) and e.get("name") == k]
            if not match:
                raise KeyError(f"{dotted}: no list entry named {k!r}")
            if last:
                raise KeyError(f"{dotted}: cannot replace a whole list entry")
            node = match[0]
        elif last:
            node[k] = value
        else:
            node = node.setdefault(k, {})


def load_config(path: str, overrides: list[str]) -> dict:
    with fileio.open_file(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override {item!r} is not key.path=value")
        _set(cfg, key.strip(), yaml.safe_load(raw) if raw.strip() else "")
    return _expand(cfg)


def stage_local(path: str | None) -> str | None:
    """Remote model / checkpoint dirs are copied to node-local disk once per node: the local ranks
    take turns under a file lock, and every rank after the first finds the finished copy."""
    if not path or not fileio.is_remote(path):
        return path
    import fcntl
    import hashlib

    key = hashlib.sha1(path.encode()).hexdigest()[:16]
    with open(os.path.join("/tmp", f"avtr_stage_{os.getuid()}_{key}.lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fileio.local_path(path)


def main() -> None:
    from transformers import AutoProcessor, TrainingArguments, set_seed

    from .data import build_data
    from .model import PARAM_GROUPS, LatentQueryModel, diffusers_root
    from .trainer import LatentQueryTrainer, VizCallback, list_checkpoints

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(level=logging.INFO if int(os.environ.get("RANK", "0")) == 0 else logging.WARNING)
    m, t = cfg["model"], dict(cfg["train"])
    rank0 = int(os.environ.get("RANK", "0")) == 0

    set_seed(cfg["seed"])
    # fla tile caps must precede the first GatedDeltaNet backward (and so from_pretrained).
    apply_fla_bwd_patch()
    apply_fla_autotune_tile_cap()

    backbone = stage_local(m["backbone"])
    init_from = stage_local(m.get("init_from"))
    if init_from and not os.path.isdir(init_from):
        raise FileNotFoundError(f"model.init_from {init_from} is not a directory")
    sana = stage_local(m.get("sana"))
    if not sana and init_from:
        sana = diffusers_root(init_from)
        if not os.path.isdir(os.path.join(sana, "transformer")):
            raise ValueError(f"model.init_from {init_from} has no SANA pipeline (transformer/ or "
                             "diffusion/transformer/): set model.sana")
    if not sana:
        raise ValueError("model.sana (or model.init_from carrying a SANA pipeline) is required")
    if not os.path.isdir(sana):  # a hub id: fetch only the pipeline parts SANA needs
        from huggingface_hub import snapshot_download

        sana = snapshot_download(sana, allow_patterns=["model_index.json", "scheduler/*", "vae/*", "transformer/*"])
    attn = m.get("attn_implementation", "flash_attention_2")
    if attn.startswith("flash_attention"):
        import importlib.util

        attn = attn if importlib.util.find_spec("flash_attn") is not None else "sdpa"
    dtype = torch.bfloat16 if t.get("bf16") else torch.float32

    processor = AutoProcessor.from_pretrained(backbone)
    model = LatentQueryModel(
        backbone, sana, num_queries=m["num_queries"], connector_layers=m["connector_layers"],
        edit_cond=m.get("edit_cond", False), edit_source_dropout=m.get("edit_source_dropout", 0.0),
        layer_agg=m.get("layer_agg", "last"), ce_loss_weight=m.get("ce_loss_weight", 0.0),
        gen_loss_weight=m.get("gen_loss_weight", 1.0), dtype=dtype, attn_implementation=attn)
    if cfg["data"]["kind"] == "imagine":
        model.register_tokens(processor.tokenizer, markers=bool(m.get("markers", False)))
    if init_from:
        model.load_checkpoint(init_from)
    model.cast_small_params(fp32=bool(m.get("fp32_latent_query", False)))
    if t.get("fsdp") and m.get("fp32_latent_query"):
        model.upcast_fsdp_root()
    group_lrs = {g: float(lr) for g, lr in cfg["trainable"].items()}
    unknown = set(group_lrs) - set(PARAM_GROUPS)
    if unknown:
        raise ValueError(f"unknown trainable groups {sorted(unknown)}; known: {sorted(PARAM_GROUPS)}")
    counts = model.set_trainable(group_lrs)
    if rank0:
        logger.info("trainable params: %s (total %.1fM)", {g: f"{n / 1e6:.2f}M" for g, n in counts.items()},
                    sum(counts.values()) / 1e6)
    model.config.use_cache = False

    resume = cfg.get("resume_from") or (list_checkpoints(t["output_dir"]) or [None])[-1]
    if resume:
        resume = stage_local(resume)
        if not os.path.isfile(os.path.join(resume, "latent_query.pt")):
            raise FileNotFoundError(f"resume_from {resume} is not a checkpoint-N dir (no latent_query.pt)")
        decoder_trains = "decoder" in group_lrs
        if decoder_trains and not os.path.isdir(os.path.join(resume, "diffusion", "transformer")):
            raise RuntimeError(f"{resume} has no diffusion/transformer but the decoder trains")
        model.load_checkpoint(resume, backbone=True)
        logger.info("resuming from %s", resume)

    dataset, collator = build_data(cfg, processor, model)
    t.update(learning_rate=next(iter(group_lrs.values())), remove_unused_columns=False, ignore_data_skip=True,
             seed=cfg["seed"], gradient_checkpointing_kwargs={"use_reentrant": False})
    training_args = TrainingArguments(**t)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainer = LatentQueryTrainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=collator, processing_class=processor,
        group_lrs=group_lrs, decay_all_params=bool(cfg.get("decay_all_params", False)),
        viz_every=int(cfg.get("viz_every", 0)), mirror_dir=cfg.get("output_mirror") or None)
    if cfg.get("viz_every", 0) > 0:
        trainer.add_callback(VizCallback(trainer, int(cfg["viz_every"]), int(cfg.get("viz_steps", 20)),
                                         m["resolution"]))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()


if __name__ == "__main__":
    main()
