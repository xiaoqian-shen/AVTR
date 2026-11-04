"""Qwen3.5 with learnable latent queries, a connector, and a SANA decoder.

Reconstruction appends queries after the prompt; visual reflection and SFT place them
inside the assistant turn. Parameter groups select which components train.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)

# SANA wants its caption conditioning at a fixed RMS magnitude: the connector ends in an RMSNorm
# whose weight is initialised to this value. A unit-magnitude condition under-drives SANA.
COND_SCALE = math.sqrt(5.5)

IMAGINE_OPEN = "<imagine>"
IMAGINE_CLOSE = "</imagine>"
# Slot placeholder inside a block. It only lets the model LOCATE the query positions by id: its
# embedding is always overwritten by the learnable query vectors.
LATENT_QUERY_TOKEN = "<latent_query>"

# Soft directive prepended to the user question of every stage-1/2 row, so the block is an expected,
# conditional continuation. Eval prompts must carry a byte-identical copy.
IMAGINE_DIRECTIVE = (
    "If a visual clue would help you answer, generate it FIRST -- before you think -- as abstract "
    "visual tokens inside <imagine>...</imagine>, then think and answer.\n"
)

# Supervised positions per lm_head + CE slice (peak logits = chunk x vocab).
_CE_CHUNK = 1024

# Named trainable groups -> parameter-name predicate (first match wins).
PARAM_GROUPS = {
    "vision": lambda n: n.startswith("backbone.model.visual."),
    "language": lambda n: n.startswith("backbone."),
    "latent_queries": lambda n: n == "query_tokens",
    "markers": lambda n: n in ("imagine_embeds", "imagine_lm_head"),
    "connector": lambda n: n.startswith("connector."),
    "decoder": lambda n: n.startswith("decoder.transformer."),
}


def group_of(name: str) -> str | None:
    return next((g for g, match in PARAM_GROUPS.items() if match(name)), None)


# ---------------------------------------------------------------------------------------------
# GatedDeltaNet in fp32
# ---------------------------------------------------------------------------------------------
def _fp32_kernel(fn):
    """Upcast bf16 / fp16 tensor args to fp32, run fn, cast fp32 outputs back to the input dtype."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        dtypes = [x.dtype for x in (*args, *kwargs.values()) if torch.is_tensor(x)]
        low = next((d for d in dtypes if d in (torch.bfloat16, torch.float16)), None)
        if low is None:
            return fn(*args, **kwargs)

        def up(x):
            return x.float() if torch.is_tensor(x) and x.dtype in (torch.bfloat16, torch.float16) else x

        def down(x):
            return x.to(low) if torch.is_tensor(x) and x.dtype == torch.float32 else x

        out = fn(*(up(a) for a in args), **{k: up(v) for k, v in kwargs.items()})
        return tuple(down(o) for o in out) if isinstance(out, tuple) else down(out)

    wrapper.__fp32_wrapped__ = True
    return wrapper


def force_fp32_gated_delta_rule(model: nn.Module) -> int:
    """Run Qwen3.5's delta-rule recurrence in fp32 (q/k/v/beta upcast). In bf16 its backward NaNs on
    the first step once gradients flow through the backbone to the query tokens."""
    n = 0
    for module in model.modules():
        for attr in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
            fn = getattr(module, attr, None)
            if callable(fn) and not getattr(fn, "__fp32_wrapped__", False):
                setattr(module, attr, _fp32_kernel(fn))
                n += 1
    return n


# ---------------------------------------------------------------------------------------------
# Connector: in-projection -> bidirectional Qwen2 encoder (RoPE + QK-norm) -> MLP -> RMSNorm
# ---------------------------------------------------------------------------------------------
class _HeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


def _encoder_classes():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
        Qwen2PreTrainedModel,
        Qwen2RMSNorm,
        Qwen2RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    class Qwen2BidirectionalSdpaAttention(Qwen2Attention):
        def __init__(self, config, layer_idx=None) -> None:
            super().__init__(config, layer_idx)
            self.is_causal = False
            head_dim = config.hidden_size // config.num_attention_heads
            self.q_norm = _HeadRMSNorm(head_dim, config.num_attention_heads)
            self.k_norm = _HeadRMSNorm(head_dim, config.num_key_value_heads)

        def forward(self, hidden_states, position_embeddings):
            shape = (*hidden_states.shape[:-1], -1, self.head_dim)
            q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            q, k = self.q_norm(q), self.k_norm(k)
            out, _ = sdpa_attention_forward(
                self, q, k, v, attention_mask=None,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling, is_causal=False,
            )
            return self.o_proj(out.reshape(*hidden_states.shape[:-1], -1).contiguous())

    class Qwen2EncoderLayer(nn.Module):
        def __init__(self, config, layer_idx: int) -> None:
            super().__init__()
            self.self_attn = Qwen2BidirectionalSdpaAttention(config, layer_idx)
            self.mlp = Qwen2MLP(config)
            self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(self, hidden_states, position_embeddings):
            h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
            return h + self.mlp(self.post_attention_layernorm(h))

    class Qwen2Encoder(Qwen2PreTrainedModel):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.layers = nn.ModuleList(Qwen2EncoderLayer(config, i) for i in range(config.num_hidden_layers))
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)
            self.post_init()

        def _init_weights(self, module) -> None:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

        def forward(self, x):
            pos = self.rotary_emb(x, torch.arange(x.shape[1], device=x.device).unsqueeze(0))
            for layer in self.layers:
                if self.training:  # always checkpointed (non-reentrant: FSDP-safe)
                    x = torch.utils.checkpoint.checkpoint(layer.__call__, x, pos, use_reentrant=False)
                else:
                    x = layer(x, pos)
            return x

    return Qwen2Encoder


class Connector(nn.Module):
    """(B, Q, d_in) query hidden states -> (B, Q, d_cond) SANA conditioning. The encoder runs at
    d_cond (in-projection first) so it stays small behind a wide backbone."""

    def __init__(self, d_in: int, d_cond: int, num_layers: int, out_scale: float) -> None:
        super().__init__()
        from diffusers.models.normalization import RMSNorm
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

        heads = max(1, d_cond // 64)
        self.in_proj = nn.Linear(d_in, d_cond)
        self.encoder = _encoder_classes()(Qwen2Config(
            hidden_size=d_cond, intermediate_size=d_cond * 4, num_hidden_layers=num_layers,
            num_attention_heads=heads, num_key_value_heads=heads, initializer_range=0.014,
            use_cache=False,
        ))
        self.head = nn.Sequential(nn.Linear(d_cond, d_cond), nn.GELU(approximate="tanh"), nn.Linear(d_cond, d_cond))
        self.out_norm = RMSNorm(d_cond, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            self.out_norm.weight.fill_(out_scale)

    def forward(self, x):
        return self.out_norm(self.head(self.encoder(self.in_proj(x))))


# ---------------------------------------------------------------------------------------------
# SANA decoder
# ---------------------------------------------------------------------------------------------
def _safetensors_in_channels(transformer_dir: str) -> int | None:
    """patch_embed in-channels of the saved weights, read from the safetensors header only."""
    path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    try:
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        return int(header["patch_embed.proj.weight"]["shape"][1])
    except (KeyError, IndexError, ValueError, OSError, struct.error):
        return None


class SanaDecoder(nn.Module):
    """SANA transformer (trainable or frozen) + frozen DC-AE VAE. Sana consumes the raw scheduler
    timesteps; they are NOT rescaled to the unit interval.

    ``edit_cond`` widens patch_embed to take [noised target latent, source latent] (channel concat).
    The source half is zero-initialised, so widening is an exact no-op at init. A missing source is a
    BLACK image, encoded every step so the VAE runs identically on every rank."""

    def __init__(self, sana_path: str, dtype: torch.dtype, edit_cond: bool) -> None:
        super().__init__()
        from diffusers import (
            AutoencoderDC,
            DPMSolverMultistepScheduler,
            FlowMatchEulerDiscreteScheduler,
            SanaTransformer2DModel,
        )

        self.transformer = SanaTransformer2DModel.from_pretrained(sana_path, subfolder="transformer", torch_dtype=dtype)
        self.vae = AutoencoderDC.from_pretrained(sana_path, subfolder="vae", torch_dtype=dtype)
        self.vae.requires_grad_(False).eval()
        # Training keeps the full FlowMatch schedule; sampling uses DPM-Solver++.
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sana_path, subfolder="scheduler")
        self.sample_scheduler = DPMSolverMultistepScheduler.from_pretrained(sana_path, subfolder="scheduler")
        cfg = self.transformer.config
        self.cond_dim = cfg.caption_channels
        # Latent channels come from the VAE, never from the (possibly widened) transformer in_channels.
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", None) or cfg.out_channels)
        self.vae_f = int(getattr(self.vae, "spatial_compression_ratio", None) or 32)
        self.vae_scaling = float(self.vae.config.scaling_factor)
        shift = getattr(self.vae.config, "shift_factor", None)
        self.vae_shift = float(shift) if shift is not None else None
        self.edit_cond = edit_cond
        if edit_cond and self.transformer.patch_embed.proj.in_channels < 2 * self.latent_channels:
            self._widen_patch_embed()
        self.transformer = self.transformer.to(dtype)

    def _widen_patch_embed(self) -> None:
        conv = self.transformer.patch_embed.proj
        c_in = conv.in_channels
        new = nn.Conv2d(2 * c_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                        bias=conv.bias is not None).to(dtype=conv.weight.dtype, device=conv.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :c_in].copy_(conv.weight)
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        self.transformer.patch_embed.proj = new
        self.transformer.register_to_config(in_channels=2 * c_in)

    def load_transformer(self, transformer_dir: str, dtype: torch.dtype):
        """Load saved transformer weights. A checkpoint with the original patch_embed width loads into a
        widened decoder (the source half stays zero); in_channels is taken from the weights, not config.json."""
        from diffusers import SanaTransformer2DModel

        in_ch = _safetensors_in_channels(transformer_dir)
        loaded = SanaTransformer2DModel.from_pretrained(
            transformer_dir, torch_dtype=dtype, **({"in_channels": in_ch} if in_ch else {}))
        sd = dict(loaded.state_dict())
        key, model_w = "patch_embed.proj.weight", self.transformer.patch_embed.proj.weight
        if key in sd and sd[key].shape[1] != model_w.shape[1]:
            aligned = torch.zeros_like(model_w)
            c = min(sd[key].shape[1], model_w.shape[1])
            aligned[:, :c].copy_(sd[key][:, :c].to(aligned.dtype))
            sd[key] = aligned
        return self.transformer.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        lat = self.vae.encode(pixels.to(next(self.vae.parameters()).dtype)).latent
        if self.vae_shift is not None:
            lat = lat - self.vae_shift
        return lat * self.vae_scaling

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        lat = latents.to(next(self.vae.parameters()).dtype) / self.vae_scaling
        if self.vae_shift is not None:
            lat = lat + self.vae_shift
        return self.vae.decode(lat).sample.clamp(-1, 1)

    def _sigmas(self, timesteps, device, n_dim, dtype):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule = self.noise_scheduler.timesteps.to(device)
        idx = [(schedule == t).nonzero().item() for t in timesteps.to(device)]
        sigma = sigmas[idx].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def diffusion_loss(self, cond, pixels, sample_weight=None, source=None, source_dropout=0.0):
        """Rectified-flow loss with uniform timestep sampling and uniform weighting. ``sample_weight``
        masks per-sample losses; the transformer still runs on the full batch (FSDP parity)."""
        from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

        x0 = self.encode(pixels)
        bsz = x0.shape[0]
        noise = torch.randn_like(x0)
        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=x0.device)
        sigmas = self._sigmas(timesteps, x0.device, x0.ndim, x0.dtype)
        hidden = (1.0 - sigmas) * x0 + sigmas * noise
        target = noise - x0
        if self.edit_cond:
            x_src = self.encode(source if source is not None else torch.zeros_like(pixels))
            if source_dropout > 0.0:
                keep = (torch.rand(bsz, device=x0.device) >= source_dropout).to(x_src.dtype)
                x_src = x_src * keep.view(bsz, *([1] * (x_src.ndim - 1)))
            hidden = torch.cat([hidden, x_src.to(hidden.dtype)], dim=1)
        pred = self.transformer(
            hidden_states=hidden, timestep=timesteps, encoder_hidden_states=cond,
            encoder_attention_mask=torch.ones(cond.shape[:2], dtype=torch.long, device=cond.device),
        ).sample
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="uniform", sigmas=sigmas)
        per_sample = torch.mean((weighting.float() * (pred.float() - target.float()) ** 2).reshape(bsz, -1), 1)
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight.to(per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, cond, height, width, num_inference_steps=20, source=None):
        device = next(self.vae.parameters()).device
        model_dtype = next(self.transformer.parameters()).dtype
        b = cond.shape[0]
        latents = torch.randn(b, self.latent_channels, height // self.vae_f, width // self.vae_f,
                              device=device, dtype=torch.float32)
        self.sample_scheduler.set_timesteps(num_inference_steps, device=device)
        mask = torch.ones(cond.shape[:2], dtype=torch.long, device=cond.device)
        x_src = None
        if self.edit_cond:
            if source is None:
                source = torch.zeros(b, 3, height, width, device=device, dtype=model_dtype)
            x_src = self.encode(source[:b]).to(model_dtype)
        for t in self.sample_scheduler.timesteps:
            model_in = latents.to(model_dtype)
            if x_src is not None:
                model_in = torch.cat([model_in, x_src], dim=1)
            pred = self.transformer(hidden_states=model_in, timestep=t.unsqueeze(0).expand(b),
                                    encoder_hidden_states=cond, encoder_attention_mask=mask).sample.float()
            latents = self.sample_scheduler.step(pred, t, latents, return_dict=False)[0]
        return self.decode(latents)


def diffusers_root(ckpt_dir: str) -> str:
    """Directory holding ``transformer/`` + ``vae/``: the checkpoint root or its ``diffusion/``."""
    nested = os.path.join(ckpt_dir, "diffusion")
    if not os.path.isdir(os.path.join(ckpt_dir, "transformer")) and os.path.isdir(os.path.join(nested, "transformer")):
        return nested
    return ckpt_dir


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
class LatentQueryModel(nn.Module):
    # forward(**batch) would make HF assume the loss is normalised by num_items_in_batch and skip its
    # gradient-accumulation division (stages 1-2 then sum over micro-batches on recent transformers).
    accepts_loss_kwargs = False

    def __init__(
        self,
        backbone_path: str,
        sana_path: str,
        num_queries: int = 64,
        connector_layers: int = 6,
        edit_cond: bool = False,
        edit_source_dropout: float = 0.0,
        layer_agg: str = "last",
        ce_loss_weight: float = 0.0,
        gen_loss_weight: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(backbone_path)
        if config.model_type not in ("qwen3_5", "qwen3_5_moe"):
            raise ValueError(f"only Qwen3.5 backbones are supported, got {config.model_type!r}")
        self.backbone = AutoModelForImageTextToText.from_pretrained(
            backbone_path, config=config, torch_dtype=dtype, attn_implementation=attn_implementation)
        force_fp32_gated_delta_rule(self.backbone)
        self.backbone.requires_grad_(False)
        text_cfg = config.text_config
        # A benign text id for the appended query slots (stage 0) and the left padding.
        self.placeholder_id = 0
        for attr in ("pad_token_id", "eos_token_id"):
            v = getattr(text_cfg, attr, None) or getattr(config, attr, None)
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if isinstance(v, int):
                self.placeholder_id = v
                break
        self.num_queries = num_queries
        self.dtype_ = dtype
        self.layer_agg = layer_agg
        self.ce_loss_weight = float(ce_loss_weight)
        self.gen_loss_weight = float(gen_loss_weight)
        self.edit_source_dropout = float(edit_source_dropout) if edit_cond else 0.0
        # Construction order (query_tokens, decoder, connector) fixes the seeded random init.
        self.query_tokens = nn.Parameter(torch.randn(num_queries, text_cfg.hidden_size) * 0.02)
        self.decoder = SanaDecoder(sana_path, dtype=dtype, edit_cond=edit_cond)
        self.connector = Connector(text_cfg.hidden_size, self.decoder.cond_dim, connector_layers, COND_SCALE)
        self.latent_query_id = -1
        self.imagine_ids: list[int] = []
        self.imagine_embeds: nn.Parameter | None = None
        self.imagine_lm_head: nn.Parameter | None = None
        self.sana_dir = sana_path
        # FSDP auto-wrap classes: the backbone's own blocks plus connector / decoder blocks.
        names = list(getattr(self.backbone, "_no_split_modules", None) or [])
        for sub in (self.connector, self.decoder):
            for m in sub.modules():
                cls = type(m).__name__
                if cls not in names and cls.endswith(("Block", "Layer")) and any(True for _ in m.parameters()):
                    names.append(cls)
        self._no_split_modules = names

    # -- HF Trainer plumbing ---------------------------------------------------------------
    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        tf = self.decoder.transformer
        if any(p.requires_grad for p in tf.parameters()):
            tf.enable_gradient_checkpointing()

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        self.backbone.enable_input_require_grads()

    def set_trainable(self, groups) -> dict[str, int]:
        """requires_grad=True exactly for the named groups; returns trainable numel per group."""
        counts: dict[str, int] = {}
        for name, p in self.named_parameters():
            g = group_of(name)
            p.requires_grad_(g in groups)
            if g in groups:
                counts[g] = counts.get(g, 0) + p.numel()
        return counts

    # -- markers ----------------------------------------------------------------------------
    def register_tokens(self, tokenizer, markers: bool) -> None:
        """Register ``<latent_query>`` (and the markers). The embedding table only GROWS: Qwen3.5
        ships spare rows, so the new ids land in existing rows and embed_tokens / lm_head keep their
        base shape. The markers get small dedicated input/output vectors (``imagine_embeds`` /
        ``imagine_lm_head``), initialised to the table mean, that override their rows at forward."""
        tokens = [IMAGINE_OPEN, IMAGINE_CLOSE, LATENT_QUERY_TOKEN] if markers else [LATENT_QUERY_TOKEN]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        if len(tokenizer) > self.backbone.get_input_embeddings().weight.shape[0]:
            self.backbone.resize_token_embeddings(len(tokenizer))
        self.latent_query_id = int(tokenizer.convert_tokens_to_ids(LATENT_QUERY_TOKEN))
        if not markers:
            return
        self.imagine_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in (IMAGINE_OPEN, IMAGINE_CLOSE)]
        in_w = self.backbone.get_input_embeddings().weight.detach()
        out_w = self.backbone.get_output_embeddings().weight.detach()
        self.imagine_embeds = nn.Parameter(in_w.mean(0, keepdim=True).float().repeat(2, 1))
        self.imagine_lm_head = nn.Parameter(out_w.mean(0, keepdim=True).float().repeat(2, 1))

    def cast_small_params(self, fp32: bool) -> None:
        """The marker vectors are always cast to the compute dtype; queries and connector too unless
        ``fp32``, which keeps their fp32 master weights: at the small learning rates used here an AdamW
        step is below bf16's ulp and they would stop moving. (Under FSDP accelerate upcasts every
        trainable flat param anyway.)"""
        if not fp32:
            self.connector.to(self.dtype_)
            self.query_tokens.data = self.query_tokens.data.to(self.dtype_)
        for p in (self.imagine_embeds, self.imagine_lm_head):
            if p is not None:
                p.data = p.data.to(self.dtype_)

    def upcast_fsdp_root(self) -> None:
        """FSDP1 flattens each wrap unit into one single-dtype tensor, and every parameter outside the
        ``_no_split_modules`` blocks lands in the root unit: fp32 queries / connector next to bf16
        embeddings, vision patch-embed / merger, SANA outer layers and VAE convs fail the wrap ("Must
        flatten tensors with uniform dtype"). Upcast the leftovers; accelerate upcasts this trainable
        flat param to fp32 after the wrap anyway."""
        wrap = set(self._no_split_modules)
        inside = {id(p) for mod in self.modules() if type(mod).__name__ in wrap for p in mod.parameters()}
        for p in self.parameters():
            if id(p) not in inside and p.dtype != torch.float32:
                p.data = p.data.float()

    # -- backbone helpers -------------------------------------------------------------------
    def _embed_with_vision(self, input_ids, pixel_values=None, image_grid_thw=None,
                           pixel_values_videos=None, video_grid_thw=None):
        """Input embeddings with the image OR video features scattered in -> (embeds, is_image)."""
        inner = self.backbone.model
        embeds = inner.get_input_embeddings()(input_ids)
        is_image = pixel_values is not None
        if is_image:
            out = inner.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        else:
            out = inner.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        feats = getattr(out, "pooler_output", out)
        if isinstance(feats, (tuple, list)):
            feats = torch.cat(list(feats), dim=0)
        feats = feats.to(embeds.device, embeds.dtype)
        kw = {"image_features": feats} if is_image else {"video_features": feats}
        image_mask, video_mask = inner.get_placeholder_mask(input_ids, inputs_embeds=embeds, **kw)
        return embeds.masked_scatter(image_mask if is_image else video_mask, feats), is_image

    def _run_lm(self, input_ids, embeds, attention_mask, mm_token_type_ids, grid, is_image, hidden_states=False):
        """3D-mRoPE positions over the whole sequence, then the language model. Video grids are
        expanded to one row per temporal position; every non-visual slot is text."""
        inner = self.backbone.model
        if not is_image:
            frames = []
            for g in grid:
                per_frame = g.unsqueeze(0).repeat(int(g[0]), 1)
                per_frame[:, 0] = 1
                frames.append(per_frame)
            grid = torch.cat(frames, dim=0)
        position_ids = inner.compute_3d_position_ids(
            input_ids=input_ids, image_grid_thw=grid if is_image else None,
            video_grid_thw=None if is_image else grid, inputs_embeds=embeds,
            attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)
        return inner.language_model(input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
                                    inputs_embeds=embeds, output_hidden_states=hidden_states)

    def _cond(self, query_hidden):
        conn_dtype = next(self.connector.parameters()).dtype
        return self.connector(query_hidden.to(conn_dtype)).to(self.dtype_)

    # -- stage 0: appended queries ----------------------------------------------------------
    def _recon_query_hidden(self, input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw):
        """[image + caption (left-padded)][queries] -> query hidden states (B, Q, H). The backbone
        weights are frozen but the graph must reach the query tokens."""
        embeds, _ = self._embed_with_vision(input_ids, pixel_values, image_grid_thw)
        b, q = input_ids.shape[0], self.num_queries
        dev = input_ids.device
        embeds = torch.cat([embeds, self.query_tokens.unsqueeze(0).expand(b, -1, -1).to(embeds.dtype)], dim=1)
        attn = torch.cat([attention_mask, attention_mask.new_ones(b, q)], dim=1)
        ids = torch.cat([input_ids, torch.full((b, q), self.placeholder_id, dtype=input_ids.dtype, device=dev)], dim=1)
        mm = torch.cat([mm_token_type_ids, mm_token_type_ids.new_zeros(b, q)], dim=1)
        mean_half = self.layer_agg == "mean_last_half"
        out = self._run_lm(ids, embeds, attn, mm, image_grid_thw, True, hidden_states=mean_half)
        if mean_half:
            # Mean over the last half of the per-layer hidden states at the query positions: the final
            # layer is specialised for next-token prediction and drops visual detail.
            hs = out.hidden_states
            k = max(1, len(hs) // 2)
            return torch.stack([h[:, -q:, :] for h in hs[-k:]], dim=0).mean(dim=0)
        return out.last_hidden_state[:, -q:, :]

    def forward_recon(self, input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw,
                      target_pixel_values, **_):
        qh = self._recon_query_hidden(input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw)
        gen = self.decoder.diffusion_loss(
            self._cond(qh), target_pixel_values.to(self.dtype_),
            source_dropout=self.edit_source_dropout if self.training else 0.0)
        return {"loss": self.gen_loss_weight * gen, "diffusion_loss": gen.detach()}

    # -- stages 1-2: inline blocks ----------------------------------------------------------
    def _imagine_hidden(self, input_ids, attention_mask, mm_token_type_ids, pixel_values=None,
                        image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None):
        if self.layer_agg != "last":
            raise ValueError("the inline-block path reads the final layer only (layer_agg: last)")
        embeds, is_image = self._embed_with_vision(input_ids, pixel_values, image_grid_thw,
                                                   pixel_values_videos, video_grid_thw)
        qmask = input_ids == self.latent_query_id
        # The j-th slot of every block takes query_tokens[j] (blocks are exactly N slots long).
        q_index = (qmask.long().cumsum(dim=1) - 1).clamp(min=0) % self.num_queries
        embeds = torch.where(qmask.unsqueeze(-1), self.query_tokens[q_index].to(embeds.dtype), embeds)
        if self.imagine_embeds is not None:
            for i, tid in enumerate(self.imagine_ids):
                embeds = torch.where((input_ids == tid).unsqueeze(-1), self.imagine_embeds[i].to(embeds.dtype), embeds)
        grid = image_grid_thw if is_image else video_grid_thw
        hidden = self._run_lm(input_ids, embeds, attention_mask, mm_token_type_ids, grid, is_image).last_hidden_state
        return hidden, qmask & attention_mask.bool()

    def _block_diffusion_loss(self, q_hidden, targets, sources):
        """One decoder call over all blocks of the micro-batch. With no blocks an all-zero dummy block
        runs at zero weight, so every rank issues the same FSDP collectives."""
        if q_hidden.shape[0] == 0:
            q_hidden = q_hidden.new_zeros((1, self.num_queries, q_hidden.shape[-1]))
            targets = targets.new_zeros((1, *targets.shape[1:]))
            sources = targets if self.decoder.edit_cond else None
            weight = q_hidden.new_zeros((1,))
        else:
            if q_hidden.shape[0] != targets.shape[0]:
                raise ValueError(f"{q_hidden.shape[0]} query blocks != {targets.shape[0]} target images")
            weight = q_hidden.new_ones((q_hidden.shape[0],))
        src = sources.to(self.dtype_) if (self.decoder.edit_cond and sources is not None) else None
        return self.decoder.diffusion_loss(
            self._cond(q_hidden), targets.to(self.dtype_), sample_weight=weight, source=src,
            source_dropout=self.edit_source_dropout if self.training else 0.0)

    def _ce_chunk(self, sel, target):
        logits = self.backbone.get_output_embeddings()(sel).float()
        if self.imagine_lm_head is not None:
            # The marker columns come from the trainable imagine_lm_head, not the lm_head rows.
            extra = sel.float() @ self.imagine_lm_head.float().t()
            ids = torch.tensor(self.imagine_ids, device=logits.device)
            logits = logits.index_copy(1, ids, extra.to(logits.dtype))
        return F.cross_entropy(logits, target, reduction="sum")

    def _ce_loss(self, hidden, labels):
        """Next-token CE at supervised positions only, in checkpointed slices of _CE_CHUNK positions
        (sum inside, one mean at the end: identical to the unchunked loss)."""
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if not bool(mask.any()):
            return hidden.new_zeros(())
        sel = hidden[:, :-1, :][mask].to(self.dtype_)
        target = shift_labels[mask]
        n = sel.shape[0]
        if n <= _CE_CHUNK:
            return self._ce_chunk(sel, target) / n
        total = 0
        for i in range(0, n, _CE_CHUNK):
            s, t = sel[i:i + _CE_CHUNK], target[i:i + _CE_CHUNK]
            if torch.is_grad_enabled() and sel.requires_grad:
                total = total + torch.utils.checkpoint.checkpoint(self._ce_chunk, s, t, use_reentrant=False)
            else:
                total = total + self._ce_chunk(s, t)
        return total / n

    @torch.no_grad()
    def _fire_stats(self, hidden, labels):
        """At supervised marker positions: margin logit(OPEN) - logit(CLOSE) from imagine_lm_head,
        split by target class, plus how often its sign matches the target (the trigger accuracy)."""
        if self.imagine_lm_head is None:
            return {}
        shift = labels[:, 1:]
        is_open, is_close = shift == self.imagine_ids[0], shift == self.imagine_ids[1]
        mask = is_open | is_close
        if not bool(mask.any()):
            return {}
        head = self.imagine_lm_head.float()
        margin = hidden[:, :-1, :][mask].to(self.dtype_).float() @ (head[0] - head[1])
        want = is_open[mask]
        pos, neg = margin[want], margin[~want]
        return {"fire_pos_sum": pos.sum(), "fire_pos_cnt": margin.new_tensor(float(pos.numel())),
                "fire_neg_sum": neg.sum(), "fire_neg_cnt": margin.new_tensor(float(neg.numel())),
                "fire_correct": ((pos > 0).sum() + (neg <= 0).sum()).float()}

    def forward_imagine(self, input_ids, attention_mask, mm_token_type_ids, labels, target_pixel_values,
                        pixel_values=None, image_grid_thw=None, pixel_values_videos=None,
                        video_grid_thw=None, source_pixel_values=None, **_):
        hidden, qmask = self._imagine_hidden(input_ids, attention_mask, mm_token_type_ids, pixel_values,
                                             image_grid_thw, pixel_values_videos, video_grid_thw)
        q_hidden = hidden[qmask].view(-1, self.num_queries, hidden.shape[-1])
        gen = self._block_diffusion_loss(q_hidden, target_pixel_values, source_pixel_values)
        if self.ce_loss_weight == 0:
            # Diffusion-only: keep CE as an observable without building its (vocab-sized) graph.
            with torch.no_grad():
                ce = self._ce_loss(hidden, labels)
            total = self.gen_loss_weight * gen
        else:
            ce = self._ce_loss(hidden, labels)
            total = self.gen_loss_weight * gen + self.ce_loss_weight * ce
        return {"loss": total, "diffusion_loss": gen.detach(), "ce_loss": ce.detach(),
                **self._fire_stats(hidden, labels)}

    # -- dispatch ---------------------------------------------------------------------------
    def forward(self, mode: str = "train", **batch):
        # Everything (incl. viz sampling) goes through the root forward so FSDP gathers in lockstep.
        if mode == "viz":
            return self.generate(**batch)
        if "labels" in batch:
            return self.forward_imagine(**batch)
        return self.forward_recon(**batch)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, mm_token_type_ids, height, width, num_inference_steps=20,
                 pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None,
                 source_pixel_values=None, **_):
        """Render each row's first block (stages 1-2; a block-less row renders from zeros) or the
        appended queries (stage 0) -> pixels (B, 3, H, W) in [-1, 1]. No CFG."""
        if self.latent_query_id >= 0:
            hidden, qmask = self._imagine_hidden(input_ids, attention_mask, mm_token_type_ids, pixel_values,
                                                 image_grid_thw, pixel_values_videos, video_grid_thw)
            rows = []
            for b in range(input_ids.shape[0]):
                pos = qmask[b].nonzero(as_tuple=True)[0][: self.num_queries]
                qh = hidden[b, pos]
                rows.append(torch.cat([qh, qh.new_zeros(self.num_queries - qh.shape[0], qh.shape[-1])]))
            qh = torch.stack(rows)
        else:
            qh = self._recon_query_hidden(input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw)
        cond = self._cond(qh)
        src = source_pixel_values.to(self.dtype_) if source_pixel_values is not None else None
        return self.decoder.sample(cond, height, width, num_inference_steps, source=src)

    # -- checkpoint loading -----------------------------------------------------------------
    def load_checkpoint(self, ckpt_dir: str, backbone: bool = False, decoder: bool = True) -> None:
        """Load a checkpoint written by ``trainer.save_checkpoint`` (or a stage-0 dir with the SANA
        components at its root): SANA transformer (when ``decoder`` and present), ``latent_query.pt``,
        and with ``backbone`` the HF safetensors at the root. Call before the FSDP wrap."""
        root = diffusers_root(ckpt_dir)
        tf_dir = os.path.join(root, "transformer")
        if decoder and os.path.isdir(tf_dir):
            missing, _ = self.decoder.load_transformer(tf_dir, self.dtype_)
            if "patch_embed.proj.weight" in missing:
                raise RuntimeError(f"SANA load from {tf_dir} did not match patch_embed")
        state = torch.load(os.path.join(ckpt_dir, "latent_query.pt"), map_location="cpu")
        with torch.no_grad():
            self.query_tokens.copy_(state["query_tokens"].to(self.query_tokens.dtype))
            self.connector.load_state_dict(
                {k[len("connector."):]: v for k, v in state.items() if k.startswith("connector.")})
            for name in ("imagine_embeds", "imagine_lm_head"):
                p = getattr(self, name)
                if p is not None and name in state:
                    p.copy_(state[name].to(p.dtype))
        if backbone:
            import glob

            from safetensors.torch import load_file

            sd = {}
            for shard in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
                sd.update(load_file(shard))
            if sd:
                _, unexpected = self.backbone.load_state_dict(sd, strict=False)
                if len(sd) == len(unexpected):
                    raise RuntimeError(f"backbone load from {ckpt_dir} matched no tensors")
        logger.info("loaded %s (backbone=%s)", ckpt_dir, backbone)

    def latent_query_state(self, full_sd: dict) -> dict:
        """The ``latent_query.pt`` payload out of a full (consolidated) state dict."""
        keys = ("query_tokens", "imagine_embeds", "imagine_lm_head")
        return {k: v for k, v in full_sd.items() if k in keys or k.startswith("connector.")}
