"""Worker setup for precision, media decoding, and Hugging Face checkpoint export."""

from __future__ import annotations

import torch
from PIL import Image, ImageFile
from qwen_vl_utils import vision_process
from transformers.models.qwen3_5 import modeling_qwen3_5

from .data import VIDEO


def keep_fp32_buffers(cls: type[torch.nn.Module]) -> None:
    """Make ``module.to(dtype)`` / ``.half()`` / ``.bfloat16()`` keep the fp32 buffers of ``cls``
    modules in fp32; device moves still apply."""
    cast = cls._apply

    def _apply(self, fn, recurse=True):
        fp32 = {k: b for k, b in self._buffers.items() if b is not None and b.dtype == torch.float32}
        out = cast(self, fn, recurse)
        for k, b in fp32.items():
            if self._buffers[k].dtype != torch.float32:
                self._buffers[k] = b.to(self._buffers[k].device)
        return out

    cls._apply = _apply


for _rotary in (modeling_qwen3_5.Qwen3_5TextRotaryEmbedding, modeling_qwen3_5.Qwen3_5VisionRotaryEmbedding):
    keep_fp32_buffers(_rotary)

# A frame of a short clip may take up to the whole total_pixels; a token covers one merged block of
# Qwen3.5's vision patches (``32**2`` pixels).
vision_process.VIDEO_MAX_TOKEN_NUM = max(vision_process.VIDEO_MAX_TOKEN_NUM, VIDEO["total_pixels"] // 32**2)

MAX_DECODE_SIDE = 4096
ImageFile.LOAD_TRUNCATED_IMAGES = True


def to_rgb(image: Image.Image) -> Image.Image:
    """The reference run's decode of an opened image (qwen_vl_utils' ``to_rgb`` hook)."""
    if max(image.size) > MAX_DECODE_SIDE:
        image.draft("RGB", (MAX_DECODE_SIDE, MAX_DECODE_SIDE))
    image = image.convert("RGB")
    if max(image.size) > MAX_DECODE_SIDE:
        image.thumbnail((MAX_DECODE_SIDE, MAX_DECODE_SIDE))
    return image


vision_process.to_rgb = to_rgb

_save_pretrained = modeling_qwen3_5.Qwen3_5ForConditionalGeneration.save_pretrained


def save_pretrained(self, *args, save_original_format: bool = False, **kwargs):
    """``save_pretrained`` under the model's own parameter names unless ``save_original_format=True``."""
    return _save_pretrained(self, *args, save_original_format=save_original_format, **kwargs)


modeling_qwen3_5.Qwen3_5ForConditionalGeneration.save_pretrained = save_pretrained
