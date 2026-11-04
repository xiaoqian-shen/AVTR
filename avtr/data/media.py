"""Prepare image and video inputs with Qwen3.5 pixel budgets.

The processor uses do_resize=False to preserve the resizing applied by qwen-vl-utils.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

# Qwen3.5 merges every TEMPORAL_PATCH sampled frames into one temporal position.
TEMPORAL_PATCH = 2


def _qwen_vl_utils():
    # qwen-vl-utils reads its reader choice at import; torchvision's reader segfaults in forked workers.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    from qwen_vl_utils import process_vision_info

    logging.getLogger("qwen_vl_utils.vision_process").setLevel(logging.WARNING)
    logging.getLogger("libav").setLevel(logging.CRITICAL)
    return process_vision_info


def grid_images(images: list, max_pixels: int | None, min_pixels: int | None) -> list:
    content = []
    for im in images:
        elem = {"type": "image", "image": im}
        if max_pixels is not None:
            elem["max_pixels"] = max_pixels
        if min_pixels is not None:
            elem["min_pixels"] = min_pixels
        content.append(elem)
    out, _ = _qwen_vl_utils()([{"role": "user", "content": content}], image_patch_size=16)
    return list(out)


def load_video(path: str, fps: float, max_frames: int, max_pixels: int, min_pixels: int):
    """-> (frames (T, C, H, W), metadata, video_kwargs, per-temporal-position timestamps in seconds)."""
    content = {"type": "video", "video": path, "fps": fps, "max_frames": max_frames,
               "max_pixels": max_pixels, "min_pixels": min_pixels}
    _, videos, video_kwargs = _qwen_vl_utils()(
        [{"role": "user", "content": [content]}], return_video_kwargs=True, return_video_metadata=True,
        image_patch_size=16)
    frames, meta = videos[0]
    get = (lambda k: meta[k]) if isinstance(meta, dict) else (lambda k: getattr(meta, k))
    idx, vfps = list(get("frames_indices")), float(get("fps"))
    if len(idx) % TEMPORAL_PATCH:
        idx += [idx[-1]] * (TEMPORAL_PATCH - len(idx) % TEMPORAL_PATCH)
    secs = [i / vfps for i in idx]
    stamps = [(secs[i] + secs[i + TEMPORAL_PATCH - 1]) / 2 for i in range(0, len(secs), TEMPORAL_PATCH)]
    return frames, meta, video_kwargs, stamps


def to_target(frame: torch.Tensor, resolution: int) -> torch.Tensor:
    """A frame (C, H, W; uint8-valued or [0, 1]) -> (3, R, R) float in [-1, 1]."""
    x = frame.float()
    x = x / 127.5 - 1.0 if x.max() > 1.5 else x * 2.0 - 1.0
    return F.interpolate(x.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False).squeeze(0)


def pil_target(image, resolution: int) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).copy()
    return to_target(torch.from_numpy(arr).permute(2, 0, 1), resolution)


def mm_token_types(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """M-RoPE token types rebuilt from the image / video pad ids, so each image is its own run
    even when the processor would merge adjacent images into one."""
    mm = torch.zeros_like(input_ids)
    mm[input_ids == tokenizer.convert_tokens_to_ids("<|image_pad|>")] = 1
    mm[input_ids == tokenizer.convert_tokens_to_ids("<|video_pad|>")] = 2
    return mm.long()
