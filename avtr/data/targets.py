"""Select visual clue targets from image regions, temporal spans, and sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re

from PIL import Image, ImageDraw

from .. import fileio

logger = logging.getLogger(__name__)

# Sources whose video rows may emit a block from a temporal span (any other source is a trigger
# negative unless an image rule below gives it a target).
TG_SOURCES = {"grounding_cot", "activitynet", "g_videoqa", "open_o3", "videochat3_lv"}
# Always target-less, bypassing every sidecar gate.
NEG_SOURCES = {"joyai_neg"}
# Bbox rows take a region crop target when no region sidecar is configured.
BOX_SOURCES = {"visualcot"}

MEDIA_TOKENS = ("<video>", "<image>", "<|video_pad|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


def strip_media_tokens(text: str) -> str:
    for tok in MEDIA_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


# ---------------------------------------------------------------------------------------------
# Temporal spans
# ---------------------------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_T_RE = re.compile(r"<t>\s*(\d+(?:\.\d+)?)\s*</t>")
_BOXED_SPAN_RE = re.compile(r"\\boxed\{\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\}")


def parse_span(answer: str | None) -> tuple[float, float] | None:
    """``From <t>a</t>s to <t>b</t>s`` (optionally inside <answer>) -> (a, b) from the first and last
    <t> tags; else a ``\\boxed{[a, b]}`` span; else None."""
    text = (answer or "").strip()
    m = _ANSWER_RE.search(text)
    if m is not None:
        text = m.group(1).strip()
    tags = _T_RE.findall(text)
    if len(tags) >= 2:
        a, b = float(tags[0]), float(tags[-1])
        return (a, b) if a <= b else (b, a)
    m = _BOXED_SPAN_RE.search(answer or "")
    if m is not None:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if hi > lo else None
    return None


def frame_in_span(frames, timestamps: list[float], start: float, end: float, rng: random.Random):
    """A random sampled frame whose merged timestamp lies in [start, end] (nearest to the midpoint if
    none does). Temporal position k maps back to the first frame of its merged group."""
    if not timestamps or int(frames.shape[0]) == 0:
        return None
    cands = [k for k, ts in enumerate(timestamps) if start <= ts <= end]
    if not cands:
        mid = 0.5 * (start + end)
        cands = [min(range(len(timestamps)), key=lambda k: abs(timestamps[k] - mid))]
    k = rng.choice(cands)
    return frames[min(k * 2, int(frames.shape[0]) - 1)]


def frame_in_spans(frames, timestamps, spans, rng: random.Random):
    """A random span first, then a random frame inside it (every cited moment equally likely)."""
    valid = [(float(p[0]), float(p[1])) for p in (spans or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not valid:
        return None
    start, end = rng.choice(valid)
    return frame_in_span(frames, timestamps, start, end, rng)


def frame_at_second(video_path: str, frames, timestamps: list[float], sec: float):
    """The exact frame played at ``sec``; the nearest sampled frame if exact decoding fails."""
    try:
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(video_path).get_frame_played_at(float(sec)).data
    except Exception:  # noqa: BLE001
        if not timestamps or int(frames.shape[0]) == 0:
            return None
        k = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - sec))
        return frames[min(k * 2, int(frames.shape[0]) - 1)]


# "From <t>a</t>s to <t>b</t>s" and "\boxed{[a, b]}" -> "[a, b]", <answer> wrapper dropped: grounding
# targets / instructions train in bare brackets (the paired <t> tags degrade right after a block).
_TG_BRACKET_RE = re.compile(r"From <t>([^<]+)</t>s to <t>([^<]+)</t>s")
_TG_BOXED_RE = re.compile(r"\\boxed\{\s*(\[[^\]]*\])\s*\}")


def to_bracket_tg(text: str | None) -> str | None:
    if not text:
        return text
    text = _TG_BOXED_RE.sub(r"\1", _TG_BRACKET_RE.sub(r"[\1, \2]", text))
    return text.replace("<answer>", "").replace("</answer>", "")


# ---------------------------------------------------------------------------------------------
# Image targets
# ---------------------------------------------------------------------------------------------
# JoyAI overlay palette ("green" is pure lime). A washed entry names the answer's overlay colour, so
# the target box is drawn in that colour; unknown / absent -> red.
_PALETTE = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "purple": (128, 0, 128),
            "orange": (255, 165, 0), "yellow": (255, 255, 0), "pink": (255, 192, 203), "brown": (165, 42, 42)}


def box_color(name: str | None) -> tuple[int, int, int]:
    return _PALETTE.get((name or "").strip().lower(), (255, 0, 0))


def _merge_boxes(boxes: list[list[float]]) -> list[list[float]]:
    """Nested -> keep the smaller; partial overlap -> union; disjoint -> keep both (to a fixed point)."""
    def contains(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]

    def intersects(a, b):
        return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])

    out = [list(map(float, b)) for b in boxes]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if contains(a, b):
                    new = b
                elif contains(b, a):
                    new = a
                elif intersects(a, b):
                    new = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                else:
                    continue
                out[i] = new
                out.pop(j)
                changed = True
                break
            if changed:
                break
    return out


def boxed_view(img, boxes_px, resolution: int, thickness: int, color, min_box_frac: float = 0.0):
    """The whole image square-resized to ``resolution`` with thick merged box(es) drawn on it. Each box
    first grows about its centre to >= ``min_box_frac`` of the image side (JoyAI point boxes)."""
    img = img.convert("RGB")
    w, h = img.size
    clip = []
    for b in boxes_px:
        x1, x2 = sorted((max(0.0, min(b[0], w)), max(0.0, min(b[2], w))))
        y1, y2 = sorted((max(0.0, min(b[1], h)), max(0.0, min(b[3], h))))
        if min_box_frac > 0.0:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            hw, hh = max((x2 - x1) / 2.0, min_box_frac * w / 2.0), max((y2 - y1) / 2.0, min_box_frac * h / 2.0)
            x1, x2 = max(0.0, cx - hw), min(float(w), cx + hw)
            y1, y2 = max(0.0, cy - hh), min(float(h), cy + hh)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            clip.append([x1, y1, x2, y2])
    region = img.resize((resolution, resolution), Image.BILINEAR)
    sx, sy = resolution / w, resolution / h
    draw = ImageDraw.Draw(region)
    for b in _merge_boxes(clip) if clip else []:
        draw.rectangle([round(v) for v in (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)], outline=color, width=thickness)
    return region


def square_region_crop(img, box_px, resolution: int, margin: float = 0.15, min_side: int = 64):
    """Square crop around the region at native resolution: side = longer edge grown by ``margin`` each
    way, floored at ``min_side``, clamped to the image; only downscaled (to ``resolution``), never
    upscaled. No box drawn."""
    img = img.convert("RGB")
    w, h = img.size
    x1, x2 = sorted((max(0.0, min(box_px[0], w)), max(0.0, min(box_px[2], w))))
    y1, y2 = sorted((max(0.0, min(box_px[1], h)), max(0.0, min(box_px[3], h))))
    side = max(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin), float(min_side))
    iside = max(1, min(int(round(side)), min(w, h)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = int(round(min(max(cx - iside / 2.0, 0.0), w - iside)))
    y0 = int(round(min(max(cy - iside / 2.0, 0.0), h - iside)))
    region = img.crop((x0, y0, x0 + iside, y0 + iside))
    if iside > resolution:
        return region.resize((resolution, resolution), Image.BILINEAR)
    return region.copy()  # crop() is lazy; own the buffer before it crosses the worker pickle


# ---------------------------------------------------------------------------------------------
# JoyAI gates
# ---------------------------------------------------------------------------------------------
# Box side floor (fraction of the image) for JoyAI's point boxes.
JOYAI_BOX_MIN_FRAC = 0.2

# BLINK-style multi-view "did the camera move left or right" questions, matched on the OPTIONS (A and
# B both left/right, no C) so pan/tilt/roll TYPE questions stay out. This family bypasses the
# base-right drop and the negative subsample.
JOYAI_MULTIVIEW_RE = re.compile(
    r"(?s)\A(?=.*camera)(?=.*(rotat|roll|orbit|mov|translat))"
    r"(?=.*\bA[.)]\s*(rolling to the )?(left|right)\b)"
    r"(?=.*\bB[.)]\s*(rolling to the )?(left|right)\b)"
    r"(?!.*\bC[.)])",
    re.I,
)


def keep_hashed(key: str | None, prob: float) -> bool:
    """Deterministic keep with probability ``prob``, keyed by record id (stable across epochs)."""
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    digest = hashlib.blake2b((key or "").encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) % 10000) < int(prob * 10000)


def joyai_target_spec(entry: dict | None, num_images: int):
    """Sidecar entry ``{"box": [[x0,y0,x1,y1], ...], "img_index": i}`` -> (i, boxes) or None."""
    if not entry or not entry.get("box"):
        return None
    idx = int(entry.get("img_index", 1))
    if idx < 0 or idx >= num_images:
        return None
    return idx, [[float(v) for v in b] for b in entry["box"]]


# ---------------------------------------------------------------------------------------------
# Target quality + sidecars
# ---------------------------------------------------------------------------------------------
def _target_quality(target) -> tuple[float, float]:
    """(std in grey levels, mean |gradient| / std). The ratio is scale-free, so blur separates
    from legitimately dark scenes. Read off a small strided grey copy."""
    import numpy as np

    if hasattr(target, "convert"):
        arr = np.asarray(target.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(getattr(target, "cpu", lambda: target)(), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=0 if arr.shape[0] <= 4 else 2)
        if arr.max() <= 1.5:
            arr = (arr + 1.0) * 127.5 if arr.min() < -0.01 else arr * 255.0
    step = max(1, min(arr.shape[:2]) // 64)
    arr = arr[::step, ::step]
    if arr.size < 16:
        return 0.0, 0.0
    grad = 0.5 * (float(np.abs(np.diff(arr, axis=0)).mean()) + float(np.abs(np.diff(arr, axis=1)).mean()))
    std = float(arr.std())
    return std, (grad / std if std > 1e-6 else 0.0)


def is_degenerate(target, min_std: float, min_sharp: float) -> bool:
    """Too flat (black title card) or too soft (blank slide, motion blur) to train on; off when neither
    threshold is positive."""
    if (min_std <= 0.0 and min_sharp <= 0.0) or target is None:
        return False
    try:
        std, sharp = _target_quality(target)
    except Exception:  # noqa: BLE001
        return False
    return (min_std > 0.0 and std < min_std) or (min_sharp > 0.0 and sharp < min_sharp)


def read_sidecar(path: str | None) -> dict:
    """A JSON sidecar (local or fsspec URL); ``{}`` when unset. A configured unreadable sidecar raises."""
    if not path:
        return {}
    with fileio.open_file(path, "rb") as fh:
        loaded = json.loads(fh.read())
    logger.info("loaded sidecar %s (%d entries)", path, len(loaded))
    return loaded
