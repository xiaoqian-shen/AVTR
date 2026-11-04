"""Stream image-caption reconstruction samples from WebDataset shards.

Shards are partitioned across ranks and workers; captions and reconstruction prompts
provide the conditioning text.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset, get_worker_info

from .. import fileio
from . import media
from .imagine import rank_world

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_CAPTION_EXTS = (".txt", ".caption", ".cap")
_JSON_CAPTION_KEYS = ("caption", "txt", "text", "short_caption", "title", "alt_text")
# Some packers stamp this on un-captioned images; treat it as no caption.
_EMPTY_CAPTIONS = frozenset({"no_caption"})
_FALLBACK_PROMPT = "Reconstruct this image."


def _sample(members: dict[str, bytes]) -> tuple[bytes, str] | None:
    img = next((members[e] for e in _IMAGE_EXTS if e in members), None)
    if img is None:
        return None
    caption = next((members[e].decode("utf-8", "replace").strip() for e in _CAPTION_EXTS if e in members), "")
    if not caption and ".json" in members:
        try:
            obj = json.loads(members[".json"].decode("utf-8", "replace"))
            caption = next((obj[k] for k in _JSON_CAPTION_KEYS if isinstance(obj.get(k), str) and obj[k].strip()), "")
        except Exception:  # noqa: BLE001
            caption = ""
    if caption.strip().lower() in _EMPTY_CAPTIONS:
        caption = ""
    return img, caption


def _iter_tar(data: bytes):
    """Yield (image_bytes, caption) per WebDataset sample (members share the key up to the first dot)."""
    key, members = None, {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            k = m.name.rsplit("/", 1)[-1].partition(".")[0]
            if key is not None and k != key:
                s = _sample(members)
                if s is not None:
                    yield s
                members = {}
            key = k
            fh = tar.extractfile(m)
            if fh is not None:
                members[os.path.splitext(m.name)[1].lower()] = fh.read()
    s = _sample(members)
    if s is not None:
        yield s


class ReconStream(IterableDataset):
    def __init__(self, cfg: dict, processor, resolution: int, seed: int) -> None:
        super().__init__()
        self.processor = processor
        self.res = resolution
        self.seed = seed
        self.shuffle_buffer = cfg.get("shuffle_buffer", 1000)
        self.prompt_prob = cfg.get("recon_prompt_prob", 0.0)
        self.prompt = cfg.get("recon_prompt", "Please reconstruct this image.")
        self.max_caption_tokens = cfg.get("max_caption_tokens", 0)
        self.max_pixels, self.min_pixels = cfg.get("image_max_pixels"), cfg.get("image_min_pixels")
        self.shards = []
        for root in cfg["roots"]:
            names = sorted(n for n in fileio.listdir(root) if n.endswith(".tar"))
            if not names:
                raise FileNotFoundError(f"no *.tar shards under {root}")
            self.shards += [f"{root.rstrip('/')}/{n}" for n in names]
        logger.info("recon: %d tar shards across %d roots", len(self.shards), len(cfg["roots"]))

    def _rows(self, epoch: int, gid: int, total: int):
        shards = list(self.shards)
        random.Random(self.seed + epoch).shuffle(shards)
        for shard in shards[gid::total]:
            try:
                with fileio.open_file(shard, "rb") as fh:
                    yield from _iter_tar(fh.read())
            except Exception as e:  # noqa: BLE001 -- skip a corrupt shard
                logger.warning("skipping shard %s: %s", shard, e)

    def _shuffled(self, epoch: int, gid: int, total: int):
        rows = self._rows(epoch, gid, total)
        if self.shuffle_buffer <= 1:
            yield from rows
            return
        rng = random.Random(self.seed + epoch + 1)
        buf = []
        for row in rows:
            buf.append(row)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf

    def __iter__(self):
        rank, world = rank_world()
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        gid, total = rank * nw + wid, world * nw
        # Every (rank, worker) needs a shard of its own: one without any would never yield, and its
        # rank would block on that worker's batch forever.
        if total > len(self.shards):
            raise RuntimeError(f"data under-sharded: {len(self.shards)} shards for {world} ranks x {nw} "
                               f"workers; need >= world_size * dataloader_num_workers = {total} shards")
        epoch, alive = 0, False
        while True:
            aug = random.Random(self.seed + epoch * 100003 + wid + 1)
            for img_bytes, caption in self._shuffled(epoch, gid, total):
                text = (self.prompt if self.prompt_prob > 0 and aug.random() < self.prompt_prob
                        else (caption or _FALLBACK_PROMPT))
                try:
                    example = self._example(img_bytes, text)
                except Exception as e:  # noqa: BLE001 -- skip an undecodable sample
                    logger.warning("skipping sample: %s", e)
                    continue
                alive = True
                yield example
            epoch += 1
            # A worker whose shards never yield a sample (unreadable, or no image members) would
            # otherwise spin here silently; a few bad shards only cost a pass.
            if not alive and epoch >= 3:
                raise RuntimeError(f"recon worker {gid}/{total}: {epoch} passes over its shards produced no sample")

    def _example(self, img_bytes: bytes, text: str) -> dict:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(io.BytesIO(img_bytes))
        tok = self.processor.tokenizer
        if self.max_caption_tokens > 0:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > self.max_caption_tokens:
                text = tok.decode(ids[: self.max_caption_tokens], skip_special_tokens=True)
        grid = media.grid_images([image.convert("RGB")], self.max_pixels, self.min_pixels)
        content = [{"type": "image"} for _ in grid] + [{"type": "text", "text": text}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[prompt], images=grid, return_tensors="pt", do_resize=False)
        mm = inputs.get("mm_token_type_ids")
        target = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "mm_token_type_ids": (mm if mm is not None else torch.zeros_like(inputs["input_ids"]))[0],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "target_pixel_values": media.to_target(target, self.res),
        }


class ReconCollator:
    """Left-pads the prompts so the appended queries sit right after each sample's content."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        L = max(b["input_ids"].shape[0] for b in batch)

        def pad(key, value):
            return torch.stack([F.pad(b[key], (L - b[key].shape[0], 0), value=value) for b in batch])

        return {"input_ids": pad("input_ids", self.pad), "attention_mask": pad("attention_mask", 0),
                "mm_token_type_ids": pad("mm_token_type_ids", 0),
                "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
                "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
                "target_pixel_values": torch.stack([b["target_pixel_values"] for b in batch])}
