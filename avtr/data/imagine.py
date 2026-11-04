"""Weighted shard streams and collation for visual reflection and SFT.

Positive rows contain a latent-query block and a diffusion target. Negative rows begin
with the closing imagination marker and supervise withholding visual imagination.
"""

from __future__ import annotations

import io
import logging
import os
import random
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from .. import fileio
from ..model import IMAGINE_CLOSE, IMAGINE_DIRECTIVE, IMAGINE_OPEN, LATENT_QUERY_TOKEN
from . import media, records
from . import targets as T

logger = logging.getLogger(__name__)

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
# Qwen3.5 renders a non-reasoning assistant turn with this closed think scaffold.
_CLOSED_THINK = "<think>\n\n</think>\n\n"
_THINK_OPENER = "<think>\n"


def rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def discover_sources(source_cfgs: list[dict]) -> list[dict]:
    """``[{name, adapter, dir, subdirs?, weight}]`` -> the same with ``shards`` listed. Order matters:
    each source's shard RNG is seeded by its position."""
    out = []
    for s in source_cfgs:
        if not s.get("dir"):
            continue
        ext = ".tar" if s["adapter"] == "spatialedit" else ".parquet"
        shards = []
        for sub in s.get("subdirs") or [""]:
            d = f"{s['dir'].rstrip('/')}/{sub}" if sub else s["dir"].rstrip("/")
            shards += sorted(f"{d}/{n}" for n in fileio.listdir(d) if n.endswith(ext))
        if not shards:
            raise ValueError(f"source {s['name']!r}: no *{ext} shards under {s['dir']}")
        if s["adapter"] not in records.ADAPTERS:
            raise ValueError(f"source {s['name']!r}: unknown adapter {s['adapter']!r}")
        out.append({**s, "shards": shards, "weight": float(s.get("weight", 1.0))})
        logger.info("source %s: %d shards, weight %s", s["name"], len(shards), s.get("weight", 1.0))
    if not out:
        raise ValueError("no data sources configured")
    return out


def _assistant_marker_index(opens: torch.Tensor, closes: torch.Tensor, n: int) -> int | None:
    """The marker carrying the assistant's open/close decision. The directive spells both markers in
    the user turn, so take the filled block's open (an open with exactly n latent slots before the last close), else
    the last close (a bare decline)."""
    if closes.numel() == 0:
        return None
    last_close = int(closes.max())
    before = opens[opens < last_close]
    if before.numel() > 0 and last_close - int(before.max()) == n + 1:
        return int(before.max())
    return last_close


class ImagineStream(IterableDataset):
    def __init__(self, cfg: dict, processor, num_queries: int, markers: bool, resolution: int, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.sources = discover_sources(cfg["sources"])
        self.processor = processor
        self.tok = processor.tokenizer
        self.n = num_queries
        self.markers = markers
        self.res = resolution
        self.seed = seed
        self.latent_id = int(self.tok.convert_tokens_to_ids(LATENT_QUERY_TOKEN))
        self.open_id = int(self.tok.convert_tokens_to_ids(IMAGINE_OPEN)) if markers else None
        self.close_id = int(self.tok.convert_tokens_to_ids(IMAGINE_CLOSE)) if markers else None
        self._rng: random.Random | None = None
        self._clue: dict = {}
        self._joyai: dict = {}
        self._region: dict = {}

    # -- streaming ---------------------------------------------------------------------------
    def _read_shard(self, shard: str, adapter: str) -> list[dict]:
        try:
            with fileio.open_file(shard, "rb") as fh:
                data = fh.read()
            if adapter == "spatialedit":
                return records.read_spatialedit_tar(data)
            import pyarrow.parquet as pq

            return pq.read_table(io.BytesIO(data)).to_pylist()
        except Exception as exc:  # noqa: BLE001 -- a bad shard must not kill the worker
            logger.warning("skipping unreadable shard %s: %s", shard, exc)
            return []

    def _iter_source(self, src: dict, part: int, n_parts: int, base_seed: int):
        epoch = empty_passes = 0
        while True:
            rng = random.Random(base_seed + epoch)
            shards = list(src["shards"])
            rng.shuffle(shards)
            mine = shards[part::n_parts] or [shards[part % len(shards)]]
            produced = False
            for shard in mine:
                rows = self._read_shard(shard, src["adapter"])
                rng.shuffle(rows)
                for row in rows:
                    for record in records.ADAPTERS[src["adapter"]](row):
                        produced = True
                        yield record
            epoch += 1
            empty_passes = 0 if produced else empty_passes + 1
            if empty_passes >= 3:  # every shard unreadable or every row dropped: fail, never spin
                raise RuntimeError(
                    f"source {src['name']!r}: {empty_passes} passes over shards {mine} produced no record")

    def __iter__(self):
        # Read here, outside the per-sample try/except: a configured but unreadable sidecar must fail.
        self._clue = T.read_sidecar(self.cfg.get("clue_sidecar"))
        self._joyai = T.read_sidecar(self.cfg.get("joyai_train_sidecar"))
        self._region = T.read_sidecar(self.cfg.get("region_sidecar"))
        rank, world = rank_world()
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        part, n_parts = rank * nw + wid, world * nw
        self._rng = random.Random(self.seed * 2654435761 + rank * 10007 + wid)
        gens = [self._iter_source(s, part, n_parts, self.seed + 1000 * i) for i, s in enumerate(self.sources)]
        weights = [s["weight"] for s in self.sources]
        pick = random.Random(self.seed * 7919 + part)
        misses = 0
        while True:
            i = pick.choices(range(len(self.sources)), weights=weights, k=1)[0]
            record = next(gens[i])
            record["_source"] = self.sources[i]["name"]
            sample = self._process(record)
            if sample is not None:
                misses = 0
                yield sample
            else:
                misses += 1
                if misses >= 50000:  # e.g. stage 1 with no row that has a target: fail, never spin
                    raise RuntimeError(f"no usable sample in the last {misses} records; check the sidecars and sources")

    # -- record -> sample --------------------------------------------------------------------
    def _process(self, record: dict) -> dict | None:
        convs = record.get("conversations") or []
        question = T.strip_media_tokens(next((str(t.get("value", "")) for t in convs if t.get("from") == "human"), ""))
        answer = str(next((t for t in convs if t.get("from") == "gpt"), {}).get("value", ""))
        if not answer:
            return None
        source = record.get("_source", "")
        system = record.get("system")
        tg = source in T.TG_SOURCES
        if tg:
            system = T.to_bracket_tg(system)
        content = (T.to_bracket_tg(answer) if tg else answer).lstrip("\n")
        video = record.pop("_video_bytes", None)
        if video is not None:
            return self._video_sample(question, answer, content, video, system, tg, record.get("spans"))
        images = record.get("image")
        if (source in T.NEG_SOURCES and self.cfg.get("joyai_neg_min_images", 1) > 1
                and len(images or []) < self.cfg["joyai_neg_min_images"]):
            return None
        if not images:
            return None
        return self._image_sample(images, question, content, record, system, tg, source)

    def _video_sample(self, question, answer, content, video_bytes, system, tg, spans):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as fh:
                fh.write(video_bytes)
            c = self.cfg
            frames, meta, vkw, stamps = media.load_video(tmp, c["video_fps"], c["video_max_frames"],
                                                         c["video_max_pixels"], c["video_min_pixels"])
            target = self._video_target(tmp, frames, stamps, question, answer, video_bytes, spans)
            has_block = tg and target is not None
            if c.get("target_only") and not has_block:
                return None
            if has_block and T.is_degenerate(target, c.get("target_min_std", 0), c.get("target_min_sharp", 0)):
                return None
            user = {"role": "user",
                    "content": [{"type": "video"}, {"type": "text", "text": IMAGINE_DIRECTIVE + question}]}
            kw = {"videos": [frames], "video_metadata": [meta], "return_tensors": "pt", "do_resize": False, **vkw}
            tgt = media.to_target(target, self.res) if has_block else None
            return self._build(user, system, content, kw, tgt, None)
        except Exception as exc:  # noqa: BLE001 -- one bad video must not kill the worker
            logger.warning("skipping bad video sample: %s", exc)
            return None
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.remove(tmp)

    def _video_target(self, path, frames, stamps, question, answer, video_bytes, spans):
        """A washed clue frame (exact ``clue_sec``) upgrades the default; otherwise a random frame inside
        the record's pre-parsed spans or its answer's temporal span; no span -> no target."""
        entry = self._clue.get(records.clue_key(video_bytes, question))
        if entry is not None and float(entry.get("gen_mask", 0.0)) > 0.0 and entry.get("clue_sec") is not None:
            return T.frame_at_second(path, frames, stamps, float(entry["clue_sec"]))
        if spans:
            return T.frame_in_spans(frames, stamps, spans, self._rng)
        span = T.parse_span(answer)
        return T.frame_in_span(frames, stamps, span[0], span[1], self._rng) if span else None

    def _image_spec(self, images, record, tg, source, force_keep):
        """-> (source_image, target_image, has_block), or None to drop the row."""
        c = self.cfg
        edit = record.get("_edit_target_bytes")
        if edit is not None:
            # SpatialEdit: no edit source, so the whole edited scene has to come through the queries.
            try:
                return None, records.decode_pil(edit), True
            except Exception:  # noqa: BLE001
                return None
        if source in T.NEG_SOURCES:
            return None, None, False
        bbox = record.get("bbox")
        if bbox and self._region:
            entry = self._region.get(record.get("_region_key")) if record.get("_region_key") else None
            if entry and float(entry.get("gen_mask", 0.0)) > 0.0 and entry.get("box_px"):
                box = tuple(float(v) for v in entry["box_px"])
                return images[0], T.square_region_crop(images[0], box, self.res, margin=0.15), True
            return None  # unwashed / unhelpful region
        if bbox and (tg or source in T.BOX_SOURCES):
            b = [float(v) for v in bbox[0]][:4]
            return None, T.square_region_crop(images[0], b, self.res, margin=0.15, min_side=self.res), True
        if c.get("joyai_train_sidecar") or tg:
            return self._joyai_spec(record.get("id"), images, force_keep)
        return None, None, False

    def _joyai_spec(self, record_id, images, force_keep):
        c = self.cfg
        entry = self._joyai.get(record_id) if record_id else None
        if entry is None:  # base model was right
            if force_keep and T.keep_hashed(record_id, c.get("joyai_mv_base_right_keep_prob", 1.0)):
                return None, None, False
            return None
        if float(entry.get("gen_mask", 0.0)) > 0.0:
            spec = T.joyai_target_spec(entry, len(images))
            if spec is not None:
                idx, boxes = spec
                # Colour-ref boxes are already traced object boxes; only point boxes are grown.
                frac = 0.0 if entry.get("family") == "color-ref" else T.JOYAI_BOX_MIN_FRAC
                tgt = T.boxed_view(images[idx], boxes, self.res, c.get("joyai_answer_box_thickness", 3),
                                   T.box_color(entry.get("color")), min_box_frac=frac)
                return images[idx], tgt, True
        if not force_keep and not T.keep_hashed(record_id, c.get("joyai_neg_keep_prob", 1.0)):
            return None
        return None, None, False

    def _image_sample(self, images, question, content, record, system, tg, source):
        c = self.cfg
        try:
            spec = self._image_spec(images, record, tg, source, bool(T.JOYAI_MULTIVIEW_RE.search(question or "")))
            if spec is None:
                return None
            src_img, tgt_img, has_block = spec
            if c.get("target_only") and not has_block:
                return None
            if has_block and T.is_degenerate(tgt_img, c.get("target_min_std", 0), c.get("target_min_sharp", 0)):
                return None
            # On multi-image rows the boxed view IS one of the inputs; feeding it as the edit source
            # would hand the decoder the answer to "which view?".
            if src_img is not None and len(images) != 1:
                src_img = None
            grid = media.grid_images(images, c["image_max_pixels"], c["image_min_pixels"])
            user = {"role": "user", "content": [{"type": "image"} for _ in grid]
                    + [{"type": "text", "text": IMAGINE_DIRECTIVE + question}]}
            kw = {"images": grid, "return_tensors": "pt", "do_resize": False}
            tgt = media.pil_target(tgt_img, self.res) if has_block and tgt_img is not None else None
            src = media.pil_target(src_img, self.res) if tgt is not None and src_img is not None else None
            return self._build(user, system, content, kw, tgt, src)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping bad image sample: %s", exc)
            return None

    def _build(self, user_turn, system, content, proc_kwargs, target, source):
        """Tokenize prompt and full turn, supervise the assistant text after the prompt (latent slots
        masked) plus the single decision marker, and attach the per-block target."""
        has_block = target is not None
        if has_block:
            block = (IMAGINE_OPEN + LATENT_QUERY_TOKEN * self.n + IMAGINE_CLOSE + "\n") if self.markers \
                else LATENT_QUERY_TOKEN * self.n + "\n"
        else:
            block = (IMAGINE_CLOSE + "\n") if self.markers else ""
        turns = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + [user_turn]
        header = self.processor.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        header += f"{_IM_START}assistant\n" + block
        if self.cfg.get("think_opener"):
            # Prompt ends at "<think>\n" (the official template's generation prompt); the completion is
            # tokenized on its own so "\n" + "\n" never merges across the boundary.
            prompt = header + _THINK_OPENER
            completion = _CLOSED_THINK[len(_THINK_OPENER):] + content + _IM_END + "\n"
            inp_prompt = self.processor(text=[prompt], **proc_kwargs)
            comp = torch.tensor([self.tok(completion, add_special_tokens=False)["input_ids"]],
                                dtype=inp_prompt["input_ids"].dtype)
            full_ids = torch.cat([inp_prompt["input_ids"], comp], dim=1)[0]
            inp_full = inp_prompt
        else:
            # The closed think scaffold stays in the masked prompt.
            prompt = header + _CLOSED_THINK
            inp_prompt = self.processor(text=[prompt], **proc_kwargs)
            inp_full = self.processor(text=[prompt + content + _IM_END + "\n"], **proc_kwargs)
            full_ids = inp_full["input_ids"][0]
        plen = int(inp_prompt["input_ids"].shape[1])
        if not torch.equal(full_ids[:plen], inp_prompt["input_ids"][0]):
            raise ValueError("prompt is not a token prefix of the full render")
        if has_block and int((full_ids == self.latent_id).sum()) != self.n:
            raise ValueError("unexpected number of latent query slots")
        labels = full_ids.clone()
        labels[:plen] = -100
        labels[full_ids == self.latent_id] = -100
        if self.markers:
            first = _assistant_marker_index((full_ids == self.open_id).nonzero(as_tuple=True)[0],
                                            (full_ids == self.close_id).nonzero(as_tuple=True)[0], self.n)
            if first is not None:
                labels[first] = full_ids[first]
        out = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "mm_token_type_ids": media.mm_token_types(full_ids, self.tok),
            "labels": labels,
            "target_pixel_values": target.unsqueeze(0).contiguous().clone() if has_block
            else torch.zeros(0, 3, self.res, self.res),
        }
        for key in ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"):
            if key in inp_full:
                out[key] = inp_full[key]
        if has_block and source is not None:
            out["source_pixel_values"] = source.unsqueeze(0).contiguous().clone()
        return out


class ImagineCollator:
    """Right-pads one micro-batch (a single sample per device: one modality per step) and concatenates the
    per-block targets / sources in (sample, block) order."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        L = max(b["input_ids"].shape[0] for b in batch)

        def pad(key, value):
            return torch.stack([F.pad(b[key], (0, L - b[key].shape[0]), value=value) for b in batch])

        out = {"input_ids": pad("input_ids", self.pad), "attention_mask": pad("attention_mask", 0),
               "mm_token_type_ids": pad("mm_token_type_ids", 0), "labels": pad("labels", -100),
               "target_pixel_values": torch.cat([b["target_pixel_values"] for b in batch])}
        for key in ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw", "source_pixel_values"):
            if key in batch[0]:
                out[key] = torch.cat([b[key] for b in batch])
        return out
