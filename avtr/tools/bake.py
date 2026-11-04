"""Export stage-2 queries and markers into a Hugging Face model vocabulary.

Use --emit-markers to preserve learned marker logits for adaptive routing.
Query output rows are suppressed; inference no longer needs the connector or SANA.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .. import fileio
from ..model import IMAGINE_CLOSE, IMAGINE_OPEN

logger = logging.getLogger(__name__)

_COPY = ("config.json", "generation_config.json", "chat_template.jinja", "processor_config.json",
         "preprocessor_config.json", "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.json", "merges.txt")
_HUB_ID = re.compile(r"[\w-][\w.-]*/[\w-][\w.-]*")


def _one(keys, suffix: str) -> str:
    hits = [k for k in keys if k.endswith(suffix)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one *{suffix}, got {hits}")
    return hits[0]


def _grow(t: torch.Tensor, rows: int) -> torch.Tensor:
    t = t.float()
    return torch.cat([t, t.new_zeros(rows - t.shape[0], t.shape[1])]) if rows > t.shape[0] else t


def _fetch(ckpt: str, dst: str) -> str:
    """``ckpt`` itself when local, else the files a bake reads, downloaded to ``dst``."""
    if not fileio.is_remote(ckpt):
        return ckpt
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for name in fileio.listdir(ckpt):
        if name in _COPY or name == "latent_query.pt" or name.endswith(".safetensors"):
            fileio.copy(f"{ckpt.rstrip('/')}/{name}", os.path.join(dst, name))
    return dst


def staging_dir(out: str, work: str) -> str:
    """An empty dir to build ``out`` in. A local ``out`` is staged on its own filesystem so it moves
    into place: inside it when it exists (it may be a mount point), else beside it. A remote ``out``
    is staged in ``work``."""
    if fileio.is_remote(out):
        dst = os.path.join(work, "out")
    elif os.path.isdir(out):
        dst = os.path.join(out, ".incomplete")
    else:
        dst = out.rstrip("/") + ".incomplete"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    return dst


def publish(dst: str, out: str) -> None:
    """Move the built ``dst`` to ``out``. Files of an existing local ``out`` that are not rebuilt stay."""
    if fileio.is_remote(out):
        # The trailing slash copies dst's contents: fsspec would nest dst inside an existing out.
        fileio.copy(dst.rstrip("/") + "/", out)
        shutil.rmtree(dst, ignore_errors=True)
    elif not os.path.exists(out):
        os.replace(dst, out)
    else:
        for name in os.listdir(dst):
            shutil.move(os.path.join(dst, name), os.path.join(out, name))
        os.rmdir(dst)


def _base_model(cfg: dict) -> str | None:
    """The backbone's Hub id if the checkpoint config recorded one (``_name_or_path``), else None.
    A path, local or remote, never goes into the model card."""
    for holder in (cfg, cfg.get("text_config") or {}):
        name = str(holder.get("_name_or_path") or "")
        if _HUB_ID.fullmatch(name) and not os.path.exists(name):
            return name
    return None


def bake(ckpt: str, out: str, work: str, num_queries: int = 64, emit_markers: bool = False) -> None:
    from transformers import AutoTokenizer

    src = _fetch(ckpt, os.path.join(work, "src"))
    if not os.path.isfile(os.path.join(src, "latent_query.pt")):
        raise FileNotFoundError(f"{ckpt} has no latent_query.pt (not a training checkpoint-N dir)")
    state = torch.load(os.path.join(src, "latent_query.pt"), map_location="cpu")
    if "imagine_embeds" not in state:
        raise ValueError(f"{ckpt}: latent_query.pt has no imagine_embeds; bake needs a stage-2 (markers: true) "
                         "checkpoint")
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        raise ValueError(f"{ckpt} has no backbone *.safetensors; bake needs a stage-2 checkpoint (the backbone "
                         "is saved only when it trains)")
    queries, marker_in = state["query_tokens"].float(), state["imagine_embeds"].float()
    marker_out = state["imagine_lm_head"].float() if emit_markers else None
    if queries.shape[0] != num_queries or marker_in.shape[0] != 2:
        raise ValueError(f"unexpected shapes: query_tokens {tuple(queries.shape)}, "
                         f"imagine_embeds {tuple(marker_in.shape)}")

    dst = staging_dir(out, work)
    for name in _COPY:
        if os.path.exists(os.path.join(src, name)):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    tok = AutoTokenizer.from_pretrained(dst)
    open_id, close_id = tok.convert_tokens_to_ids(IMAGINE_OPEN), tok.convert_tokens_to_ids(IMAGINE_CLOSE)
    if not isinstance(open_id, int) or not isinstance(close_id, int):
        raise ValueError("the checkpoint tokenizer has no imagine markers (not a stage-2 marker checkpoint)")
    new_tokens = [f"<latent_query_{i}>" for i in range(num_queries)]
    q_start = tok.get_vocab().get(new_tokens[0])
    if q_start is None:
        q_start = len(tok)
        tok.add_special_tokens({"additional_special_tokens": new_tokens})
    ids = tok.convert_tokens_to_ids(new_tokens)
    if ids != list(range(q_start, q_start + num_queries)):
        raise ValueError(f"<latent_query_i> ids {ids[0]}..{ids[-1]} are not {q_start}..{q_start + num_queries - 1}")
    tok.save_pretrained(dst)
    for probe in (new_tokens[0], new_tokens[-1], IMAGINE_OPEN, IMAGINE_CLOSE):
        if len(tok(probe, add_special_tokens=False)["input_ids"]) != 1:
            raise ValueError(f"{probe} does not tokenize to one id")

    tensors = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys()})
    embed_key = _one(tensors, "embed_tokens.weight")
    if not any(k.endswith("lm_head.weight") for k in tensors):  # tied checkpoint without a head tensor
        tensors["lm_head.weight"] = tensors[embed_key].clone()
    head_key = _one(tensors, "lm_head.weight")
    rows = max(tensors[embed_key].shape[0], q_start + num_queries)
    emb = _grow(tensors[embed_key], rows)
    emb[q_start:q_start + num_queries] = queries
    emb[open_id], emb[close_id] = marker_in[0], marker_in[1]
    head = _grow(tensors[head_key], rows)  # the head is written untied (see tie_word_embeddings below)
    head[q_start:q_start + num_queries] = 0
    if marker_out is not None:
        head[open_id], head[close_id] = marker_out[0], marker_out[1]
    tensors[embed_key], tensors[head_key] = emb, head
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()},
              os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config", cfg)
    text["vocab_size"] = rows
    if "vocab_size" in cfg:
        cfg["vocab_size"] = rows
    for holder in (cfg, text):
        for k in ("dtype", "torch_dtype"):
            if k in holder:
                holder[k] = "bfloat16"
        if holder.get("tie_word_embeddings"):
            holder["tie_word_embeddings"] = False
    cfg.update(imagine_start_token_id=int(open_id), imagine_end_token_id=int(close_id),
               latent_query_start_token_id=q_start, num_latent_queries=num_queries,
               imagine_markers_emittable=bool(emit_markers))
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    gen_path = os.path.join(dst, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as fh:
            gen = json.load(fh)
    else:
        gen = {k: cfg[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id") if k in cfg}
    suppress = set(gen.get("suppress_tokens") or []) | set(range(q_start, q_start + num_queries))
    suppress = suppress - {open_id, close_id} if emit_markers else suppress | {open_id, close_id}
    gen["suppress_tokens"] = sorted(suppress)
    with open(gen_path, "w") as fh:
        json.dump(gen, fh, indent=2)
    base = _base_model(cfg)
    front = f"base_model: {json.dumps(base)}\n" if base else ""
    with open(os.path.join(dst, "README.md"), "w") as fh:
        fh.write(f"---\n{front}library_name: transformers\n---\n\n"
                 f"Vocab-baked imagine checkpoint: the {num_queries} `<latent_query_i>` input rows hold the learned "
                 "query vectors and the `<imagine>` / `</imagine>` rows the trained marker embeddings. Token ids are "
                 "in `config.json`; the query tokens are suppressed in `generation_config.json`.\n")
    publish(dst, out)
    if src != ckpt:
        shutil.rmtree(src, ignore_errors=True)
    logger.info("baked %s -> %s (q_start=%d open=%d close=%d rows=%d)", ckpt, out, q_start, open_id, close_id, rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="stage-2 checkpoint dir (local or fsspec URL)")
    p.add_argument("--out", required=True, help="destination HF model dir (local or fsspec URL)")
    p.add_argument("--work", default="/tmp/avtr_bake", help="scratch for a remote --ckpt / --out (emptied after)")
    p.add_argument("--num-queries", type=int, default=64)
    p.add_argument("--emit-markers", action="store_true", help="fold imagine_lm_head into the marker output rows")
    a = p.parse_args()
    bake(a.ckpt, a.out, a.work, a.num_queries, a.emit_markers)


if __name__ == "__main__":
    main()
