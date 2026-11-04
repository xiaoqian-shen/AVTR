"""Extract training-format query and marker weights from an exported AVTR checkpoint."""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import shutil

import torch
from safetensors import safe_open

from .. import fileio
from ..model import IMAGINE_CLOSE, IMAGINE_OPEN
from .bake import publish, staging_dir

logger = logging.getLogger(__name__)


def _connector_state(path: str, width: int) -> dict:
    """``connector.*`` of a training checkpoint-N dir or its ``latent_query.pt`` (local or fsspec URL)."""
    if not path.rstrip("/").endswith(".pt"):
        path = f"{path.rstrip('/')}/latent_query.pt"
    with fileio.open_file(path, "rb") as fh:
        state = torch.load(io.BytesIO(fh.read()), map_location="cpu")
    conn = {k: v for k, v in state.items() if k.startswith("connector.")}
    if not conn:
        raise ValueError(f"{path} has no connector.* tensors")
    in_proj = conn.get("connector.in_proj.weight")
    if in_proj is not None and in_proj.shape[1] != width:
        raise ValueError(f"{path}: the connector reads width {in_proj.shape[1]}, the bake's queries are {width} wide")
    return conn


def unbake(ckpt: str, out: str, work: str, connector_from: str | None = None) -> None:
    from transformers import AutoTokenizer

    src = fileio.local_path(ckpt, cache_dir=os.path.join(work, "src"))
    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)
    if "latent_query_start_token_id" not in cfg:
        raise ValueError(f"{ckpt} is not a vocab-baked checkpoint (config.json has no latent_query_start_token_id)")
    # The marker rows of a bake without --emit-markers are the base lm_head rows, not imagine_lm_head.
    if cfg.get("imagine_markers_emittable") is False:
        raise ValueError(f"{ckpt} was baked without --emit-markers: its marker output rows are not imagine_lm_head")
    q_start, n = int(cfg["latent_query_start_token_id"]), int(cfg.get("num_latent_queries", 64))
    tok = AutoTokenizer.from_pretrained(src)
    open_id, close_id = tok.convert_tokens_to_ids(IMAGINE_OPEN), tok.convert_tokens_to_ids(IMAGINE_CLOSE)

    rows: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
                    s = f.get_slice(key)
                    want = (list(range(q_start, q_start + n)) + [open_id, close_id] if "embed" in key
                            else [open_id, close_id])
                    rows["embed" if "embed" in key else "head"] = torch.stack([s[i:i + 1][0].float() for i in want])
    if set(rows) != {"embed", "head"}:
        raise ValueError(f"{ckpt}: expected embed_tokens.weight and lm_head.weight tensors, found {sorted(rows)}")
    head = rows["head"]
    if float(head[0].norm()) == 0.0:
        raise ValueError("imagine_lm_head OPEN row is zero: the bake ran without --emit-markers")
    state = {"query_tokens": rows["embed"][:n], "imagine_embeds": rows["embed"][n:], "imagine_lm_head": head}
    if connector_from:
        state.update(_connector_state(connector_from, rows["embed"].shape[1]))
    dst = staging_dir(out, work)
    # *.incomplete: the staging dir itself lies inside src when --out is, or is inside, --ckpt.
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.complete", "*.incomplete", "README.md"),
                    dirs_exist_ok=True)
    torch.save(state, os.path.join(dst, "latent_query.pt"))
    publish(dst, out)
    if src != ckpt:
        shutil.rmtree(os.path.join(work, "src"), ignore_errors=True)
    logger.info("unbaked %s -> %s%s", ckpt, out, f" (connector from {connector_from})" if connector_from else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="vocab-baked checkpoint dir (local or fsspec URL)")
    p.add_argument("--out", required=True, help="destination dir (local or fsspec URL)")
    p.add_argument("--connector-from", help="training checkpoint-N dir (or its latent_query.pt) whose connector.* "
                                            "tensors are added, so the output loads with load_checkpoint")
    p.add_argument("--work", default="/tmp/avtr_unbake", help="scratch for remote inputs / --out (emptied after)")
    a = p.parse_args()
    unbake(a.ckpt, a.out, a.work, a.connector_from)


if __name__ == "__main__":
    main()
