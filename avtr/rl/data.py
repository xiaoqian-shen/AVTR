"""Convert grounding and gate-learning pools to verl training Parquet.

Usage: python -m avtr.rl.data --grounding POOL --gate POOL --out train.parquet
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import re

import pandas as pd

from ..data.targets import strip_media_tokens, to_bracket_tg
from .rewards import gold_windows

logger = logging.getLogger(__name__)

# Pixel budgets of the reference run: every video gets the same whole-clip budget over frames x height x
# width (the HF video processor's size.longest_edge), each image a fixed pixel range. qwen_vl_utils' video
# ``total_pixels`` counts one frame per temporal patch (a token spans several frames), so it is the
# whole-clip budget divided by the temporal patch size; avtr.rl.runtime lifts its per-frame cap to match.
VIDEO = {"fps": 2.0, "max_frames": 256, "total_pixels": 25_165_824 // 2, "min_pixels": 4096}
IMAGE = {"min_pixels": 262_144, "max_pixels": 1_048_576}
# The imagine directive, which the prompts of the released RL row list carry before the question; the agent loop
# adds it back (avtr.directive), so the parquet holds the bare question.
_DIRECTIVE = re.compile(r"If a visual clue would help you answer, generate it FIRST.*?then think and answer\.\s*",
                        re.DOTALL)


def read_pool(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _remap(path: str, roots: list[tuple[str, str]]) -> str:
    for old, new in roots:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _question(prompt) -> str:
    text = next(m for m in prompt if m["role"] == "user")["content"]
    return _DIRECTIVE.sub("", strip_media_tokens(text)).strip()


def _media(row, column: str) -> list[str]:
    value = row.get(column)
    return [] if value is None else [str(v) for v in value]


def grounding_row(row, roots) -> dict:
    gold = str(row["answer"])
    if not gold_windows(gold):
        raise ValueError(f"unparseable grounding answer {gold!r}")
    videos = [{"video": _remap(v, roots), **VIDEO} for v in _media(row, "videos")]
    # SFT and the eval phrase the grounding answer as "[start, end]"
    question = to_bracket_tg(_question(row["prompt"]))
    return {"prompt": [{"role": "user", "content": "<video>" * len(videos) + question}], "images": [],
            "videos": videos, "data_source": str(row["data_source"]), "reward_model": {"ground_truth": gold},
            "extra_info": {"task": "grounding"}}


def gate_row(row, roots) -> dict:
    images = [{"image": _remap(p, roots), **IMAGE} for p in _media(row, "images")]
    videos = [{"video": _remap(v, roots), **VIDEO} for v in _media(row, "videos")]
    head = "<image>" * len(images) + "<video>" * len(videos)
    return {"prompt": [{"role": "user", "content": head + _question(row["prompt"])}], "images": images,
            "videos": videos, "data_source": str(row["data_source"]),
            "reward_model": {"ground_truth": str(row["answer"]).strip()}, "extra_info": {"task": "gate"}}


def mix(grounding: list[dict], gate: list[dict], gate_fraction: float) -> list[dict]:
    """Both pools, the one short of its share cycled (whole passes, then a prefix) until gate rows make up
    ``gate_fraction`` of the result. An even split is the reference run's equal mix of its pools."""
    if not 0.0 < gate_fraction < 1.0:
        raise ValueError(f"--gate-fraction must lie in (0, 1), got {gate_fraction}")
    if not grounding or not gate:
        return [dict(r) for r in grounding + gate]
    total = max(len(grounding) / (1 - gate_fraction), len(gate) / gate_fraction)
    return ([dict(grounding[i % len(grounding)]) for i in range(round(total * (1 - gate_fraction)))]
            + [dict(gate[i % len(gate)]) for i in range(round(total * gate_fraction))])


def build(grounding: pd.DataFrame, gate: pd.DataFrame, gate_fraction: float, roots, seed: int) -> pd.DataFrame:
    rows = mix([grounding_row(r, roots) for _, r in grounding.iterrows()],
               [gate_row(r, roots) for _, r in gate.iterrows()], gate_fraction)
    random.Random(seed).shuffle(rows)
    for i, row in enumerate(rows):
        row["extra_info"] = {**row["extra_info"], "index": i}
        row["agent_name"] = "imagine_route"
    n_gate = sum(row["extra_info"]["task"] == "gate" for row in rows)
    logger.info("%d grounding + %d gate rows -> %d + %d rows", len(grounding), len(gate), len(rows) - n_gate, n_gate)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--grounding", required=True, help="grounding pool: a parquet file or a dir of them")
    p.add_argument("--gate", required=True, help="gate pool: a parquet file or a dir of them")
    p.add_argument("--out", required=True)
    p.add_argument("--gate-fraction", type=float, default=0.5, help="share of gate rows after cycling")
    p.add_argument("--media-root", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite media paths starting with OLD to start with NEW (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    roots = [tuple(r.split("=", 1)) for r in a.media_root]
    df = build(read_pool(a.grounding), read_pool(a.gate), a.gate_fraction, roots, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
