"""Validate vocabulary-baked query IDs, marker IDs, and routing compatibility."""

from __future__ import annotations

import json
import os

from lmms_eval.models.model_utils.imagine.format import (
    IMAGINE_CLOSE,
    IMAGINE_OPEN,
    query_token,
    QUERY_TOKEN_FORMAT,
    TOOL_CALL_TOKENS,
)
from loguru import logger as eval_logger

# The config.json keys the bake writes for the query tokens.
CFG_QUERY_START = "latent_query_start_token_id"
CFG_NUM_QUERIES = "num_latent_queries"
CFG_OPEN_ID = "imagine_start_token_id"
CFG_CLOSE_ID = "imagine_end_token_id"
CFG_MARKERS_EMITTABLE = "imagine_markers_emittable"


class BakeConfigMixin:
    """Construction-time checks of ``Qwen3_5_ImagineVLLM``, which sets ``_bake`` and the route."""

    # ---- construction ---------------------------------------------------------------------

    @staticmethod
    def _load_bake_config(pretrained: str) -> dict:
        """Read exported query and marker metadata from config.json."""
        path = os.path.join(pretrained, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        if CFG_QUERY_START in cfg:
            return {
                **cfg,
                "query_start": int(cfg[CFG_QUERY_START]),
                "num_queries": int(cfg[CFG_NUM_QUERIES]),
                "query_token_format": QUERY_TOKEN_FORMAT,
            }
        raise ValueError(
            f"{path} has no imagine token ids. qwen3_5_imagine_vllm needs a VOCAB-BAKED "
            "checkpoint: run `scripts/bake.sh --emit-markers` on the imagine checkpoint and point "
            "`pretrained` at its output, not at the raw checkpoint dir."
        )

    def _require_emittable_markers(self) -> None:
        """Require trained marker output rows for adaptive routing."""
        if self._bake.get(CFG_MARKERS_EMITTABLE):
            return
        if self._force_block or self._mute:
            eval_logger.warning(
                "imagine_vllm: bake has imagine_markers_emittable=false, but route={} never reads "
                "the marker logits.",
                self._route_name(),
            )
            return
        raise ValueError(
            "config.json says imagine_markers_emittable=false: this bake DROPPED imagine_lm_head, so "
            "<imagine> is scored from untrained rows and the gate is meaningless (it reads "
            "as 0% fire). Re-bake with `scripts/bake.sh --emit-markers`, or run a forced bracket "
            "(force_imagine_block / mute_imagine)."
        )

    def _marker_ids(self) -> tuple[int, int]:
        ids = (
            int(self._bake[CFG_OPEN_ID]),
            int(self._bake[CFG_CLOSE_ID]),
        )
        for tok, want in zip((IMAGINE_OPEN, IMAGINE_CLOSE), ids):
            got = self.tokenizer(tok, add_special_tokens=False)["input_ids"]
            if got != [want]:
                raise ValueError(
                    f"{tok} tokenizes to {got} but config.json says {want}: the tokenizer in this "
                    "dir does not match the baked weights."
                )
        return ids

    def _resolve_query_ids(self) -> list[int]:
        """The query-token ids, each of which MUST be a single in-vocab id at ``q_start+i``.

        They are input-only: their lm_head rows are zeros, and a zero logit is not negligible (one
        exp(0) per query in the denominator), so they are masked out of the output distribution for the whole
        run. Checked against config.json so a tokenizer/weight mismatch is loud rather than a
        silently wrong block.
        """
        q_start = self._bake["query_start"]
        ids: list[int] = []
        for i in range(self.num_queries):
            piece = query_token(i, self._bake["query_token_format"])
            got = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
            if got != [q_start + i]:
                raise ValueError(
                    f"{piece} tokenizes to {got}, expected [{q_start + i}] from config.json."
                )
            ids.append(q_start + i)
        return ids

    def _resolve_tool_ids(self) -> list[int]:
        """``<tool_call>`` is a base-Qwen tool-use token with no executor here. It is suppressed
        because greedy decode otherwise falls into ``<box>``/``<tool_call>`` loops that run to the
        token cap."""
        out: list[int] = []
        for s in TOOL_CALL_TOKENS:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                out.append(int(tid))
        return out

    @staticmethod
    def _parse_threshold(value: object) -> float | None:
        if value is None or str(value).strip().lower() in ("", "none"):
            return None
        return float(str(value))

    def _route_name(self) -> str:
        if self._force_block:
            return "force(always-fire)"
        if self._mute:
            return "mute(never-fire)"
        return (
            "adaptive-threshold" if self._threshold is not None else "adaptive-argmax"
        )
