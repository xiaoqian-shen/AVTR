"""Normalize visual and textual gate probabilities over their two-token pairs."""

from __future__ import annotations

import json
import os
from itertools import accumulate

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from .routes import GATE_THINK, GATE_VISUAL, RouteTokens

# Finite, so that the entropy's p * logit vanishes at a masked entry.
MASKED = -1.0e9


def gate_pairs(tokens: RouteTokens) -> dict[int, tuple[int, int]]:
    """Gate kind (``GATE_THINK`` / ``GATE_VISUAL``) -> the token pair the gate chooses between."""
    return {kind: tokens.gate_pair(kind) for kind in (GATE_THINK, GATE_VISUAL)}


def _lengths(t: torch.Tensor) -> list[int]:
    # The TransferQueue returns a field dense only when all of its rows have one length.
    return t.offsets().diff().tolist() if t.is_nested else [t.shape[-1]] * t.shape[0]


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.values() if t.is_nested else t.reshape(-1)


def live_gates(micro_batch: TensorDict,
               pairs: dict[int, tuple[int, int]]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Row, position and pair of every gate of a micro-batch that is in the loss. ``position`` indexes the
    row's prompt + response and holds the logits that predict the gate token.

    The gates are the agent loop's ``extra_fields["gates"]`` / ``["gate_kinds"]``, kept where the response
    mask is set: a padding row carries a copy of a real row's record over a response that is out of the loss.
    """
    found = []
    for i, extra in enumerate(tu.get(micro_batch, "extra_fields") or []):
        extra = getattr(extra, "data", extra) or {}
        offsets, kinds = extra.get("gates") or [], extra.get("gate_kinds") or []
        if len(offsets) != len(kinds):
            raise ValueError(f"row {i}: extra_fields has gates {offsets} but gate_kinds {kinds}")
        found += [(i, k, kind) for k, kind in zip(offsets, kinds, strict=True)]
    if not found:
        return [], [], []
    seq_lens, resp_lens = _lengths(micro_batch["input_ids"]), _lengths(micro_batch["response_mask"])
    seq_start, resp_start = [0, *accumulate(seq_lens)], [0, *accumulate(resp_lens)]
    prompt_lens = [s - r for s, r in zip(seq_lens, resp_lens, strict=True)]
    found = [(i, k, kind) for i, k, kind in found if 0 <= k < resp_lens[i]]
    mask, ids = _flat(micro_batch["response_mask"]), _flat(micro_batch["input_ids"])
    live = mask[torch.tensor([resp_start[i] + k for i, k, _ in found], dtype=torch.long, device=mask.device)]
    drawn = ids[torch.tensor([seq_start[i] + prompt_lens[i] + k for i, k, _ in found], dtype=torch.long,
                             device=ids.device)]
    rows, positions, chosen = [], [], []
    for (i, k, kind), on, token in zip(found, live.tolist(), drawn.tolist(), strict=True):
        if not on:
            continue
        pair = pairs[kind]
        if token not in pair:
            raise ValueError(f"row {i}: the gate at response offset {k} is token {token}, outside its pair {pair}")
        rows.append(i)
        positions.append(prompt_lens[i] + k - 1)
        chosen.append(pair)
    return rows, positions, chosen


@torch.no_grad()
def restrict_to_pairs_(logits: torch.Tensor, index: tuple[torch.Tensor, ...], pairs: torch.Tensor) -> None:
    """In place: in each logits row ``logits[index][j]``, every entry but the two ids ``pairs[j]`` becomes
    ``MASKED``."""
    rows = logits[index]
    kept = rows.gather(-1, pairs)
    logits[index] = rows.fill_(MASKED).scatter_(-1, pairs, kept)


class GateLogProbEngine(FSDPEngineWithLMHead):
    """verl's FSDP language-model engine with every live gate scored over its pair (module docstring)."""

    _pairs: dict[int, tuple[int, int]] | None = None

    def gate_pairs(self) -> dict[int, tuple[int, int]]:
        """The pairs of the checkpoint the engine trains, resolved as the agent loop resolves them."""
        if self._pairs is None:
            with open(os.path.join(self.model_config.local_path, "config.json")) as fh:
                self._pairs = gate_pairs(RouteTokens.from_pretrained(self.model_config.tokenizer, json.load(fh)))
        return self._pairs

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        rows, positions, pairs = live_gates(micro_batch, self.gate_pairs())
        if rows:
            if (tu.get_non_tensor_data(micro_batch, "use_fused_kernels", False) or self.use_ulysses_sp
                    or isinstance(output.logits, DTensor)):
                raise NotImplementedError("scoring the gates over their pair needs the micro-batch's full logits on "
                                          "each rank: no fused kernels, sequence parallelism or vocabulary sharding")
            logits = output.logits
            rows_t, positions_t = (torch.tensor(x, dtype=torch.long, device=logits.device) for x in (rows, positions))
            if tu.get_non_tensor_data(micro_batch, "use_remove_padding", True):
                starts = micro_batch["input_ids"].offsets().to(logits.device)
                index = (torch.zeros_like(rows_t), starts[rows_t] + positions_t)  # logits (1, packed tokens, V)
            else:
                index = (rows_t, positions_t)  # logits (rows, padded length, V)
            restrict_to_pairs_(logits, index, torch.tensor(pairs, dtype=torch.long, device=logits.device))
        return super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)


def _install() -> None:
    # EngineRegistry.register refuses a taken key, so the registered FSDP engine is replaced in place.
    swapped = 0
    for backend in ("fsdp", "fsdp2"):
        slots = EngineRegistry._engines["language_model"][backend]
        for key, cls in slots.items():
            if cls is FSDPEngineWithLMHead:
                slots[key] = GateLogProbEngine
                swapped += 1
    if not swapped:
        raise RuntimeError("verl registers no FSDPEngineWithLMHead for GateLogProbEngine to replace")


_install()
