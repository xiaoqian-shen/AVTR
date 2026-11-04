"""Compute content advantages, gate targets, and RL loss weights.

Grounding uses Dr.GRPO; gate-learning content advantages are centered per route.
Gate targets use a softmax over route rewards.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from .routes import ROUTES

THINK_GATE_SKIPS = ("none", "all_correct", "ties")


def _members(keys: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    out: dict[Hashable, list[int]] = {}
    for i, k in enumerate(keys):
        out.setdefault(k, []).append(i)
    return out


def group_advantages(scores: torch.Tensor, keys: Sequence[Hashable], norm_by_std: bool,
                     eps: float = 1e-4) -> torch.Tensor:
    """``R - mean`` over the rows sharing a key, divided by ``std + eps`` (unbiased; a lone row has
    no spread) when ``norm_by_std``."""
    adv = torch.zeros_like(scores)
    for idx in _members(keys).values():
        r = scores[idx]
        a = r - r.mean()
        if norm_by_std:
            a = a / ((r.std() if len(idx) > 1 else r.new_zeros(())) + eps)
        adv[idx] = a
    return adv


def gate_coefficients(scores: torch.Tensor, groups: Sequence[Hashable], routes: Sequence[str],
                      beta: float) -> tuple[torch.Tensor, list[float]]:
    """Per-row gate coefficient ``q(z_i) / n(z_i)``, and the entropy of each group's ``q``."""
    coef = torch.zeros_like(scores)
    entropies = []
    for idx in _members(groups).values():
        by_route: dict[str, list[int]] = {}
        for i in idx:
            by_route.setdefault(routes[i], []).append(i)
        names = [z for z in ROUTES if z in by_route]
        q = torch.softmax(torch.stack([scores[by_route[z]].mean() for z in names]).double() / beta, dim=0)
        entropies.append(float(-(q * q.clamp_min(1e-9).log()).sum()))
        for z, qz in zip(names, q.tolist(), strict=True):
            coef[by_route[z]] = qz / len(by_route[z])
    return coef, entropies


def think_gate_skips(credit: torch.Tensor, groups: Sequence[Hashable], routes: Sequence[str], mode: str,
                     tol: float = 1e-6) -> torch.Tensor:
    """(B,) bool: the rows of the groups whose think gate is left untrained.

    ``all_correct``: every rollout got the same nonzero answer credit. ``ties``: the routes' mean
    credits agree within ``tol`` (all-correct groups included). The routes of such a group differ
    only by their costs, which would pull the think gate toward not reasoning where reasoning
    changes nothing; the visual gate keeps them, as they stop it firing where the block does not help.
    """
    if mode not in THINK_GATE_SKIPS:
        raise ValueError(f"think_gate_skip must be one of {THINK_GATE_SKIPS}, got {mode!r}")
    skip = torch.zeros(len(groups), dtype=torch.bool)
    if mode == "none":
        return skip
    for idx in _members(groups).values():
        values = [float(credit[i]) for i in idx]
        by_route: dict[str, list[float]] = {}
        for i, c in zip(idx, values, strict=True):
            by_route.setdefault(routes[i], []).append(c)
        means = [sum(v) / len(v) for v in by_route.values()]
        uniform = max(values) - min(values) < tol and min(values) > 0
        if uniform or (mode == "ties" and max(means) - min(means) < tol):
            skip[idx] = True
    return skip


def token_advantages(content: torch.Tensor, gate: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
    """(B, T) advantages: the gate coefficient at a row's gates, its content advantage elsewhere."""
    return torch.where(gate_mask.bool(), gate.unsqueeze(-1), content.unsqueeze(-1))


def loss_weights(response_mask: torch.Tensor, gate_mask: torch.Tensor, share: float,
                 mini_batches: Sequence[Hashable]) -> torch.Tensor:
    """(B, T) per-token weights of a token-mean loss.

    A row with n live gates gives each gate ``share / n * T / N`` and its other tokens ``1 - share``,
    where T is the loss-token count of the row's PPO mini-batch (``mini_batches[i]``) and N the
    number of rows with gates in it; a row without (a grounding row) keeps unit weight. The loss
    divides by T, so every gate ends with the coefficient ``share / (n N)`` whatever the length of
    its rollout or of the rows it was batched with, and the gates hold ``share`` of the mini-batch's
    loss weight.
    """
    mask = response_mask.double()
    gates = gate_mask.bool() & response_mask.bool()
    n_gates = gates.sum(-1, keepdim=True).double()
    gated = n_gates > 0
    scale = torch.ones_like(n_gates)
    for idx in _members(mini_batches).values():
        scale[idx] = mask[idx].sum().clamp(min=1.0) / gated[idx].sum().clamp(min=1)
    content = torch.where(gated, torch.full_like(n_gates, 1.0 - share), torch.ones_like(n_gates))
    return torch.where(gates, share / n_gates.clamp(min=1) * scale, content)


def importance_weights(old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, response_mask: torch.Tensor,
                       gate_mask: torch.Tensor, cap: float) -> torch.Tensor:
    """(B, T) truncated importance weights ``pi_old / pi_rollout`` (vLLM samples, the trainer scores):
    each row takes the geometric mean of its token ratios, capped at ``cap``. The injected gates
    were never sampled, so their ratio is neutral: they add nothing to the mean but still count in its length.
    """
    log_ratio = (old_log_probs - rollout_log_probs).clamp(-20.0, 20.0).masked_fill(gate_mask.bool(), 0.0)
    mask = response_mask.to(log_ratio.dtype)
    seq = ((log_ratio * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1.0)).exp()
    return seq.clamp(max=cap).expand_as(log_ratio).clone()
