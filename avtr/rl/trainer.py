"""verl PPO trainer with AVTR rewards, advantages, and mini-batch ordering."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.v1 import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import response_to_nested

from .advantages import (
    THINK_GATE_SKIPS,
    gate_coefficients,
    group_advantages,
    importance_weights,
    loss_weights,
    think_gate_skips,
    token_advantages,
)
from .rewards import RouteCost, route_shaping
from .routes import GATE_THINK, ROUTES, THINKS


@dataclass(frozen=True)
class StageConfig:
    distill_beta: float
    gate_loss_share: float
    grounding_loss_weight: float
    grounding_norm_by_std: bool
    think_gate_skip: str
    is_cap: float
    cost: RouteCost

    def __post_init__(self):
        if self.think_gate_skip not in THINK_GATE_SKIPS:
            raise ValueError(f"avtr.think_gate_skip must be one of {THINK_GATE_SKIPS}, got {self.think_gate_skip!r}")

    @classmethod
    def from_config(cls, cfg) -> StageConfig:
        return cls(cfg.distill_beta, cfg.gate_loss_share, cfg.grounding_loss_weight, cfg.grounding_norm_by_std,
                   cfg.think_gate_skip, cfg.is_cap,
                   RouteCost(cfg.think_cost, cfg.block_cost, cfg.noblock_bonus, cfg.unclosed_min_chars))


def mini_batch_ids(n_rows: int, dp_size: int, mini_batch_size: int, shuffle: bool, seed: int) -> list[int]:
    """The PPO mini-batch each row of a batch is trained in, as verl's actor update forms them.

    ``update_actor`` cuts the batch, in order, into ``dp_size`` equal shards, one per data-parallel
    rank; each rank walks its shard in mini-batches of ``mini_batch_size / dp_size`` rows with
    ``tensordict_utils.make_iterator`` (shuffled with seed ``seed + rank`` when ``shuffle``), and the
    ranks' k-th mini-batches make up the k-th optimizer step.
    """
    if n_rows % dp_size or mini_batch_size % dp_size:
        raise ValueError(f"{n_rows} rows in mini-batches of {mini_batch_size} do not split over {dp_size} ranks")
    shard = n_rows // dp_size
    ids = [0] * n_rows
    for rank in range(dp_size):
        rows = TensorDict({"row": torch.arange(shard)}, batch_size=shard)
        batches = tu.make_iterator(rows, mini_batch_size // dp_size, epochs=1, seed=seed + rank,
                                   dataloader_kwargs={"shuffle": shuffle})
        for k, mini_batch in enumerate(batches):
            for j in mini_batch["row"].tolist():
                ids[rank * shard + j] = k
    return ids


def compute_batch(scores: torch.Tensor, response_mask: torch.Tensor, old_log_probs: torch.Tensor,
                  rollout_log_probs: torch.Tensor, uids: Sequence[str], extras: Sequence[dict | None],
                  mini_batches: Sequence[Hashable], stage: StageConfig
                  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """(B, T) advantages (loss weights folded in) and importance weights of one batch, plus metrics.
    ``scores`` is each row's rollout score (IoU or answer correctness); ``extras[i]`` is row i's
    agent-loop record, None for a padding row; ``mini_batches[i]`` is the PPO mini-batch row i is
    trained in."""
    scores = scores.double()
    rows = [i for i, e in enumerate(extras) if e is not None]
    gate = [i for i in rows if extras[i]["task"] == "gate"]
    grounding = [i for i in rows if extras[i]["task"] == "grounding"]
    content = torch.zeros_like(scores)
    coef = torch.zeros_like(scores)
    gate_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    think_off = torch.zeros_like(response_mask, dtype=torch.bool)
    metrics: dict[str, float] = {}

    if grounding:
        r = scores[grounding]
        content[grounding] = stage.grounding_loss_weight * group_advantages(
            r, [uids[i] for i in grounding], norm_by_std=stage.grounding_norm_by_std)
        metrics["avtr/grounding/iou"] = r.mean().item()
    if gate:
        info = [extras[i]["reward_extra_info"] for i in gate]
        acc = scores[gate]
        fired = [bool(x["fired"]) for x in info]
        thinking = [bool(x["thinking"]) for x in info]
        groups = [uids[i] for i in gate]
        routes = [extras[i]["route"] for i in gate]
        total = acc + torch.tensor(route_shaping(acc.tolist(), fired, thinking, groups, stage.cost), dtype=acc.dtype)
        content[gate] = group_advantages(total, list(zip(groups, routes, strict=True)), norm_by_std=False)
        gate_coef, entropies = gate_coefficients(total, groups, routes, stage.distill_beta)
        coef[gate] = gate_coef
        skip = think_gate_skips(acc, groups, routes, stage.think_gate_skip)
        for k, i in enumerate(gate):
            gates = extras[i]["gates"]
            gate_mask[i, gates] = True
            if skip[k]:
                kinds = extras[i]["gate_kinds"]
                think_off[i, [p for p, kind in zip(gates, kinds, strict=True) if kind == GATE_THINK]] = True
        think_rows = [k for k, z in enumerate(routes) if z in THINKS]
        metrics.update({
            "avtr/gate/acc": acc.mean().item(),
            "avtr/gate/reward": total.mean().item(),
            "avtr/gate/distill_target_entropy": sum(entropies) / len(entropies),
            "avtr/gate/think_skipped": float(len({g for g, s in zip(groups, skip.tolist(), strict=True) if s})),
            "avtr/gate/fired_and_thinking": sum(f and t for f, t in zip(fired, thinking, strict=True)) / len(gate),
            "avtr/gate/empty_think_rate": sum(not thinking[k] for k in think_rows) / max(len(think_rows), 1),
            "avtr/gate/think_chars": sum(info[k]["think_chars"] for k in think_rows) / max(len(think_rows), 1),
            "avtr/gate/truncated_rate": sum(info[k]["truncated"] for k in think_rows) / max(len(think_rows), 1),
            "avtr/gate/think_resample_rate": sum(info[k]["resampled"] > 0 for k in think_rows)
            / max(len(think_rows), 1),
            "avtr/gate/think_resample_exhausted": sum(info[k]["resample_exhausted"] for k in think_rows)
            / max(len(think_rows), 1),
            "avtr/gate/force_closed_rate": sum(x["force_closed"] for x in info) / len(gate),
            "avtr/gate/scored_row_frac": (content[gate] != 0).double().mean().item(),
        })
        for z in ROUTES:
            idx = [k for k, r in enumerate(routes) if r == z]
            if idx:
                metrics[f"avtr/gate/acc_{z}"] = acc[idx].mean().item()

    advantages = token_advantages(content, coef, gate_mask).masked_fill(think_off, 0.0) * response_mask.double()
    mass = advantages.abs().sum(-1)
    metrics["avtr/grounding/mass_frac"] = (mass[grounding].sum() / mass.sum()).item() if mass.sum() > 0 else 0.0
    advantages = advantages * loss_weights(response_mask, gate_mask, stage.gate_loss_share, mini_batches)
    weights = importance_weights(old_log_probs, rollout_log_probs, response_mask, gate_mask, stage.is_cap)
    padding = [i for i, e in enumerate(extras) if e is None]
    weights[padding] = 1.0
    live = response_mask.bool()
    if live.any():
        metrics["avtr/is_weight_mean"] = weights[live].mean().item()
        metrics["avtr/is_weight_capped_frac"] = (weights[live] >= stage.is_cap).double().mean().item()
    return advantages.float(), weights.float(), metrics


def _values(column) -> list:
    values = column.tolist() if hasattr(column, "tolist") else list(column)
    return [getattr(v, "data", v) for v in values]


class ImagineRLTrainer(PPOTrainerSync):
    def __init__(self, config):
        actor = config.actor_rollout_ref.actor
        if actor.loss_agg_mode != "token-mean":
            raise ValueError("the RL stage's loss weights are token-mean weights: set "
                             "actor_rollout_ref.actor.loss_agg_mode=token-mean")
        if actor.ppo_epochs > 1 and actor.shuffle:
            raise ValueError("the gate weights follow one mini-batch split per step, which actor.shuffle redraws every "
                             "epoch: with ppo_epochs > 1 set actor_rollout_ref.actor.shuffle=false")
        super().__init__(config)

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """The reference run's split of the step's rows into PPO mini-batches, in place of verl's length balancing.

        The rows (in the reference run's order, ``sampler.py``) are permuted with a per-step seed derived from
        ``avtr.shuffle_seed`` and the padding verl needs is appended, as the reference run appends its inert
        rows after the shuffle. The reference run then gave each of its ``avtr.shuffle_ranks`` trainer GPUs a
        contiguous slice and cut every slice into consecutive mini-batch pieces. verl splits the same way
        (``actor.shuffle`` off), so the rows are laid out to give every optimizer step the reference run's rows
        whatever the GPU count here.
        """
        wg = self.actor_rollout_wg
        if "actor" not in wg._dispatch_info:
            wg._dispatch_info["actor"] = wg._query_dispatch_info("actor")
        dp = max(wg._dispatch_info["actor"]) + 1
        generator = torch.Generator().manual_seed(self.config.avtr.shuffle_seed + (self.global_steps - 1) * 1_000_003)
        batch.reorder(torch.randperm(len(batch), generator=generator).tolist())
        multiple = self._get_required_batch_multiple(dp)
        batch = upsample_batch_to_divisible_size(batch, multiple, self.tokenizer.eos_token_id)
        rows, ranks = len(batch), self.config.avtr.shuffle_ranks
        step = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        if step % ranks or rows % step:
            raise ValueError(f"PPO mini-batches of {step} rows cannot be laid out over avtr.shuffle_ranks={ranks}")
        if dp != ranks:  # position (rank, mini-batch, offset) of the reference layout -> verl's rank-major order
            layout = torch.arange(rows).view(ranks, rows // step, step // ranks).transpose(0, 1)
            batch.reorder(layout.reshape(rows // step, dp, step // dp).transpose(0, 1).reshape(-1).tolist())
        return batch

    def _mini_batch_ids(self, n_rows: int) -> list[int]:
        wg, actor = self.actor_rollout_wg, self.config.actor_rollout_ref.actor
        if "actor" not in wg._dispatch_info:
            wg._dispatch_info["actor"] = wg._query_dispatch_info("actor")
        return mini_batch_ids(n_rows, max(wg._dispatch_info["actor"]) + 1,
                              actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n, actor.shuffle,
                              actor.data_loader_seed)

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs",
                                              "extra_fields"])
        nested_mask = data["response_mask"]
        padded = data.select("response_mask", "rm_scores", "rollout_log_probs", "old_log_probs").to_padded_tensor()
        extras = [None if tag.get("is_padding", False) else e
                  for tag, e in zip(batch.tags, _values(data["extra_fields"]), strict=True)]
        advantages, weights, batch_metrics = compute_batch(
            padded["rm_scores"].sum(-1), padded["response_mask"], padded["old_log_probs"],
            padded["rollout_log_probs"], [str(u) for u in _values(data["uid"])], extras,
            self._mini_batch_ids(len(batch)), StageConfig.from_config(self.config.avtr))
        metrics.update(batch_metrics)
        output = TensorDict({"advantages": response_to_nested(advantages, nested_mask),
                             "returns": response_to_nested(advantages, nested_mask),
                             "rollout_is_weights": response_to_nested(weights, nested_mask)}, batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
